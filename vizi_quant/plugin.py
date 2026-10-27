#!/usr/bin/env python3
"""vizi-quant: measure every region over every frame, plot the traces, export.

The manifest is declared here, in code.  vizi launches ``vizi_quant_plugin.py``
at the repository root, which runs :func:`main`; ``vizi-plugin.json`` is
generated from these declarations by ``tools/write_manifest.py``.

Threads: the SDK runs the push handlers and commands, which only change state
and return; the sweep runs on the engine's worker (``engine.py``); everything
the panel shows is drawn by one renderer thread here, at most every
``RENDER_EVERY_S``, from whatever the state is when it wakes -- so a sweep
landing numbers forty times a second costs the panel two redraws a second.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

from vizi_plugin_sdk import (
    CommandContext,
    HostError,
    Appearance,
    WidgetSize,
    appearance,
    command,
    on_regions_changed,
    on_shutdown,
    on_view_changed,
    panel,
    panel_opened,
    plugin,
    serve,
    settings_dir,
    widget_geometry,
    widgets as w,
)
import vizi_plugin_sdk

from vizi_quant import derived, export, perf, plot, stats
from vizi_quant.cache import Cache, identity_of
from vizi_quant.engine import Engine

VERSION = "0.1.0"
PANEL = "quant"
TABLE = "regions"
RENDER_EVERY_S = 0.5
PLOT_STATS = stats.INTENSITY
#: The same statistics in the words the panel uses.
STAT_NAMES = ("mean", "bright (85–95th percentile)", "dim (5–15th percentile)", "spread (std)")
MODES = ("raw brightness (F)", "ΔF/F")
BASELINES = ("moving (follows drift)", "fixed frames")

plugin(
    "org.focolab.vizi_quant",
    name="vizi-quant",
    version=VERSION,
    permissions=["data_access"],
    subscriptions=["view", "regions"],
    # Protocol 1.0 hosts are vizi 1.0.0 and later.
    min_vizi="1.0.0",
)


class State:
    """Everything the handlers share.  ``lock`` guards it; the engine has its
    own lock for the regions and traces."""

    def __init__(self):
        self.lock = threading.RLock()
        self.engine: "Engine | None" = None
        self.view = None  # the HostViewState of the open dataset
        self.generation = 0
        self.t = 0
        self.selected = ""  # the region selected in vizi
        #: The regions highlighted here: the rows selected in the table, or
        #: vizi's selection when that changed since.  Selecting rows always
        #: moves it, also when vizi does not take the selection.
        self.chosen: tuple[str, ...] = ()
        #: Regions this panel asked vizi to select, oldest first, whose view
        #: pushes have not come back yet: those pushes are echoes, not the
        #: person choosing in vizi, and leave ``chosen`` alone.
        self.asked: list[str] = []
        #: What the table shows: its keys in order and each row's cells, as
        #: last sent, so a redraw sends only the rows that changed.
        self.table: "dict[str, tuple] | None" = None
        self.table_selection: "tuple[str, ...] | None" = None
        self.table_running = False
        #: The plot image's drawn size (width, height) in pixels and the
        #: screen's pixel ratio, from vizi's geometry push.
        self.plot_size: "tuple[int, int, float]" = (plot.WIDTH_PX, plot.HEIGHT_PX, 1.0)
        #: vizi's panel colours by name, from its appearance push; ``None``
        #: until one arrives, when the plot keeps matplotlib's light defaults.
        self.colors: "dict[str, str] | None" = None
        self.plot_stat = 0
        self.plot_channel = 0
        self.plot_mode = 0
        self.baseline = derived.Baseline()
        self.status = "Open a dataset to measure its regions"
        self.axes: "plot.Axes | None" = None
        self.panel = None
        self.rows: list[str] = []
        self.dff_cache: dict = {}
        self.wake = threading.Event()


S = State()


# -- settings: the person's choices, kept across sessions -------------------------


def _settings_path():
    return settings_dir() / "settings.json"


def load_settings() -> None:
    try:
        saved = json.loads(_settings_path().read_text())
    except Exception:  # noqa: BLE001 -- no settings yet, or unreadable: the defaults stand
        return
    b = saved.get("baseline", {})
    S.baseline = derived.Baseline(
        mode=b.get("mode", derived.SLIDING),
        window=int(b.get("window", 101)),
        percentile=float(b.get("percentile", 10.0)),
        first=int(b.get("first", 0)),
        last=int(b.get("last", 9)),
    )
    S.plot_stat = int(saved.get("plot_stat", 0)) % len(PLOT_STATS)
    S.plot_mode = int(saved.get("plot_mode", 0)) % len(MODES)


def save_settings() -> None:
    b = S.baseline
    doc = {
        "baseline": {"mode": b.mode, "window": b.window, "percentile": b.percentile, "first": b.first, "last": b.last},
        "plot_stat": S.plot_stat,
        "plot_mode": S.plot_mode,
    }
    try:
        _settings_path().write_text(json.dumps(doc, indent=1))
    except Exception:  # noqa: BLE001 -- a setting not kept is not worth failing a command
        pass


# -- the dataset -------------------------------------------------------------------


def say(text: str) -> None:
    with S.lock:
        S.status = text
    S.wake.set()


#: The view and regions pushes arrive on threads of their own, and both may be
#: the first to name a new dataset: opening one is serialized, or the second
#: would see the generation set and the engine not yet made.
_OPENING = threading.Lock()


def ensure_dataset(ctx, generation: int) -> "Engine | None":
    """Make the engine match the open dataset: a new generation is a new
    dataset (even the same file reopened), 0 is none."""
    with _OPENING:
        with S.lock:
            if generation == S.generation:
                return S.engine
            old, S.engine, S.generation, S.view = S.engine, None, generation, None
            S.dff_cache.clear()
            S.asked.clear()
            S.table = S.table_selection = None
        if old is not None:
            old.stop("dataset closed")
            old.close()
        if generation == 0:
            say("Open a dataset to measure its regions")
            return None
        view = ctx.host.view_state()
        cache = Cache(view.file_state_dir, identity_of(view))
        engine = Engine(ctx.host, view, cache, on_progress=S.wake.set, on_status=say)
        with S.lock:
            S.engine, S.view, S.t = engine, view, int(view.t)
            S.plot_channel = min(S.plot_channel, int(view.shape_tczyx[1]) - 1)
    publish_choices(ctx.panel(PANEL), view)
    # Nothing is measured until the person asks; what an earlier session
    # measured shows at once from the cache.
    say("")
    return engine


def publish_choices(p, view) -> None:
    """The panel's dataset-dependent choices: the channel lists."""
    n = int(view.shape_tczyx[1])
    p.set_options("plot_channel", [f"ch{c + 1}" for c in range(n)])
    p.set_value("plot_channel", f"ch{S.plot_channel + 1}")
    p.set_visible("plot_channel", n > 1)  # one channel: nothing to choose


@on_view_changed
def view_changed(ctx, change) -> None:
    engine = ensure_dataset(ctx, int(change.open_generation))
    with S.lock:
        S.t = int(change.t)
        new = change.selected_region_id
        if new != S.selected:
            S.selected = new
            if new and new in S.asked:
                del S.asked[: S.asked.index(new) + 1]  # an echo; older asks are answered too
            else:
                S.asked.clear()
                S.chosen = (new,) if new else ()
    if engine is not None:
        engine.view_moved(int(change.t))
    S.wake.set()


@on_regions_changed
def regions_changed(ctx, change) -> None:
    engine = ensure_dataset(ctx, int(change.open_generation))
    if engine is None:
        return
    deleted = [c.region_id for c in change.changes if c.deleted]
    changed = [c.region_id for c in change.changes if not c.deleted]
    try:
        records = ctx.host.get_rois(region_ids=changed) if changed else ()
    except HostError as exc:
        say(f"Could not read the regions: {exc}")
        return
    engine.set_regions(records, bool(change.reset), deleted)
    S.wake.set()


@panel_opened(PANEL)
def opened(ctx) -> None:
    p = ctx.panel(PANEL)
    look = getattr(ctx, "appearance", None)
    with S.lock:
        S.panel, S.table, S.table_selection = p, None, None
        if look is not None:
            S.colors = dict(look.hex_colors)
        view = S.view
    p.set_value("plot_stat", STAT_NAMES[S.plot_stat])
    p.set_value("plot_mode", MODES[S.plot_mode])
    b = S.baseline
    p.set_value("baseline", BASELINES[0 if b.mode == derived.SLIDING else 1])
    p.set_value("window", str(b.window))
    p.set_value("percentile", f"{b.percentile:g}")
    p.set_value("range", f"{b.first + 1}-{b.last + 1}")  # shown from 1, as vizi's T is
    if view is not None:
        publish_choices(p, view)
    S.wake.set()


@on_shutdown
def shutting_down(notice) -> None:
    with S.lock:
        engine = S.engine
    if engine is not None:
        engine.close()


# -- the renderer --------------------------------------------------------------------


def _series(regions, c: int, k: int, mode: int):
    out = []
    for r in regions:
        tr = r.trace
        f = np.where(tr.value_state[c] == 1, tr.values[c, :, k], np.nan)
        if mode == 1:
            key = (r.region_id, r.content_hash, c, k)
            hit = S.dff_cache.get(key)
            if hit is None or hit[0] != (tr.version, S.baseline):
                hit = ((tr.version, S.baseline), derived.dff(f, S.baseline))
                S.dff_cache[key] = hit
            f = hit[1]
        out.append((r.region_id, r.name, f))
    return out


def _cell(values, t: int) -> "float | None":
    """A trace's value on frame ``t``, or a blank cell where there is none."""
    v = values[t] if t < len(values) else np.nan
    return float(v) if np.isfinite(v) else None


def last_measured(series) -> int:
    """The last frame, 1-based, on which any region has a value: where the
    plot's axis ends while a run is going."""
    last = 0
    for _, _, values in series:
        done = np.flatnonzero(np.isfinite(values))
        if done.size:
            last = max(last, int(done[-1]) + 1)
    return max(last, 1)


def send_table(p, rows: dict, running: bool) -> None:
    """Bring the table to ``rows``: whole when its set of regions changes or a
    run starts or ends, otherwise only the rows whose cells changed."""
    with S.lock:
        last, was_running = S.table, S.table_running
        S.table, S.table_running = dict(rows), running
    if last is None or list(last) != list(rows) or was_running != running:
        p.set_table(TABLE, list(rows.items()))
        return
    changed = [(key, cells) for key, cells in rows.items() if last.get(key) != cells]
    if changed:
        p.patch_table(TABLE, upsert=changed)


def progress_text(done: int, total: int, running: bool, regions: bool = True) -> str:
    """The one line beside Measure, in frames: a frame counts once every
    region is measured on it."""
    if not regions:
        return "No regions to measure"
    if running:
        return f"Measuring frame {done:,} / {total:,}"
    if total == 1:
        return "Measured" if done else "Not measured yet"
    if total and done == total:
        return f"All {total:,} frames measured"
    return f"{done:,} / {total:,} frames measured"


def show_baseline(p, mode: int, b: derived.Baseline) -> None:
    """The ΔF/F baseline group shows only while ΔF/F is shown, and of its
    boxes only the ones the chosen method uses."""
    dff = mode == 1
    moving = b.mode == derived.SLIDING
    p.set_visible("baseline_row", dff)
    p.set_visible("moving_row", dff and moving)
    p.set_visible("fixed_row", dff and not moving)


def render_once() -> None:
    with S.lock:
        p, engine, t, chosen, size, colors = S.panel, S.engine, S.t, S.chosen, S.plot_size, S.colors
        c, k, mode, status, b = S.plot_channel, S.plot_stat, S.plot_mode, S.status, S.baseline
    if p is None:
        return
    p.set_value("status", status)
    show_baseline(p, mode, b)
    if engine is None:
        if S.table:
            p.set_table(TABLE, [])
        S.table = {}
        p.set_value("progress", "")
        return
    regions, channels = engine.snapshot()
    done, total = engine.frame_progress()
    running = engine.running
    p.set_value("progress", progress_text(done, total, running, bool(regions)))
    # One button: Measure, which is Stop while it runs.
    p.set_visible("run", not running)
    p.set_visible("stop", running)
    raw = _series(regions, c, k, 0)
    dff = _series(regions, c, k, 1)
    rows = {
        r.region_id: (r.name or r.region_id, _cell(f, t), _cell(d, t))
        for r, (_, _, f), (_, _, d) in zip(regions, raw, dff)
    }
    send_table(p, rows, running)
    keys = tuple(k for k in chosen if k in rows)
    if keys != S.table_selection:
        p.set_value(TABLE, list(keys))
        S.table_selection = keys
    label = PLOT_STATS[k] + (f", ch{c + 1}" if engine.n_channels > 1 else "")
    png, axes = plot.render(
        dff if mode == 1 else raw,
        engine.frames,
        t,
        selected=set(keys),
        ylabel=("ΔF/F " if mode == 1 else "") + label,
        size=size,
        upto=last_measured(raw) if running else None,
        colors=colors,
    )
    with S.lock:
        S.axes = axes
    p.push_image("plot", png)


def _renderer() -> None:
    while True:
        S.wake.wait()
        S.wake.clear()
        try:
            render_once()
        except Exception as exc:  # noqa: BLE001 -- the renderer outlives one bad frame
            with S.lock:
                S.status = f"Drawing failed: {exc}"
        time.sleep(RENDER_EVERY_S)


threading.Thread(target=_renderer, daemon=True, name="vizi-quant-render").start()


# -- commands -------------------------------------------------------------------------


def _engine_or_say() -> "Engine | None":
    with S.lock:
        engine = S.engine
    if engine is None:
        say("No dataset is open")
    return engine


@command("quant.run", title="Measure every region on every frame", menu=None)
def run(ctx: CommandContext) -> str:
    engine = _engine_or_say()
    if engine is None:
        return "no dataset"
    if engine.running:
        return "already measuring"
    engine.start(lambda title: ctx.start_task(title))
    say("")
    return "measuring"


@command("quant.stop", title="Stop measuring", menu=None)
def stop(ctx: CommandContext) -> str:
    engine = _engine_or_say()
    if engine is not None and engine.running:
        engine.stop("stopped by the person", disarm=True)
        say("Stopped; what was measured is kept")
    return "stopped"


def _dropdown_index(ctx, options) -> int:
    value = ctx.args.get("value")
    if value in options:
        return options.index(value)
    return int(ctx.args.get("index", 0))


@command("quant.plot_stat", title="Statistic to plot", menu=None)
def choose_plot_stat(ctx: CommandContext) -> None:
    with S.lock:
        S.plot_stat = _dropdown_index(ctx, STAT_NAMES)
    save_settings()
    S.wake.set()


@command("quant.plot_channel", title="Channel to plot", menu=None)
def choose_plot_channel(ctx: CommandContext) -> None:
    value = str(ctx.args.get("value", "ch1"))
    with S.lock:
        S.plot_channel = int(value[2:]) - 1 if value.startswith("ch") else int(ctx.args.get("index", 0))
    S.wake.set()


@command("quant.plot_mode", title="Plot raw F or ΔF/F", menu=None)
def choose_plot_mode(ctx: CommandContext) -> None:
    with S.lock:
        S.plot_mode = _dropdown_index(ctx, MODES)
    save_settings()
    S.wake.set()


def _set_baseline(**changes) -> None:
    with S.lock:
        b = S.baseline
        S.baseline = derived.Baseline(**{**b.__dict__, **changes})
    save_settings()
    S.wake.set()


@command("quant.baseline", title="ΔF/F baseline", menu=None)
def choose_baseline(ctx: CommandContext) -> None:
    _set_baseline(mode=derived.SLIDING if _dropdown_index(ctx, BASELINES) == 0 else derived.FIXED)


@command("quant.window", title="Baseline window (frames)", menu=None)
def set_window(ctx: CommandContext) -> None:
    text = str(ctx.args.get("value", "")).strip()
    if not text.isdigit() or int(text) < 1:
        say(f"Window {text!r} is not a positive whole number of frames")
        raise ValueError("window must be a positive whole number")
    _set_baseline(window=int(text))


@command("quant.percentile", title="Baseline percentile", menu=None)
def set_percentile(ctx: CommandContext) -> None:
    text = str(ctx.args.get("value", "")).strip()
    try:
        value = float(text)
        if not 0 <= value <= 100:
            raise ValueError
    except ValueError:
        say(f"Percentile {text!r} is not a number from 0 to 100")
        raise ValueError("percentile must be 0..100") from None
    _set_baseline(percentile=value)


@command("quant.range", title="Baseline frame range", menu=None)
def set_range(ctx: CommandContext) -> None:
    text = str(ctx.args.get("value", "")).strip()
    try:
        first, last = (int(v) for v in text.split("-"))
        if first < 1 or last < first:
            raise ValueError
    except ValueError:
        say(f"Frames {text!r} is not first-last counted from 1, e.g. 1-10")
        raise ValueError("range must be first-last, from 1") from None
    # The panel counts frames from 1, as vizi's T does; the baseline and the
    # exports' volume index count from 0.
    _set_baseline(first=first - 1, last=last - 1)


@command("quant.pick", title="Select regions", menu=None)
def pick(ctx: CommandContext) -> str:
    """The person changed the table's selection: those regions are drawn bold,
    and the one just added is selected in vizi.  Every cause is answered."""
    engine = S.engine
    keys = tuple(str(k) for k in ctx.args.get("rows", ()) if engine is not None and str(k) in engine.regions)
    # The rows are this panel's own highlight whatever vizi answers: without
    # the person's "Control view" consent vizi refuses the selection, and the
    # rows must not jump back to the region vizi still has selected.
    with S.lock:
        before, S.chosen, S.table_selection = S.chosen, keys, keys
    S.wake.set()
    if not ctx.cause:
        return "no dataset"
    added = [k for k in keys if k not in before]
    target = added[-1] if added else (keys[0] if len(keys) == 1 else "")
    if not target or (target == S.selected and not S.asked):
        ctx.decline_view()
        return "declined"
    # Noted before asking: the view push that follows -- it may arrive before
    # this call returns, or after the next ask -- is this request's echo and
    # does not shrink the table's selection to one.
    with S.lock:
        S.asked.append(target)
    # One request: select the region and bring the camera onto it, as vizi's
    # own frame gesture does.
    result = ctx.move_view(select_region=target, frame_region=target)
    if not result.applied:
        with S.lock:
            if target in S.asked:
                S.asked.remove(target)
    say("" if result.applied else _refusal(result, "select and frame the region in vizi", f"select_region={target} frame_region={target}"))
    return result.outcome


@appearance
def theme_changed(ctx, look: Appearance) -> None:
    """vizi's theme, after the handshake and on every switch: the plot is
    redrawn in its colours."""
    with S.lock:
        S.colors = dict(look.hex_colors)
    S.wake.set()


@widget_geometry(PANEL)
def plot_resized(ctx, size: WidgetSize) -> None:
    """vizi says how large the plot is drawn: redraw it at exactly that many
    pixels, which vizi shows 1:1."""
    if size.widget_id != "plot":
        return
    width, height = size.pixels
    if width < 50 or height < 50:
        return  # collapsed; keep the last
    with S.lock:
        S.plot_size = (width, height, float(size.dpr))
    S.wake.set()


def _refusal(result, what: str, asked: str) -> str:
    """What to tell the person when vizi did not apply a view request -- and a
    line on stderr, which vizi keeps in this plugin's log, so that what vizi
    answered can be read afterwards (the panel's message is not kept)."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # The answer first: vizi's capture has been seen to drop the end of the
    # last line it holds when the session ends.
    print(f"{stamp} view request {result.outcome} {result.reason}: {asked}", file=sys.stderr, flush=True)
    if result.reason == "NOT_ENABLED":
        return f"To {what}: Tools → Plugin Manager…, tick “Control view” for vizi-quant"
    return f"vizi did not {what}: {result.reason}"


@command("quant.plot_click", title="Go to a frame on the plot", menu=None)
def plot_click(ctx: CommandContext) -> str:
    if not ctx.cause:
        return "no dataset"
    with S.lock:
        t = plot.frame_at(S.axes, float(ctx.args.get("x", -1)), float(ctx.args.get("y", -1)))
    if t is None:
        ctx.decline_view()
        return "outside the plot"
    result = ctx.move_view(t=t)
    say("" if result.applied else _refusal(result, "go to that frame", f"t={t}"))
    return result.outcome


def _stem(name: str) -> str:
    lowered = name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".ome.zarr", ".tiff", ".tif", ".zarr", ".nd2", ".czi", ".h5"):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name.rsplit(".", 1)[0] if "." in name else name


def _export(ctx: CommandContext, kind: str) -> str:
    engine = _engine_or_say()
    if engine is None:
        return "no dataset"
    host = ctx.host
    with S.lock:
        view = S.view
        baseline = S.baseline
    stem = _stem(view.dataset_name or "dataset")

    def work() -> None:
        try:
            path = host.request_save_path(
                f"{stem}-quant-{kind}.csv", extension="csv", title=f"Export {kind} measurements", timeout=1800
            )
        except HostError as exc:
            say(f"Export failed: {exc}")
            return
        if path is None:
            say("Export cancelled")
            return
        with engine.lock:
            regions, channels = engine.snapshot()
            table = export.Table(
                regions=regions,
                channels=channels,
                frames=engine.frames,
                time_step_s=export.time_step_seconds(view.time_increment, view.time_unit),
                spacing_unit=view.spacing_unit if any(view.axis_known_zyx) else "",
                known_xyz=tuple(reversed(view.axis_known_zyx)),
                baseline=baseline,
            )
            text = export.wide(table) if kind == "wide" else export.long(table)
            prov = export.provenance(
                table,
                view=view,
                plugin_version=VERSION,
                sdk_version=vizi_plugin_sdk.__version__,
                kind=kind,
                filter_seen=engine.filtered,
            )
        try:
            path.write_text(text, encoding="utf-8", newline="")
            path.with_name(path.stem + ".provenance.json").write_text(prov, encoding="utf-8")
        except OSError as exc:
            say(f"Export failed: {exc}")
            return
        # The name, not the path: a long label widens the dock under the plot.
        say(f"Exported {len(table.regions)} regions to {path.name}")

    threading.Thread(target=work, daemon=True, name=f"vizi-quant-export-{kind}").start()
    return "exporting"


@command("quant.export_wide", title="Export the wide table", menu=None)
def export_wide(ctx: CommandContext) -> str:
    return _export(ctx, "wide")


@command("quant.export_long", title="Export the long table", menu=None)
def export_long(ctx: CommandContext) -> str:
    return _export(ctx, "long")


# -- the panel -------------------------------------------------------------------------

panel(
    PANEL,
    title="Quantify",
    view_targets=["t", "select_region", "frame_region"],
    # The Tools entry reads "Quantify" (vizi writes "Show <title>" without one).
    menu_label="Quantify",
    root=w.column(
        "root",
        # 1. Measure (Stop while it runs) and how far it has got.
        w.row(
            "run_row",
            w.button("run", "Measure", command=run),
            w.button("stop", "Stop", command=stop),
            w.label("progress", ""),
        ),
        # 2. What the list's value and the plot show.
        w.row(
            "show_row",
            w.label("show_label", "Show"),
            w.dropdown("plot_stat", list(STAT_NAMES), command=choose_plot_stat, value=STAT_NAMES[0]),
            w.dropdown("plot_channel", ["ch1"], command=choose_plot_channel, value="ch1"),
            w.dropdown("plot_mode", list(MODES), command=choose_plot_mode, value=MODES[0]),
        ),
        # 3. The regions, with the chosen statistic on the frame vizi shows.
        w.table(
            TABLE,
            [
                w.table_column("name", "Region"),
                w.table_column("f", "F", type="number", precision=1),
                w.table_column("dff", "ΔF/F", type="number", precision=3),
            ],
            command=pick,
            selectable=True,
            multiple=True,
            sortable=True,
            moves_view=True,
            visible_rows=4,
        ),
        # 4. The traces.
        w.image("plot", command=plot_click, moves_view=True),
        # 5. How ΔF/F's baseline is found; shown only with ΔF/F.
        w.row(
            "baseline_row",
            w.label("baseline_label", "ΔF/F baseline (F0)"),
            w.dropdown("baseline", list(BASELINES), command=choose_baseline, value=BASELINES[0]),
        ),
        # The boxes sit in the sentence that says what they do.
        w.row(
            "moving_row",
            w.label("moving_a", "F0: percentile"),
            w.line_edit("percentile", command=set_percentile, placeholder="10", value="10"),
            w.label("moving_b", "of the nearest"),
            w.line_edit("window", command=set_window, placeholder="101", value="101"),
            w.label("moving_c", "frames"),
        ),
        w.row(
            "fixed_row",
            w.label("fixed_a", "F0: mean of frames"),
            w.line_edit("range", command=set_range, placeholder="1-10", value="1-10"),
            w.label("fixed_b", "(first-last, as T shows them)"),
        ),
        # 6. The tables.
        w.row(
            "export_row",
            w.button("export_wide", "Export table…", command=export_wide),
            w.button("export_long", "Export tidy table…", command=export_long),
        ),
        # 7. Empty unless something needs saying.
        w.label("status", "Open a dataset to measure its regions"),
    ),
)


def main() -> None:
    load_settings()
    perf.start(lambda: S.engine)
    serve()


if __name__ == "__main__":
    main()
