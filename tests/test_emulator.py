"""The plugin end to end against the SDK's host emulator, over the real
transport -- no vizi.

Every expected number is computed here from the dataset and the emulator's
box rule, never by asking the plugin; the plugin is read only through what it
pushes to the panel and the files it exports.
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from vizi_quant import derived, stats

ROOT = Path(__file__).resolve().parents[1]
# The SDK's host emulator, vendored at the pinned SDK commit (tests/_vendor/README.md).
sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))

from vizi_hostsim import Dataset, Host, Segment, segment_id  # noqa: E402
from vizi_plugin_sdk import pb2, roi_pb2  # noqa: E402

PANEL, TABLE = "quant", "regions"
T, C, Z, Y, X = 6, 2, 4, 12, 12
SPACING = (2.0, 0.5, 0.5)
DT = 0.25


def source(t: int, c: int) -> np.ndarray:
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    return (500 * c + 10 * z + 5 * y + x + 60 * ((t + x + y) % 5) + 7 * t).astype(np.uint16)


def box(lo, hi) -> roi_pb2.Roi:
    return roi_pb2.Roi(type_name="box", vertices=[
        roi_pb2.Vertex(position=roi_pb2.Vec3(z=lo[0], y=lo[1], x=lo[2])),
        roi_pb2.Vertex(position=roi_pb2.Vec3(z=hi[0], y=hi[1], x=hi[2])),
    ])


BOXES = {"alpha": ((0, 2, 2), (1, 4, 5)), "beta": ((1, 6, 1), (3, 9, 3)), "beta2": ((0, 5, 5), (2, 7, 8))}


def expected(lo, hi, t, c):
    """(geometry, intensity) by the emulator's rule: indices lo..hi inclusive."""
    block = source(t, c)[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    mask = np.ones(block.shape, bool)
    return stats.geometry(mask, lo, SPACING), stats.intensity(block.ravel().astype(np.float64))


def wait_for(pred, what, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        got = pred()
        if got:
            return got
        time.sleep(0.05)
    raise AssertionError(f"{what} within {timeout:g}s")


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    plugins = tmp_path_factory.mktemp("plugins")
    env = {**os.environ, "VIZI_PLUGINS_DIR": str(plugins)}
    subprocess.run([sys.executable, str(ROOT / "vizi_quant_plugin.py"), "--install"], check=True, env=env, cwd=ROOT,
                   capture_output=True)
    return plugins / "org.focolab.vizi_quant"


@pytest.fixture
def session(installed, tmp_path):
    dataset = Dataset(
        shape_tczyx=(T, C, Z, Y, X), np_dtype=np.dtype(np.uint16), wire_dtype=pb2.DTYPE_U16,
        open_generation=7, volume=source, t=2, visible_channels=(0,), time_increment_s=DT,
        spacing_zyx=SPACING, spacing_unit="um", dataset_name="synthetic.ome.tif",
    )
    host = Host(installed, dataset)
    host.file_state_dir = str(tmp_path / "state")
    ids = {n: host.add_region(box(*BOXES[n]), name=n) for n in ("alpha", "beta")}
    host.add(Segment.plugin_arena(segment_id(1), "P", 8 << 20, host.directory, subject=PANEL))
    with host:
        wait_for(lambda: host.ready, "Ready")
        host.open_panel(PANEL, grant=segment_id(1))
        yield host, dataset, ids, tmp_path


def rows(host):
    return [key for key, _ in host.table_rows(PANEL, TABLE)]


def cell(host, column, row):
    """A table cell by column id: name, f, dff."""
    found = dict(host.table_rows(PANEL, TABLE)).get(row)
    return None if found is None else found[("name", "f", "dff").index(column)]


def table_selection(host):
    return list(host.tables[(PANEL, TABLE)].selection)


def png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def measured_all(host, n_regions):
    want = f"All {T:,} frames measured"
    return lambda: host.status_of(PANEL, widget="progress") == want


def export(host, tmp_path, which):
    out = tmp_path / f"{which}.csv"
    host.save_path_answer = str(out)
    done = host.invoke_widget(PANEL, f"export_{which.rstrip('0123456789')}")
    assert done.ok, done.message
    wait_for(lambda: host.status_of(PANEL, widget="status").endswith(out.name), f"the {which} export", 10)
    return out


def test_measure_export_and_follow_edits(session):
    host, dataset, ids, tmp_path = session
    assert {"view", "regions"} <= host.subscriptions
    wait_for(lambda: rows(host) == [ids["alpha"], ids["beta"]], "a row per region, in the host's order")
    # Nothing is measured until the person asks.
    time.sleep(1.0)
    assert host.status_of(PANEL, widget="progress") == f"0 / {T} frames measured"

    assert host.invoke_widget(PANEL, "run").ok
    wait_for(measured_all(host, 2), "every frame of both regions", 30)
    _, values = expected(*BOXES["alpha"], 2, 0)
    wait_for(lambda: cell(host, "f", ids["alpha"]) is not None, "alpha's F")
    assert cell(host, "f", ids["alpha"]) == pytest.approx(values[0]), "the mean on the frame on screen (t=2)"
    assert cell(host, "name", ids["alpha"]) == "alpha"
    assert any(p == PANEL and w == "plot" for p, w, _ in host.images), "the traces were drawn"

    wide = list(csv.reader(export(host, tmp_path, "wide").open()))
    header = wide[0]
    assert header[0].startswith("volume index (using region id ") and header[1] == "time (s)"
    for t in range(T):
        row = wide[1 + t]
        assert row[0] == str(t) and row[1] == f"{t * DT:.3f}"
        for name in ("alpha", "beta"):
            geo, _ = expected(*BOXES[name], t, 0)
            assert row[header.index(f"{name} n")] == str(int(geo[0]))
            assert row[header.index(f"{name} x (um)")] == f"{geo[1]:.3f}"
            assert row[header.index(f"{name} size z (um)")] == f"{geo[6]:.3f}"
            for c in range(C):
                _, values = expected(*BOXES[name], t, c)
                for k, stat in enumerate(stats.INTENSITY):
                    assert row[header.index(f"{name} ch{c + 1} {stat}")] == f"{values[k]:.3f}", (name, t, c, stat)
    prov = json.loads((tmp_path / "wide.provenance.json").read_text())
    assert prov["voxels"].startswith("raw") and len(prov["regions"]) == 2

    long = list(csv.DictReader(export(host, tmp_path, "long").open()))
    means = [r for r in long if r["region"] == "alpha" and r["channel"] == "1" and r["statistic"] == "mean"]
    assert [int(r["frame"]) for r in means] == list(range(T))
    f = np.array([expected(*BOXES["alpha"], t, 0)[1][0] for t in range(T)])
    want = derived.dff(f, derived.Baseline())
    got = {int(r["frame"]): float(r["value"]) for r in long
           if r["region"] == "alpha" and r["channel"] == "1" and r["statistic"] == "dff_mean"}
    assert all(abs(got[t] - want[t]) < 1e-12 for t in range(T) if np.isfinite(want[t]))

    # An edit: beta redrawn is measured again by itself -- the sweep is armed.
    redrawn = pb2.RoiRecord()
    redrawn.CopyFrom(host.rois[ids["beta"]])
    redrawn.roi.CopyFrom(box(*BOXES["beta2"]))
    with host._lock:
        host.rois[ids["beta"]] = redrawn
    geo, values = expected(*BOXES["beta2"], 3, 1)
    attempt = [0]

    def remeasured():
        attempt[0] += 1
        wide = list(csv.reader(export(host, tmp_path, f"wide{attempt[0]}").open()))
        row = wide[4]
        return row[wide[0].index("beta n")] == str(int(geo[0])) and row[wide[0].index("beta ch2 mean")] == f"{values[0]:.3f}"

    wait_for(remeasured, "beta measured again through its new shape", 30)
    wait_for(measured_all(host, 2), "every frame of both, again", 30)

    # A deletion takes the row away.
    host.delete_region(ids["alpha"])
    wait_for(lambda: rows(host) == [ids["beta"]], "alpha's row gone")


def test_plot_click_moves_the_frame(session):
    host, dataset, ids, _ = session
    wait_for(lambda: rows(host), "rows")
    host.set_view_toggle(PANEL, True)
    wait_for(lambda: any(p == PANEL and w == "plot" for p, w, _ in host.images), "a plot")
    done = host.invoke_widget(PANEL, "plot", x=280, y=150, button="left", modifiers="", image=0)
    assert done.ok, done.message
    req, res = host.view_requests[-1]
    assert req.HasField("t") and 0 <= req.t < T and res.outcome == pb2.VIEW_OUTCOME_APPLIED, (req, res)
    done = host.invoke_widget(PANEL, "plot", x=1, y=1, button="left", modifiers="", image=0)
    assert host.view_requests[-1][0].decline, "a click outside the axes is declined, not left unanswered"
    done = host.select_table_rows(PANEL, TABLE, [ids["beta"]])
    req, res = host.view_requests[-1]
    assert req.select_region == ids["beta"] and req.frame_region == ids["beta"]
    assert res.outcome == pb2.VIEW_OUTCOME_APPLIED


def test_a_row_click_stays_when_vizi_refuses_the_selection(session):
    """Without the person's "Control view" consent vizi refuses the
    selection and keeps another region selected.  The clicked row stays
    highlighted, and the status says what to tick rather than NOT_ENABLED."""
    host, dataset, ids, _ = session
    wait_for(lambda: rows(host) == [ids["alpha"], ids["beta"]], "rows")
    done = host.select_table_rows(PANEL, TABLE, [ids["beta"]])
    assert done.ok, done.message
    req, res = host.view_requests[-1]
    assert req.select_region == ids["beta"] and req.frame_region == ids["beta"]
    assert res.outcome == pb2.VIEW_OUTCOME_REFUSED
    host.await_status(PANEL, "To select and frame the region in vizi: Tools → Plugin Manager…, tick “Control view” for vizi-quant", timeout=5)
    assert "NOT_ENABLED" not in host.status_of(PANEL)
    time.sleep(1.5)  # three redraws: the row must not fall back to vizi's selection
    assert table_selection(host) == [ids["beta"]]


def test_several_rows_selected_and_the_one_added_goes_to_vizi(session):
    """A multiple table: the selection is drawn bold, and the row the person
    just added is the one vizi is asked to select and frame; clearing
    declines; vizi's resulting selection is an echo, not a new choice."""
    host, dataset, ids, _ = session
    wait_for(lambda: rows(host) == [ids["alpha"], ids["beta"]], "rows")
    host.set_view_toggle(PANEL, True)
    assert host.select_table_rows(PANEL, TABLE, [ids["alpha"]]).ok
    first = host.view_requests[-1][0]
    assert first.select_region == ids["alpha"] and first.frame_region == ids["alpha"], "one request selects and frames"
    assert host.select_table_rows(PANEL, TABLE, [ids["alpha"], ids["beta"]]).ok
    req, res = host.view_requests[-1]
    assert req.select_region == ids["beta"] and req.frame_region == ids["beta"]
    assert res.outcome == pb2.VIEW_OUTCOME_APPLIED
    time.sleep(1.0)
    assert table_selection(host) == [ids["alpha"], ids["beta"]], "vizi's one selection does not shrink the table's"
    assert host.select_table_rows(PANEL, TABLE, []).ok
    assert host.view_requests[-1][0].decline, "a cleared selection is answered with a decline"


def test_the_table_sorts_on_the_host_and_patches_by_key(session):
    host, dataset, ids, _ = session
    wait_for(lambda: rows(host) == [ids["alpha"], ids["beta"]], "rows")
    assert host.invoke_widget(PANEL, "run").ok
    wait_for(measured_all(host, 2), "measured", 30)
    # the progress line and the rows' patch are separate patches
    wait_for(lambda: all(cell(host, "f", k) is not None for k in rows(host)), "every row's F")
    by_f = host.table_sorted(PANEL, TABLE, "f")
    assert by_f == sorted(rows(host), key=lambda k: cell(host, "f", k))
    assert rows(host) == [ids["alpha"], ids["beta"]], "a sort leaves the plugin's order underneath"


def corner_rgb(data: bytes) -> tuple[int, int, int]:
    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB").getpixel((1, 1))


def test_the_plot_follows_vizi_s_theme(session):
    """The emulator sends vizi's appearance after Ready (dark by default) and
    again on a switch: the plot's background is the panel's in each."""
    host, dataset, ids, _ = session
    wait_for(lambda: any(p == PANEL and w == "plot" for p, w, _ in host.images), "a plot")
    wait_for(lambda: corner_rgb(host.images[-1][2].payload) == (0x2A, 0x2A, 0x2A), "a dark plot")
    assert host.set_theme("light")
    wait_for(lambda: corner_rgb(host.images[-1][2].payload) == (0xFF, 0xFF, 0xFF), "a light plot after the switch")
    assert host.set_theme("dark")
    wait_for(lambda: corner_rgb(host.images[-1][2].payload) == (0x2A, 0x2A, 0x2A), "dark again")


def test_the_plot_is_redrawn_at_the_size_vizi_draws_it(session):
    host, dataset, ids, _ = session
    wait_for(lambda: any(p == PANEL and w == "plot" for p, w, _ in host.images), "a plot")
    assert host.resize(PANEL, "plot", 600, 300, 2.0)
    wait_for(lambda: png_size(host.images[-1][2].payload) == (1200, 600), "a 1200x600 plot after a 600x300 @2x resize")
    # a click is in the pushed image's own pixels: the middle of the big image is inside the axes
    host.set_view_toggle(PANEL, True)
    done = host.invoke_widget(PANEL, "plot", x=600, y=300, button="left", modifiers="", image=0)
    assert done.ok and host.view_requests[-1][0].HasField("t")


def test_baseline_frames_are_typed_from_1(session):
    """The panel counts frames as vizi's T does, from 1; the baseline and the
    exports count from 0."""
    host, dataset, ids, tmp_path = session
    wait_for(lambda: rows(host), "rows")
    assert host.invoke_widget(PANEL, "baseline", value="fixed frames", index=1).ok
    assert host.invoke_widget(PANEL, "range", value="2-3").ok
    assert not host.invoke_widget(PANEL, "range", value="0-3").ok, "there is no frame 0 in the panel"
    export(host, tmp_path, "long")
    prov = json.loads((tmp_path / "long.provenance.json").read_text())
    assert prov["dff_baseline"] == {"mode": "fixed", "frames": [1, 2]}


def test_reads_are_raw_under_a_shared_filter(session):
    """Plugins See Filtered Data is on and the filter blanks every volume:
    the sweep still measures the file's own voxels."""
    host, dataset, ids, tmp_path = session
    wait_for(lambda: rows(host), "rows")
    dataset.applied_filter = pb2.AppliedFilter(plugin_id="org.example.denoise", stage_id="denoise")
    dataset.filter_fn = lambda volume, t, c: np.zeros_like(volume)
    assert host.invoke_widget(PANEL, "run").ok
    wait_for(measured_all(host, 2), "every frame of both regions", 30)
    wide = list(csv.reader(export(host, tmp_path, "wide").open()))
    _, values = expected(*BOXES["alpha"], 1, 0)
    assert values[0] > 0 and wide[2][wide[0].index("alpha ch1 mean")] == f"{values[0]:.3f}"
    assert json.loads((tmp_path / "wide.provenance.json").read_text())["voxels"].startswith("raw")


@pytest.mark.parametrize("spacing", [SPACING, (None, 0.5, 0.5)], ids=["calibrated", "z-unrecorded"])
def test_cache_survives_a_new_session(installed, tmp_path, spacing):
    """Measured once, the numbers come back from the file's folder -- also
    for a file with no Z pitch, which the host sends as NaN."""
    state = tmp_path / "state"
    for attempt in range(2):
        dataset = Dataset(
            shape_tczyx=(T, C, Z, Y, X), np_dtype=np.dtype(np.uint16), wire_dtype=pb2.DTYPE_U16,
            open_generation=3 + attempt, volume=source, t=0, spacing_zyx=spacing, spacing_unit="um",
        )
        host = Host(installed, dataset)
        host.file_state_dir = str(state)
        rid = host.add_region(box(*BOXES["alpha"]), name="alpha")
        if attempt == 1:
            # the same region id and geometry as the first session
            record = host.rois.pop(rid)
            record.region_id = first_id
            host.rois[first_id] = record
            rid = first_id
        host.add(Segment.plugin_arena(segment_id(1), "P", 8 << 20, host.directory, subject=PANEL))
        with host:
            wait_for(lambda: host.ready, "Ready")
            host.open_panel(PANEL, grant=segment_id(1))
            wait_for(lambda: rows(host) == [rid], "the row")
            if attempt == 0:
                assert host.invoke_widget(PANEL, "run").ok
                wait_for(measured_all(host, 1), "measured", 30)
                first_id = rid
                host.shutdown()
            else:
                wait_for(measured_all(host, 1), "served from the cache without measuring", 10)
                assert not host.inclusion_requests, "nothing was read again"
                header, row = list(csv.reader(export(host, tmp_path, "wide").open()))[:2]
                lo, hi = BOXES["alpha"]
                geo = stats.geometry(np.ones(np.subtract(hi, lo) + 1, bool), lo, [v or 1.0 for v in spacing])
                z = "alpha z (um)" if spacing[0] else "alpha z"
                assert row[header.index(z)] == f"{geo[3]:.3f}" and row[header.index("alpha x (um)")] == f"{geo[1]:.3f}"
    assert list((state / "cache-v1").glob("*.npz"))
