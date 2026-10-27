"""How the sweep divides one frame's reads (the 64 MB rule)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vizi_quant.engine import UNION_MAX_BYTES, Counters, Engine, FilteredRead, plan_windows


def test_one_read_when_the_union_fits():
    boxes = [((0, 0, 0), (4, 10, 10)), ((2, 50, 80), (6, 60, 99))]
    plan = plan_windows(boxes, 2, 64 * 2**20)
    assert plan == [((0, 0, 0), (6, 60, 99), [0, 1])]


def test_each_region_alone_when_the_union_is_too_big():
    # two small regions at opposite corners of an N10-sized frame: the union
    # is ~1.4 GB at u16, past the cap, so each is read on its own
    boxes = [((0, 0, 0), (10, 20, 20)), ((134, 2080, 2280), (144, 2100, 2300))]
    plan = plan_windows(boxes, 2, UNION_MAX_BYTES)
    assert plan == [((0, 0, 0), (10, 20, 20), [0]), ((134, 2080, 2280), (144, 2100, 2300), [1])]


def test_the_cap_is_in_bytes():
    boxes = [((0, 0, 0), (1, 1, 1000))]
    assert len(plan_windows(boxes * 2, 4, 4000)) == 1
    assert len(plan_windows([((0, 0, 0), (1, 1, 1000)), ((0, 0, 0), (1, 2, 1000))], 4, 4000)) == 2
    assert plan_windows([], 2, 1) == []


class _Volume(np.ndarray):
    filter = None


def test_reads_ask_for_raw_and_a_filtered_answer_is_refused():
    """Every read asks for ``raw``; a reply that comes back through the
    shared filter all the same is never used."""
    asked = []

    def read_volume(t, c, level, roi, raw=False):
        asked.append(raw)
        volume = np.zeros((1, 1, 1), np.uint16).view(_Volume)
        volume.filter = answer
        return volume

    engine = SimpleNamespace(host=SimpleNamespace(read_volume=read_volume), counters=Counters())
    answer = None
    assert Engine._read(engine, 0, 0, (0, 0, 0), (1, 1, 1)).shape == (1, 1, 1)
    answer = "org.example.denoise"
    with pytest.raises(FilteredRead):
        Engine._read(engine, 0, 0, (0, 0, 0), (1, 1, 1))
    assert asked == [True, True]


def test_progress_is_one_short_line_in_frames():
    from vizi_quant.plugin import progress_text

    assert progress_text(451, 3000, True) == "Measuring frame 451 / 3,000"
    assert progress_text(451, 3000, False) == "451 / 3,000 frames measured"
    assert progress_text(3000, 3000, False) == "All 3,000 frames measured"
    assert progress_text(0, 3000, False, regions=False) == "No regions to measure"
    assert progress_text(1, 1, False) == "Measured", "one frame: no 'All 1 frames'"


def test_a_refused_view_request_is_written_to_the_log(capsys):
    """Every refused view request goes to stderr, which vizi keeps in the
    plugin's log, with vizi's answer first; the panel says what to tick."""
    from vizi_quant.plugin import _refusal

    refused = SimpleNamespace(applied=False, outcome="REFUSED", reason="NOT_ENABLED")
    text = _refusal(refused, "select and frame the region in vizi", "select_region=abc frame_region=abc")
    assert text == "To select and frame the region in vizi: Tools → Plugin Manager…, tick “Control view” for vizi-quant"
    assert capsys.readouterr().err.rstrip().endswith("view request REFUSED NOT_ENABLED: select_region=abc frame_region=abc")
    assert _refusal(SimpleNamespace(outcome="REFUSED", reason="STALE"), "go to that frame", "t=7") == (
        "vizi did not go to that frame: STALE"
    )
    assert "view request REFUSED STALE: t=7" in capsys.readouterr().err


def test_the_plot_is_drawn_at_the_size_asked():
    from vizi_quant import plot

    series = [(f"r{i}", f"r{i}", np.linspace(0, 1, 20)) for i in range(3)]
    for size in ((470, 210, 1.0), (1070, 130, 2.0), (300, 60, 1.0)):
        png, axes = plot.render(series, 20, 5, selected={"r1"}, ylabel="mean", size=size)
        assert (int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")) == size[:2]
        assert 0 <= axes.left < axes.right <= size[0] and 0 <= axes.top < axes.bottom <= size[1]


def test_while_measuring_the_axis_ends_at_the_last_frame_measured():
    """While measuring, after k of N frames the plot spans frames 1..k
    (1-based, as the panel shows them) so the traces fill it; finished, it
    spans 1..N."""
    from vizi_quant import plot
    from vizi_quant.plugin import last_measured

    N, k = 300, 37
    values = np.full(N, np.nan)
    values[:k] = 1.0
    series = [("a", "a", values), ("b", "b", np.full(N, np.nan))]
    assert last_measured(series) == k
    _, axes = plot.render(series, N, 0, upto=last_measured(series))
    assert (axes.x0, axes.x1) == (0.5, k + 0.5)
    values[:] = 1.0
    _, axes = plot.render(series, N, 0, upto=None)
    assert (axes.x0, axes.x1) == (0.5, N + 0.5)
    assert last_measured([("c", "c", np.full(N, np.nan))]) == 1, "nothing yet: the first frame"


def test_a_click_on_the_plot_gives_vizi_s_frame():
    """The axis counts from 1; the frame sent to vizi counts from 0."""
    from vizi_quant import plot

    series = [("a", "a", np.linspace(0, 1, 10))]
    _, axes = plot.render(series, 10, 0)
    x_of = lambda shown: axes.left + (shown - axes.x0) / (axes.x1 - axes.x0) * (axes.right - axes.left)
    y = (axes.top + axes.bottom) / 2
    assert plot.frame_at(axes, x_of(1), y) == 0
    assert plot.frame_at(axes, x_of(10), y) == 9


def test_the_plot_is_drawn_in_the_colours_given():
    import io as _io

    from PIL import Image

    from vizi_quant import plot

    dark = {"panel_bg": "#2a2a2a", "text": "#ffffff", "text_disabled": "#7f7f7f", "accent": "#2a82da", "border": "#4a4a4a"}
    series = [("a", "a", np.linspace(0, 1, 40)), ("b", "b", np.linspace(1, 0, 40))]
    corner = lambda png: Image.open(_io.BytesIO(png)).convert("RGB").getpixel((1, 1))
    png, _ = plot.render(series, 40, 3, selected={"a"}, colors=dark)
    assert corner(png) == (0x2A, 0x2A, 0x2A)
    pixels = {tuple(int(v) for v in px) for px in np.asarray(Image.open(_io.BytesIO(png)).convert("RGB")).reshape(-1, 3)}
    assert (0x7F, 0x7F, 0x7F) in pixels, "the unselected trace in the disabled colour"
    assert (0x1F, 0x77, 0xB4) in pixels, "the selected trace keeps its own colour"
    png, _ = plot.render(series, 40, 3)
    assert corner(png) == (0xFF, 0xFF, 0xFF), "no appearance from vizi: today's light plot"
