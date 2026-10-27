"""The exports' shapes, and ΔF/F."""

from __future__ import annotations

import csv
import io
import json

import numpy as np

from vizi_quant import cache, derived, export, stats
from vizi_quant.cache import Trace
from vizi_quant.engine import Region

from test_parity import HERE, SPACING_ZYX, C, T, fixture_volume


def _fixture_regions():
    """The parity fixture's regions with traces filled by ``stats`` -- what
    the engine would hold after a sweep."""
    meta = json.loads((HERE / "host-masks.json").read_text())
    masks = np.load(HERE / "host-masks.npz")
    volume = fixture_volume()
    regions = []
    for name, m in meta.items():
        mask, origin = masks[name.replace(" ", "_")], tuple(m["origin_zyx"])
        (oz, oy, ox), (dz, dy, dx) = origin, mask.shape
        trace = Trace(T, C)
        for t in range(T):
            trace.record_geometry(t, stats.geometry(mask, origin, SPACING_ZYX))
            for c in range(C):
                window = volume[t, oz:oz + dz, c, oy:oy + dy, ox:ox + dx]
                trace.record_values(c, t, stats.intensity(stats.region_values(window, mask)))
        regions.append(Region(m["region_id"], name, m["content_hash"], 1, True, trace))
    return regions


def test_wide_is_builtin_measures_csv():
    """Everything but the set id in the first header cell, byte for byte."""
    table = export.Table(regions=_fixture_regions(), channels=[0, 1], frames=T, time_step_s=0.25, spacing_unit="um")
    ours = export.wide(table).encode("utf-8")
    theirs = (HERE / "builtin-measure.csv").read_bytes()
    first, rest = ours.split(b",", 1)
    their_first, their_rest = theirs.split(b",", 1)
    assert first.decode() == f"volume index (using region id {export.set_id(table.regions)})"
    assert their_first.startswith(b"volume index (using region id ")
    assert rest == their_rest


def test_wide_leaves_unmeasured_cells_empty_and_honours_outputs():
    regions = _fixture_regions()[:1]
    regions[0].trace.geom_state[2] = 0
    regions[0].trace.value_state[:, 2] = 0
    table = export.Table(regions=regions, channels=[1], frames=T, time_step_s=None, spacing_unit="",
                         outputs=("mean", "n"))
    rows = list(csv.reader(io.StringIO(export.wide(table))))
    assert rows[0][1:] == ["Box 1 n", "Box 1 ch2 mean"]
    assert rows[3] == ["2", "", ""]
    assert rows[1][1] == "8192"


def test_uncalibrated_headers_carry_no_unit():
    """vizi's built-in Measure on an uncalibrated file writes ``Box 1 x`` and
    ``Box 1 size x``: no unit and no time column."""
    regions = _fixture_regions()[:1]
    table = export.Table(regions=regions, channels=[0], frames=T, time_step_s=None, spacing_unit="")
    header = export.wide(table).split("\r\n", 1)[0].split(",")
    assert header[1:8] == ["Box 1 n", "Box 1 x", "Box 1 y", "Box 1 z", "Box 1 size x", "Box 1 size y", "Box 1 size z"]


def test_set_id_follows_geometry_not_names_or_order():
    a = _fixture_regions()
    b = list(reversed(_fixture_regions()))
    b[0].name = "renamed"
    assert export.set_id(a) == export.set_id(b) and len(export.set_id(a)) == 12
    a[0].content_hash = "0" * 16
    assert export.set_id(a) != export.set_id(b)


def test_long_has_a_row_per_statistic():
    regions = _fixture_regions()[:1]
    table = export.Table(regions=regions, channels=[0], frames=T, time_step_s=0.25, spacing_unit="um")
    rows = list(csv.DictReader(io.StringIO(export.long(table))))
    geo = [r for r in rows if r["channel"] == ""]
    assert len(geo) == T * len(stats.GEOMETRY)
    raw = [r for r in rows if r["channel"] == "1" and not r["statistic"].startswith("dff_")]
    assert len(raw) == T * len(stats.INTENSITY)
    assert {r["time_s"] for r in raw} == {f"{t * 0.25:.6g}" for t in range(T)}


def test_dff_sliding_and_fixed():
    f = np.array([10.0, 10.0, 20.0, 10.0, np.nan, 10.0])
    sliding = derived.dff(f, derived.Baseline(window=3, percentile=0))
    assert sliding[2] == 1.0 and sliding[0] == 0.0 and np.isnan(sliding[4])
    fixed = derived.dff(f, derived.Baseline(mode=derived.FIXED, first=0, last=1))
    assert fixed[2] == 1.0 and np.isnan(fixed[4])
    assert np.isnan(derived.dff(np.full(4, np.nan), derived.Baseline())).all()


def test_time_units():
    assert export.time_step_seconds(250, "ms") == 0.25
    assert export.time_step_seconds(0, "s") is None
    assert export.time_step_seconds(1.0, "fortnight") is None


def test_provenance_names_the_fallback_axes():
    """With only the display fallback for a pitch, every axis is exported in
    voxels and the provenance says so."""
    from types import SimpleNamespace

    view = SimpleNamespace(
        dataset_name="gcamp_MMStack.ome.tif", dataset_uri="/x", dataset_series=None,
        shape_tczyx=(T, C, 24, 96, 96), dtype="uint16", spacing_zyx=(1.0, 0.3208, 0.3208),
        spacing_unit="", spacing_provenance="DISPLAY_FALLBACK",
        time_increment=0.3, time_unit="s", filter=None,
    )
    table = export.Table(regions=_fixture_regions()[:1], channels=[0], frames=T, time_step_s=0.3, spacing_unit="")
    doc = json.loads(export.provenance(table, view=view, plugin_version="0", sdk_version="0", kind="wide"))
    assert doc["dataset"]["geometry_units"] == {"x": "voxels", "y": "voxels", "z": "voxels"}
    assert doc["dataset"]["fallback_axes"] == ["z", "y", "x"]
    assert doc["dataset"]["spacing_zyx"] is None
    assert doc["dataset"]["spacing_zyx_offered"] == [1.0, 0.3208, 0.3208]


def test_partly_calibrated_file_scales_the_known_axes():
    """XY pitch from the file, Z unrecorded and sent as NaN: X and Y columns
    carry the unit, Z stays in voxels, and no NaN reaches a file."""
    from vizi_plugin_sdk.server import HostViewState

    view = HostViewState(
        t=0, z=0, c=0, level=0, has_crop=False, crop_min_zyx=(), crop_max_zyx=(), mip=False, levels=(),
        selected_region_id="", dataset_uri="/x", dataset_name="gcamp_MMStack.ome.tif",
        open_generation=1, spacing_zyx=(float("nan"), 0.3208, 0.3208), spacing_unit="um",
        spacing_provenance="FILE", shape_tczyx=(T, C, 24, 96, 96), dtype="uint16",
        time_increment=0.3, time_unit="s",
    )
    assert view.axis_known_zyx == (False, True, True)
    assert cache.identity_of(view) == cache.identity_of(view)  # a NaN would never equal itself
    assert cache.identity_of(view)["spacing_zyx"] == [None, 0.3208, 0.3208]
    table = export.Table(
        regions=_fixture_regions()[:1], channels=[0], frames=T, time_step_s=0.3, spacing_unit="um",
        known_xyz=tuple(reversed(view.axis_known_zyx)),
    )
    header = export.wide(table).split(export.EOL)[0].split(",")
    name = table.regions[0].name
    assert header[3:9] == [f"{name} x (um)", f"{name} y (um)", f"{name} z", f"{name} size x (um)", f"{name} size y (um)", f"{name} size z"]
    assert "x_um" in export.long(table) and ",z," in export.long(table)
    text = export.provenance(table, view=view, plugin_version="0", sdk_version="0", kind="wide")
    assert "NaN" not in text
    doc = json.loads(text)["dataset"]
    assert doc["geometry_units"] == {"x": "um", "y": "um", "z": "voxels"}
    assert doc["fallback_axes"] == ["z"]
    assert doc["spacing_zyx"] == [None, 0.3208, 0.3208]
    assert doc["spacing_zyx_offered"] == [None, 0.3208, 0.3208]
