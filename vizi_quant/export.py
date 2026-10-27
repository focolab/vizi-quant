"""The two tables and the provenance file, as text.  No host, no I/O.

- **Wide** is built-in Measure's CSV, byte for byte but the set id (the
  parity fixture holds it): a row per frame, every region's series as
  columns, an empty cell where a region was not measured, values to three
  decimals, CRLF line ends.
- **Long** (tidy) is one row per region, frame, channel and statistic, with
  ΔF/F of each intensity statistic beside the raw ones.
- **Provenance** says what the numbers were measured from and how.

The set id names the regions a table came from: the first 12 hex of SHA-256
over ``"<region_id>\\t<content_hash>\\n"`` for the measured regions, sorted by
id -- the same regions with the same geometry give the same id in any session,
any edit to their voxels gives another.  Built-in Measure wrote its saved
set's own id there; a plugin cannot read that, and this one identifies the
geometry rather than a save.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from . import derived, stats

#: RFC 4180's line end, which is also what built-in Measure wrote.
EOL = "\r\n"

#: Output names as the panel offers them, and the columns each one writes.
OUTPUTS = ("mean", "p85-95", "p05-15", "std", "n", "center", "size")


@dataclass
class Table:
    """What an export needs, copied out of the engine under its lock."""

    regions: list  # engine.Region, in list order
    channels: list[int]
    frames: int
    time_step_s: "float | None"
    spacing_unit: str  # "um", or "" when no axis's pitch is known (voxel units)
    #: Which of x, y, z carry ``spacing_unit``; the others are in voxels (a
    #: file may record XY and not Z).
    known_xyz: tuple = (True, True, True)
    outputs: tuple = OUTPUTS
    baseline: derived.Baseline = field(default_factory=derived.Baseline)


def set_id(regions) -> str:
    lines = "".join(f"{r.region_id}\t{r.content_hash}\n" for r in sorted(regions, key=lambda r: r.region_id))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()[:12]


def time_step_seconds(increment: float, unit: str) -> "float | None":
    if not increment or increment <= 0:
        return None
    scale = {"s": 1.0, "": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "min": 60.0, "h": 3600.0}.get(unit)
    return None if scale is None else increment * scale


def csv_field(text: str) -> str:
    if any(ch in text for ch in ',"\r\n'):
        return '"' + text.replace('"', '""') + '"'
    return text


def _num(value: float) -> str:
    return "" if not np.isfinite(value) else f"{value:.3f}"


def _units(table: Table) -> list:
    """Per x, y, z: the unit its geometry columns are in, or "" for voxels."""
    return [table.spacing_unit if known else "" for known in table.known_xyz]


def _suffix(table: Table) -> list:
    """Per x, y, z: `` (um)`` after a geometry column, or nothing where the
    pitch is unknown -- built-in Measure's header on an uncalibrated dataset."""
    return [f" ({u})" if u else "" for u in _units(table)]


def wide(table: Table) -> str:
    unit = _suffix(table)
    header = [f"volume index (using region id {set_id(table.regions)})"]
    if table.time_step_s is not None:
        header.append("time (s)")
    columns = []  # (header, function(t) -> cell)
    for r in table.regions:
        tr, name = r.trace, r.name
        measured = tr.geom_state == 1
        if "n" in table.outputs:
            columns.append((f"{name} n", lambda t, tr=tr, m=measured: str(int(tr.geom[t, 0])) if m[t] else ""))
        if "center" in table.outputs:
            for i, axis in enumerate("xyz"):
                columns.append((f"{name} {axis}{unit[i]}", lambda t, tr=tr, i=i, m=measured: _num(tr.geom[t, 1 + i]) if m[t] else ""))
        if "size" in table.outputs:
            for i, axis in enumerate("xyz"):
                columns.append((f"{name} size {axis}{unit[i]}", lambda t, tr=tr, i=i, m=measured: _num(tr.geom[t, 4 + i]) if m[t] else ""))
        for c in table.channels:
            for k, stat in enumerate(stats.INTENSITY):
                if stat in table.outputs:
                    columns.append((
                        f"{name} ch{c + 1} {stat}",
                        lambda t, tr=tr, c=c, k=k: _num(tr.values[c, t, k]) if tr.value_state[c, t] == 1 else "",
                    ))
    header += [h for h, _ in columns]
    lines = [",".join(csv_field(h) for h in header)]
    for t in range(table.frames):
        row = [str(t)]
        if table.time_step_s is not None:
            row.append(f"{t * table.time_step_s:.3f}")
        row += [cell(t) for _, cell in columns]
        lines.append(",".join(row))
    return EOL.join(lines) + EOL


def long(table: Table) -> str:
    unit = [f"_{u}" if u else "" for u in _units(table)]
    geo_names = ["n"] + [f"{a}{u}" for a, u in zip("xyz", unit)] + [f"size_{a}{u}" for a, u in zip("xyz", unit)]
    lines = ["region_id,region,frame,time_s,channel,statistic,value"]
    for r in table.regions:
        tr, rid, name = r.trace, csv_field(r.region_id), csv_field(r.name)
        stamp = [("" if table.time_step_s is None else f"{t * table.time_step_s:.6g}") for t in range(table.frames)]
        for t in range(table.frames):
            if tr.geom_state[t] != 1:
                continue
            for i, stat in enumerate(geo_names):
                value = str(int(tr.geom[t, 0])) if i == 0 else repr(float(tr.geom[t, i]))
                lines.append(f"{rid},{name},{t},{stamp[t]},,{stat},{value}")
        for c in table.channels:
            measured = tr.value_state[c] == 1
            for k, stat in enumerate(stats.INTENSITY):
                f = np.where(measured, tr.values[c, :, k], np.nan)
                d = derived.dff(f, table.baseline)
                for t in np.flatnonzero(measured):
                    lines.append(f"{rid},{name},{t},{stamp[t]},{c + 1},{stat},{float(f[t])!r}")
                    if np.isfinite(d[t]):
                        lines.append(f"{rid},{name},{t},{stamp[t]},{c + 1},dff_{stat},{float(d[t])!r}")
    return EOL.join(lines) + EOL


def provenance(table: Table, *, view, plugin_version: str, sdk_version: str, kind: str, filter_seen=None) -> str:
    known_zyx = tuple(reversed(table.known_xyz)) if table.spacing_unit else (False, False, False)
    offered = list(view.spacing_zyx) if view.spacing_zyx else None
    pitch = [float(v) if k else None for v, k in zip(offered or (0, 0, 0), known_zyx)]
    doc = {
        "written_by": {"plugin": "org.focolab.vizi_quant", "version": plugin_version, "sdk": sdk_version},
        "table": kind,
        "dataset": {
            "name": view.dataset_name,
            "uri": view.dataset_uri,
            "series": getattr(view, "dataset_series", None),
            "shape_tczyx": list(view.shape_tczyx),
            "dtype": str(view.dtype),
            # Per axis: a file may record XY and not Z.  A known
            # axis is scaled by its pitch; an unknown one is exported in
            # voxels and named in fallback_axes, its pitch null.  What the
            # host sent is kept beside it, never used.
            "spacing_zyx": pitch if any(known_zyx) else None,
            "spacing_unit": table.spacing_unit,
            "spacing_provenance": view.spacing_provenance,
            "geometry_units": {a: u or "voxels" for a, u in zip("xyz", _units(table))},
            "fallback_axes": [a for a, k in zip("zyx", known_zyx) if not k],
            "spacing_zyx_offered": [float(v) if np.isfinite(v) else None for v in offered] if offered else None,
            "time_increment": view.time_increment or None,
            "time_unit": view.time_unit,
        },
        "region_set_id": set_id(table.regions),
        "regions": [
            {"region_id": r.region_id, "name": r.name, "content_hash": r.content_hash, "revision": r.revision}
            for r in table.regions
        ],
        "channels_measured": [c + 1 for c in table.channels],
        "voxels": "raw: the file's own voxels at level 0" if filter_seen is None else f"FILTERED: {filter_seen}",
        "view_filter_at_export": None if getattr(view, "filter", None) is None else str(view.filter),
        "statistics": {
            "stats_version": stats.STATS_VERSION,
            "footprint": "host read_roi_inclusion at each frame (static regions once)",
            "sample": f"stride ceil(n / {stats.SAMPLE_CAP}) through the region's voxels in z,y,x order; n exact",
            "std": "population",
            "bands": "mean of values between the two percentiles (numpy linear), inclusive; plain mean if empty",
            "center": "mean voxel centre (index + 0.5) times pitch, X Y Z",
            "size": "bounding-box voxel extent times pitch, X Y Z",
        },
        "dff_baseline": table.baseline.describe(),
    }
    return json.dumps(doc, indent=2) + "\n"
