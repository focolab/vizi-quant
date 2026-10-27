"""Raw measurements, per region, kept beside the dataset across sessions.

A ``Trace`` is one region's raw numbers over every frame, for one shape of
that region: the key is ``(region_id, content_hash)`` plus the stats-code
version, and the channel indexes inside it.  ``content_hash`` alone is not a
key -- two regions with identical geometry share one -- and ``region_id``
alone would serve numbers measured through a shape the region no longer has.

The store is the plugin's folder for the open file (``file_state_dir``), one
``.npz`` per trace, written whole through a temporary and a rename so a crash
mid-write leaves the previous file.  Each file carries the dataset identity it
was measured on (name, series, shape, dtype, pitch); a file whose identity or
stats version differs is ignored, never served.  With no folder (a dataset
with no local path) the cache lives in memory for the session.

Derived quantities are never stored here: they are functions of these numbers.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from . import stats

UNMEASURED, MEASURED, ABSENT = 0, 1, 2


class Trace:
    def __init__(self, frames: int, channels: int):
        self.geom = np.full((frames, len(stats.GEOMETRY)), np.nan)
        self.geom_state = np.zeros(frames, np.int8)
        self.values = np.full((channels, frames, len(stats.INTENSITY)), np.nan)
        self.value_state = np.zeros((channels, frames), np.int8)
        #: Bumped on every write, so a reader can tell its derived copy is stale.
        self.version = 0
        self.dirty = False

    @property
    def frames(self) -> int:
        return self.geom_state.size

    def needs(self, channels) -> np.ndarray:
        """Per frame: is anything in ``channels`` (or the geometry) missing."""
        need = self.geom_state == UNMEASURED
        for c in channels:
            need |= self.value_state[c] == UNMEASURED
        return need

    def done_frames(self, channels) -> int:
        return int((~self.needs(channels)).sum())

    def record_absent(self, t: int) -> None:
        self.geom_state[t] = ABSENT
        self.value_state[:, t] = ABSENT
        self._touch()

    def record_geometry(self, t: int, geom: np.ndarray) -> None:
        self.geom[t] = geom
        self.geom_state[t] = MEASURED
        self._touch()

    def record_values(self, c: int, t: int, values: np.ndarray) -> None:
        self.values[c, t] = values
        self.value_state[c, t] = MEASURED
        self._touch()

    def _touch(self) -> None:
        self.version += 1
        self.dirty = True


def identity_of(view) -> dict:
    """What a cached number depends on besides the region: the dataset and
    its calibration (the pitch moves every centre and size)."""
    # None on an axis the file did not record (never NaN: NaN != NaN would
    # make every cache file look like another dataset's).
    spacing = list(view.measured_spacing_zyx) if any(view.axis_known_zyx) else None
    return {
        "dataset_name": view.dataset_name,
        "dataset_series": getattr(view, "dataset_series", None),
        "shape_tczyx": list(view.shape_tczyx),
        "dtype": str(view.dtype),
        "spacing_zyx": spacing,
        "spacing_unit": view.spacing_unit if spacing else "",
        "stats_version": stats.STATS_VERSION,
    }


class Cache:
    def __init__(self, directory: "Path | None", identity: dict):
        self.identity = identity
        self.frames, self.channels = identity["shape_tczyx"][0], identity["shape_tczyx"][1]
        self.directory = None if directory is None else Path(directory) / f"cache-v{stats.STATS_VERSION}"
        self._traces: dict[tuple[str, str], Trace] = {}
        self._lock = threading.Lock()

    def trace(self, region_id: str, content_hash: str) -> Trace:
        key = (region_id, content_hash)
        with self._lock:
            found = self._traces.get(key)
            if found is None:
                found = self._load(key) or Trace(self.frames, self.channels)
                self._traces[key] = found
            return found

    def _path(self, key) -> "Path | None":
        if self.directory is None:
            return None
        region_id, content_hash = key
        return self.directory / f"{region_id}-{content_hash}.npz"

    def _load(self, key) -> "Trace | None":
        path = self._path(key)
        if path is None or not path.exists():
            return None
        try:
            with np.load(path) as data:
                if json.loads(str(data["identity"])) != self.identity:
                    return None
                trace = Trace(self.frames, self.channels)
                trace.geom[:] = data["geom"]
                trace.geom_state[:] = data["geom_state"]
                trace.values[:] = data["values"]
                trace.value_state[:] = data["value_state"]
                return trace
        except (OSError, ValueError, KeyError):
            # A cache is an optimisation: a file it cannot read is measured again.
            return None

    def flush(self) -> None:
        """Write every trace changed since the last flush."""
        if self.directory is None:
            return
        with self._lock:
            pending = [(k, t) for k, t in self._traces.items() if t.dirty]
        if not pending:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        for key, trace in pending:
            trace.dirty = False
            path = self._path(key)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(
                tmp,
                identity=np.array(json.dumps(self.identity)),
                geom=trace.geom,
                geom_state=trace.geom_state,
                values=trace.values,
                value_state=trace.value_state,
            )
            os.replace(tmp, path)
