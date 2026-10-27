"""The background sweep: every region, every frame, every measured channel.

One worker thread measures frame by frame, always taking next the unmeasured
frame nearest the one on screen, so what the person is looking at fills in
first.  It pauses while the person scrubs (a frame change in the last
``SCRUB_QUIET_S``) and resumes when the slider rests.  It starts only when the
person presses Measure, at every dataset size -- opening a dataset measures
nothing -- and once asked it stays armed for the dataset, so
a region drawn or edited later is measured without asking again, until the
person presses Stop.  What was measured before is served from the cache
without asking.

The host says which voxels a region covers (``read_roi_inclusion``); a region
with no pose, no time span and no region parent covers the same voxels at
every frame, so it is rasterised once.  Per frame and channel the worker reads
either one window around every region that needs that frame, or one window per
region when the regions are sparse enough that the union would mostly be
empty space.

A region edit gives the region a new ``content_hash`` and so a new trace: the
sweep carries on, and the edited region is measured again from the frame on
screen outward.  Its old trace stays on disk for the shape it had.

Reads are the file's own voxels: every read asks for ``raw``, which vizi
answers with the file's voxels even while the person has *Plugins See
Filtered Data* on.  A read that comes back through a filter all the same
stops the sweep and says so, and nothing filtered is recorded.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from vizi_plugin_sdk import HostError, pb2

from . import stats
from .cache import Cache, Trace

#: How long the frame must rest before a paused sweep resumes.
SCRUB_QUIET_S = 0.75
#: The window around every region that needs a frame is read in one request
#: when it is at most this many bytes; past it, each region is read on its own.
#: The host's cost is per request -- on a 60-region time series, one small read
#: per region measures about 25 times slower than one read of the frame -- so
#: one read wins whenever it fits; the cap bounds the memory one window holds.
#: VIZI_QUANT_UNION_MAX_BYTES overrides it for the performance measurement.
UNION_MAX_BYTES = int(os.environ.get("VIZI_QUANT_UNION_MAX_BYTES", 64 * 1024 * 1024))
FLUSH_EVERY_S = 2.0


class Counters:
    """What the sweep asked of the host, for the performance measurement
    (``perf.py``); cheap enough to keep always."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reads = 0
        self.read_voxels = 0
        self.inclusions = 0
        self.frames = 0

    def add(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, getattr(self, k) + v)

    def snapshot(self) -> dict:
        with self.lock:
            return {"reads": self.reads, "read_voxels": self.read_voxels, "inclusions": self.inclusions, "frames": self.frames}


@dataclass
class Region:
    region_id: str
    name: str
    content_hash: str
    revision: int
    #: The same voxels at every frame: no pose, no time span, no region parent.
    static: bool
    trace: Trace = field(repr=False, default=None)


def is_static(roi) -> bool:
    if roi.pose_translation or roi.pose_rotation or roi.pose_stretch:
        return False
    if roi.HasField("t_start") or roi.HasField("t_end"):
        return False
    if roi.HasField("parent") and roi.parent.region_id:
        return False
    return True


def plan_windows(boxes, itemsize: int, cap_bytes: int):
    """Which windows to read for one frame and channel.  ``boxes`` are the
    regions' ``(lo_zyx, hi_zyx)`` (hi exclusive).  Returns ``[(lo, hi,
    [indices of the boxes inside])]``: the union of every box when it holds at
    most ``cap_bytes``, else each box alone."""
    if not boxes:
        return []
    lo = tuple(int(v) for v in np.min([b[0] for b in boxes], axis=0))
    hi = tuple(int(v) for v in np.max([b[1] for b in boxes], axis=0))
    if int(np.prod(np.subtract(hi, lo))) * itemsize <= cap_bytes:
        return [(lo, hi, list(range(len(boxes))))]
    return [(tuple(int(v) for v in b[0]), tuple(int(v) for v in b[1]), [i]) for i, b in enumerate(boxes)]


class FilteredRead(Exception):
    def __init__(self, applied):
        super().__init__("read came back filtered")
        self.applied = applied


class Engine:
    """Owns the regions of one open dataset and the worker that measures them.
    ``on_progress()`` is called (from the worker) whenever numbers land."""

    def __init__(self, host, view, cache: Cache, on_progress, on_status):
        self.host = host
        self.generation = int(view.open_generation)
        self.frames = int(view.shape_tczyx[0])
        self.n_channels = int(view.shape_tczyx[1])
        # Per axis: an axis whose voxel size the file does not record stays in voxels.
        self.spacing = tuple(1.0 if v is None else v for v in view.measured_spacing_zyx)
        self.itemsize = np.dtype(view.dtype).itemsize if view.dtype is not None else 2
        self.cache = cache
        self.on_progress = on_progress
        self.on_status = on_status
        self.lock = threading.RLock()
        self.wake = threading.Condition(self.lock)
        self.regions: dict[str, Region] = {}
        self.order: list[str] = []
        self.channels: list[int] = list(range(self.n_channels))
        self.current_t = int(view.t)
        self.last_move = 0.0
        self.running = False
        #: Pressed Measure and not Stop: new work restarts the sweep by itself.
        self.armed = False
        self.task = None
        self._task_factory = None
        self.stopped = False
        self.filtered = None  # the AppliedFilter a read came back through, if one did
        self._masks: dict[tuple[str, str], tuple[np.ndarray, tuple[int, int, int]]] = {}
        self.counters = Counters()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="vizi-quant-sweep")
        self._thread.start()

    # -- what the person and the host change ------------------------------------

    def set_regions(self, records, reset: bool, deleted=()) -> None:
        """Apply a ``RegionsChanged``: ``records`` are the changed regions'
        ``RoiRecord``s, read with ``get_rois(region_ids=...)``."""
        with self.lock:
            if reset:
                self.regions.clear()
                self.order.clear()
            for region_id in deleted:
                self.regions.pop(region_id, None)
                if region_id in self.order:
                    self.order.remove(region_id)
            for rec in records:
                region = Region(rec.region_id, rec.name, rec.content_hash, rec.revision, is_static(rec.roi))
                region.trace = self.cache.trace(rec.region_id, rec.content_hash)
                self.regions[rec.region_id] = region
                if rec.region_id not in self.order:
                    self.order.append(rec.region_id)
            if self.armed and records:
                self._resume()
            self.wake.notify_all()

    def view_moved(self, t: int) -> None:
        with self.lock:
            if t != self.current_t:
                self.current_t = t
                self.last_move = time.monotonic()
                self.wake.notify_all()

    def set_channels(self, channels) -> None:
        with self.lock:
            self.channels = sorted({int(c) for c in channels if 0 <= int(c) < self.n_channels})
            if self.armed:
                self._resume()
            self.wake.notify_all()

    def start(self, task_factory) -> None:
        """``task_factory(title)`` makes the host task a sweep reports to --
        the context's ``start_task``, kept so a later restart can make another."""
        with self.lock:
            self._task_factory = task_factory
            self.armed = True
            self.filtered = None
            self._resume()

    def _resume(self) -> None:
        """Run, with a task (lock held)."""
        if self.running:
            return
        self.task = self._task_factory("measuring regions") if self._task_factory else None
        self.running = True
        self.wake.notify_all()

    def stop(self, message: str = "stopped", *, disarm: bool = False) -> None:
        with self.lock:
            self.running = False
            if disarm:
                self.armed = False
            task, self.task = self.task, None
            self.wake.notify_all()
        if task is not None:
            task.done(message)

    def close(self) -> None:
        with self.lock:
            self.stopped = True
            self.running = False
            self.wake.notify_all()
        self._thread.join(timeout=5)
        self.cache.flush()

    # -- progress ------------------------------------------------------------------

    def progress(self) -> tuple[int, int]:
        """(frames measured, frames wanted) over every region."""
        with self.lock:
            regions, channels = list(self.regions.values()), list(self.channels)
        done = sum(r.trace.done_frames(channels) for r in regions)
        return done, len(regions) * self.frames

    def frame_progress(self) -> tuple[int, int]:
        """(frames on which every region is measured, frames) -- what the
        person watches; :meth:`progress` counts the work."""
        with self.lock:
            regions, channels = list(self.regions.values()), list(self.channels)
        need = np.zeros(self.frames, bool)
        for r in regions:
            need |= r.trace.needs(channels)
        return int((~need).sum()), self.frames

    def snapshot(self):
        """The regions in list order, for drawing and export (under the lock)."""
        with self.lock:
            return [self.regions[r] for r in self.order if r in self.regions], list(self.channels)

    # -- the worker -------------------------------------------------------------------

    def _next_frame(self) -> "int | None":
        """The unmeasured frame nearest the one on screen (earlier on a tie)."""
        need = np.zeros(self.frames, bool)
        for region in self.regions.values():
            need |= region.trace.needs(self.channels)
        if not need.any():
            return None
        candidates = np.flatnonzero(need)
        return int(candidates[np.argmin(np.abs(candidates - self.current_t) * 2 + (candidates > self.current_t))])

    def _loop(self) -> None:
        last_flush = time.monotonic()
        backoff = 0.1
        while True:
            with self.lock:
                while not self.stopped and not (self.running and self.channels and self.regions):
                    self.wake.wait(timeout=1.0)
                if self.stopped:
                    return
                quiet = time.monotonic() - self.last_move
                if quiet < SCRUB_QUIET_S:
                    self.wake.wait(timeout=SCRUB_QUIET_S - quiet)
                    continue
                if self.task is not None and self.task.cancelled:
                    pending_stop = "cancelled"
                else:
                    pending_stop = None
                t = None if pending_stop else self._next_frame()
                todo = [] if t is None else [r for r in self.regions.values() if r.trace.needs(self.channels)[t]]
                channels = list(self.channels)
            if pending_stop:
                self.stop(pending_stop, disarm=True)
                continue
            if t is None:
                self.cache.flush()
                self.stop("every frame measured")
                self.on_progress()
                continue
            try:
                self._measure_frame(t, todo, channels)
                self.counters.add(frames=1)
                backoff = 0.1
            except FilteredRead as exc:
                with self.lock:
                    self.filtered = exc.applied
                self.stop("reads are filtered", disarm=True)
                self.on_status(
                    "Stopped: vizi is handing plugins filtered data (Plugins See Filtered Data). "
                    "vizi-quant measures the file's own voxels only; turn that off to measure."
                )
                continue
            except HostError as exc:
                self.on_status(f"Measuring failed at frame {t}: {exc} (retrying)")
                time.sleep(backoff)
                backoff = min(5.0, backoff * 2)
                continue
            self._report_task()
            self.on_progress()
            if time.monotonic() - last_flush > FLUSH_EVERY_S:
                self.cache.flush()
                last_flush = time.monotonic()

    def _report_task(self) -> None:
        task = self.task
        if task is None:
            return
        done, total = self.progress()
        if total:
            frame, frames = self.frame_progress()
            task.progress(done / total, f"frame {frame:,} / {frames:,}")

    def _mask(self, region: Region, t: int):
        """``(mask, origin)`` or ``None`` when the region is absent at ``t``."""
        key = (region.region_id, region.content_hash)
        if region.static and key in self._masks:
            return self._masks[key]
        found = self.host.read_roi_inclusion(region.region_id, t)
        self.counters.add(inclusions=1)
        if found is not None and region.static:
            self._masks[key] = found
        return found

    def _measure_frame(self, t: int, todo: list[Region], channels: list[int]) -> None:
        footprints = []
        for region in todo:
            try:
                found = self._mask(region, t)
            except HostError as exc:
                if exc.kind == pb2.Error.NOT_FOUND:
                    continue  # deleted meanwhile; the RegionsChanged that says so drops it
                raise
            if found is None or found[0].size == 0 or not np.asarray(found[0]).any():
                self._commit(region, lambda tr: tr.record_absent(t))
                continue
            mask, origin = np.asarray(found[0], dtype=bool), tuple(int(o) for o in found[1])
            geom = stats.geometry(mask, origin, self.spacing)
            self._commit(region, lambda tr, g=geom: tr.record_geometry(t, g))
            footprints.append((region, mask, origin))
        if not footprints:
            return
        for c in channels:
            wanted = [f for f in footprints if f[0].trace.value_state[c, t] == 0]
            for window_lo, window, members in self._windows(t, c, wanted):
                for region, mask, origin in members:
                    rel = tuple(o - w for o, w in zip(origin, window_lo))
                    sub = window[rel[0]:rel[0] + mask.shape[0], rel[1]:rel[1] + mask.shape[1], rel[2]:rel[2] + mask.shape[2]]
                    values = stats.intensity(stats.region_values(sub, mask))
                    self._commit(region, lambda tr, v=values: tr.record_values(c, t, v))

    def _windows(self, t: int, c: int, footprints):
        """Yield ``(window origin, window, regions in it)``, as
        :func:`plan_windows` divides the regions."""
        boxes = [(f[2], tuple(int(v) for v in np.add(f[2], f[1].shape))) for f in footprints]
        for lo, hi, members in plan_windows(boxes, self.itemsize, UNION_MAX_BYTES):
            yield lo, self._read(t, c, lo, hi), [footprints[i] for i in members]

    def _read(self, t, c, lo, hi) -> np.ndarray:
        # raw: the file's own voxels even under Plugins See Filtered Data.  A
        # reply that comes back filtered all the same is refused below.
        volume = self.host.read_volume(
            t, c, 0, roi=(tuple(int(v) for v in lo), tuple(int(v) for v in hi)), raw=True
        )
        self.counters.add(reads=1, read_voxels=int(volume.size))
        if getattr(volume, "filter", None) is not None:
            raise FilteredRead(volume.filter)
        return volume

    def _commit(self, region: Region, write) -> None:
        """Write into the region's trace only if it is still that region: an
        edit meanwhile replaced it, and these numbers belong to the old shape."""
        with self.lock:
            current = self.regions.get(region.region_id)
            if current is None or current.content_hash != region.content_hash:
                return
            write(current.trace)
