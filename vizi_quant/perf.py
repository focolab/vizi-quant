"""The performance log: one JSON line a second while the plugin runs, for
measuring what a sweep costs.  Off unless ``VIZI_QUANT_PERF_LOG``
names a file; the host's own trace (``vizi --trace=DIR``) supplies the
viewer's side.  Each line: wall time, the plugin's CPU seconds and resident
memory, whether the sweep runs, and the engine's cumulative counters (host
volume reads, voxels read, inclusion reads, frames measured).

Portable: CPU time comes from ``time.process_time``; resident memory is read
from ``/proc`` where there is one (Linux) and is ``NaN`` elsewhere."""

from __future__ import annotations

import json
import os
import sys
import threading
import time


def start(get_engine) -> None:
    path = os.environ.get("VIZI_QUANT_PERF_LOG")
    if not path:
        return

    def loop() -> None:
        with open(path, "a", encoding="utf-8") as out:
            while True:
                out.write(json.dumps(sample(get_engine())) + "\n")
                out.flush()
                time.sleep(1.0)

    threading.Thread(target=loop, daemon=True, name="vizi-quant-perf").start()


def sample(engine) -> dict:
    """One line of the log."""
    line = {
        "time": time.time(),
        "cpu_s": time.process_time(),  # user + system CPU of this process
        "rss_mb": _rss_mb(),
        "running": bool(engine and engine.running),
        **(engine.counters.snapshot() if engine else {}),
    }
    if engine:
        line["done"], line["total"] = engine.progress()
    return line


def _rss_mb() -> float:
    if not sys.platform.startswith("linux"):
        return float("nan")
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, AttributeError):
        return float("nan")
