"""Quantities derived from raw traces, computed when asked and never stored.

One quantity, ΔF/F.  The baseline F0 is, by default, a low percentile of F
in a sliding window centred on each frame; the
alternative is the mean of F over a fixed frame range.  Unmeasured frames are
NaN in and NaN out, and a window with no measured frame gives no baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SLIDING, FIXED = "sliding", "fixed"


@dataclass(frozen=True)
class Baseline:
    mode: str = SLIDING
    #: Sliding: the window in frames (odd; an even one is widened by one) and
    #: the percentile taken in it.
    window: int = 101
    percentile: float = 10.0
    #: Fixed: the first and last frame, inclusive.
    first: int = 0
    last: int = 9

    def describe(self) -> dict:
        if self.mode == FIXED:
            return {"mode": FIXED, "frames": [self.first, self.last]}
        return {"mode": SLIDING, "window_frames": self.width, "percentile": self.percentile}

    @property
    def width(self) -> int:
        return max(1, self.window | 1)


def baseline(f: np.ndarray, how: Baseline) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    if how.mode == FIXED:
        part = f[max(0, how.first): how.last + 1]
        level = np.nanmean(part) if np.isfinite(part).any() else np.nan
        return np.full(f.shape, level)
    half = how.width // 2
    padded = np.concatenate([np.full(half, np.nan), f, np.full(half, np.nan)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, how.width)
    out = np.full(f.shape, np.nan)
    has = np.isfinite(windows).any(axis=1)
    if has.any():
        out[has] = np.nanpercentile(windows[has], how.percentile, axis=1)
    return out


def dff(f: np.ndarray, how: Baseline) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    f0 = baseline(f, how)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (f - f0) / f0
    out[~np.isfinite(out)] = np.nan
    return out
