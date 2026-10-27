"""What one region measures at one frame: pure functions, no host.

Every convention here is vizi's built-in Measure's, as its own export shows
(``tests/fixtures/parity``), and the parity test holds them there:

- intensity statistics over at most ``SAMPLE_CAP`` voxels, taken as a stride
  of ``ceil(n / SAMPLE_CAP)`` through the region's voxels in z, y, x order;
  the count ``n`` is always exact;
- ``std`` is the population standard deviation;
- a band (``p85-95``, ``p05-15``) is the mean of the values between the two
  percentiles (numpy's default linear method), both ends inclusive, and the
  plain mean when the band is empty;
- the centre is the mean voxel CENTRE (index + 0.5) times the pitch, and the
  size is the voxel extent of the region's bounding box times the pitch, both
  reported X, Y, Z.

``STATS_VERSION`` keys the cache: change any number this module produces and
bump it, or a cache written by the old code is served as if it were new.
"""

from __future__ import annotations

import numpy as np

STATS_VERSION = 1
SAMPLE_CAP = 200_000

#: The intensity statistics, in the order built-in Measure writes them.
INTENSITY = ("mean", "p85-95", "p05-15", "std")
#: Geometry, channel-independent: count, centre X Y Z, size X Y Z.
GEOMETRY = ("n", "x", "y", "z", "size x", "size y", "size z")


def subsample(values: np.ndarray) -> np.ndarray:
    n = values.size
    if n <= SAMPLE_CAP:
        return values
    return values[:: -(-n // SAMPLE_CAP)]


def band_mean(values: np.ndarray, lo: float, hi: float) -> float:
    a, b = np.percentile(values, [lo, hi])
    inside = values[(values >= a) & (values <= b)]
    return float(inside.mean()) if inside.size else float(values.mean())


def intensity(values: np.ndarray) -> np.ndarray:
    """``INTENSITY`` over ``values`` (the region's voxels in z, y, x order);
    NaN for a region that covers nothing."""
    if values.size == 0:
        return np.full(len(INTENSITY), np.nan)
    v = subsample(np.asarray(values, dtype=np.float64))
    return np.array([v.mean(), band_mean(v, 85, 95), band_mean(v, 5, 15), v.std()])


def geometry(mask: np.ndarray, origin_zyx, spacing_zyx) -> np.ndarray:
    """``GEOMETRY`` of a bbox-local boolean mask at level-0 ``origin_zyx``.
    ``spacing_zyx`` is the pitch, or ones for voxel units."""
    zz, yy, xx = np.nonzero(mask)
    n = zz.size
    if n == 0:
        return np.array([0.0] + [np.nan] * 6)
    sz, sy, sx = (float(s) for s in spacing_zyx)
    oz, oy, ox = (float(o) for o in origin_zyx)
    centre = ((xx.mean() + ox + 0.5) * sx, (yy.mean() + oy + 0.5) * sy, (zz.mean() + oz + 0.5) * sz)
    size = (
        (int(xx.max()) - int(xx.min()) + 1) * sx,
        (int(yy.max()) - int(yy.min()) + 1) * sy,
        (int(zz.max()) - int(zz.min()) + 1) * sz,
    )
    return np.array([float(n), *centre, *size])


def region_values(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The voxels of ``volume`` under ``mask`` (same shape), in z, y, x order
    -- the order the stride subsample is defined over."""
    return np.asarray(volume)[np.asarray(mask, dtype=bool)]
