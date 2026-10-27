"""Numeric parity with vizi's built-in Measure.

The fixture (``fixtures/parity``) was made with vizi's built-in Measure: the
dataset below written as an OME-TIFF (0.5 x 0.5 x 2.0 um, 0.25 s), four
regions drawn in it (a box, an off-lattice ellipse, a whole-volume box past
the 200,000-voxel sample cap, an off-lattice box), built-in Measure run over
every frame with every output on, and its CSV exported verbatim
(``builtin-measure.csv``).  ``host-masks.npz`` is what ``read_roi_inclusion``
returned for the same regions at t=0, so the check runs on the voxels the
host says each region covers, exactly as the plugin does.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from vizi_quant import stats

HERE = Path(__file__).parent / "fixtures" / "parity"
SPACING_ZYX = (2.0, 0.5, 0.5)
T, Z, C, Y, X = 4, 24, 2, 96, 96


def fixture_volume() -> np.ndarray:
    """The fixture dataset, TZCYX -- the generator the OME-TIFF was written
    from, so the test needs no TIFF reader."""
    rng = np.random.default_rng(42)
    z, y, x = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    a = np.empty((T, Z, C, Y, X), np.uint16)
    for t in range(T):
        for c in range(C):
            base = 100 + 50 * c + 3 * z + y + 2 * x + 40 * t
            noise = rng.gamma(2.0, 60.0 + 20 * c, size=(Z, Y, X))
            a[t, :, c] = np.clip(base + noise, 0, 65535).astype(np.uint16)
    return a


@pytest.fixture(scope="module")
def golden():
    rows = list(csv.reader((HERE / "builtin-measure.csv").open()))
    return rows[0], rows[1:]


@pytest.fixture(scope="module")
def regions():
    meta = json.loads((HERE / "host-masks.json").read_text())
    masks = np.load(HERE / "host-masks.npz")
    return {name: (masks[name.replace(" ", "_")], tuple(m["origin_zyx"])) for name, m in meta.items()}


def _cell(header, row, column):
    return row[header.index(column)]


@pytest.mark.parametrize("name", ["Box 1", "Box 2", "Box 3", "Box 4"])
def test_matches_builtin_measure(golden, regions, name):
    header, rows = golden
    volume = fixture_volume()
    mask, origin = regions[name]
    (oz, oy, ox), (dz, dy, dx) = origin, mask.shape
    geo = stats.geometry(mask, origin, SPACING_ZYX)
    for t, row in enumerate(rows):
        assert _cell(header, row, f"{name} n") == str(int(geo[0]))
        for i, axis in enumerate(("x", "y", "z")):
            assert _cell(header, row, f"{name} {axis} (um)") == f"{geo[1 + i]:.3f}", (t, axis)
            assert _cell(header, row, f"{name} size {axis} (um)") == f"{geo[4 + i]:.3f}", (t, axis)
        for c in range(C):
            window = volume[t, oz:oz + dz, c, oy:oy + dy, ox:ox + dx]
            got = stats.intensity(stats.region_values(window, mask))
            for k, stat in enumerate(stats.INTENSITY):
                want = _cell(header, row, f"{name} ch{c + 1} {stat}")
                assert f"{got[k]:.3f}" == want, (name, t, c, stat)


def test_the_whole_volume_box_is_subsampled(regions):
    """Box 3 is the one past the cap, so the stride rule is what it checks."""
    mask, _ = regions["Box 3"]
    assert mask.sum() > stats.SAMPLE_CAP


def test_empty_band_falls_back_to_mean():
    v = np.array([5.0, 5.0, 5.0, 100.0])
    assert stats.band_mean(np.array([7.0]), 85, 95) == 7.0
    assert np.isfinite(stats.intensity(v)).all()


def test_nothing_covered():
    assert np.isnan(stats.intensity(np.array([]))).all()
    geo = stats.geometry(np.zeros((2, 2, 2), bool), (0, 0, 0), (1, 1, 1))
    assert geo[0] == 0 and np.isnan(geo[1:]).all()
