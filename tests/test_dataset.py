"""Augmentation must move orientation labels exactly as it moves the roof they describe."""

import itertools

import numpy as np
import pytest

from roofphoto.models.dataset import IGNORE, augment, facet_class, octant


def downhill_octant(height: np.ndarray) -> int:
    """Octant a height field slopes down towards, from its gradient (rows run south, columns east)."""
    d_row, d_col = np.gradient(height)
    east, north = -d_col.mean(), d_row.mean()  # downhill = minus the gradient; north is up (-rows)
    return octant(float(np.degrees(np.arctan2(east, north)) % 360))


def test_octants_and_classes():
    assert [octant(a) for a in (0, 22, 23, 90, 180, 270, 337, 338)] == [0, 0, 1, 2, 4, 6, 7, 0]
    assert facet_class(3.0, 90.0) == 1  # under 5°: flat, whatever the azimuth
    assert facet_class(40.0, 0.0) == 2  # north
    assert facet_class(40.0, 315.0) == 9  # north-west


@pytest.mark.parametrize("az", [0, 45, 90, 135, 180, 225, 270, 315])
def test_every_rotation_and_flip_remaps_orientation_like_the_roof(az):
    n = 16
    rows, cols = np.mgrid[0:n, 0:n].astype(float)
    east, north = np.sin(np.radians(az)), np.cos(np.radians(az))
    height = -(cols * east - rows * north)  # a plane sloping down towards az
    assert downhill_octant(height) == octant(az)  # the synthetic roof faces the way we think
    orient = np.full((n, n), facet_class(40.0, az), dtype=np.uint8)
    orient[0, 0] = IGNORE
    for k, lr, ud in itertools.product(range(4), (False, True), (False, True)):
        img, o, (h,) = augment(height[None], orient, [height.astype(np.float32)], k, lr, ud)
        expected = 2 + downhill_octant(img[0])
        assert set(np.unique(o)) == {expected, IGNORE}, (az, k, lr, ud)
        np.testing.assert_allclose(h, img[0], atol=1e-5)  # pitch moves with the image
