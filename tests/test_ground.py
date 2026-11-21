"""Ground check on synthetic data: a 'photo' and a sparse 'LiDAR' sampling of the same scene."""

import numpy as np
import pytest
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

from roofphoto.geo.align import gradients
from roofphoto.geo.ground import filled_edges, rasterize_mean, search_shift
from roofphoto.geo.raster import pixel_to_world, sample

RES = 0.08
X0, Y0 = 50_150.0, 499_800.0
SIZE = 500  # pixels: a 40 m square


def scene():
    """A photo of random blobs and stripes (like paving and paint), and its transform."""
    rng = np.random.default_rng(3)
    img = gaussian_filter(rng.normal(size=(SIZE, SIZE)), 6) * 40
    img[100:110, :] += 60  # an east-west stripe
    img[:, 300:306] += 60  # a north-south stripe
    return img, from_origin(X0, Y0, RES, RES)


def lidar_points(img, t, shift, density=25.0, holes=True):
    """Points scattered over the scene. A point at (x, y) sees the feature the photo shows at (x, y) + shift,
    i.e. the photo is displaced by +shift relative to the LiDAR."""
    rng = np.random.default_rng(4)
    n = int(density * (SIZE * RES - 4) ** 2)
    x = rng.uniform(X0 + 2, X0 + SIZE * RES - 2, n)
    y = rng.uniform(Y0 - SIZE * RES + 2, Y0 - 2, n)
    if holes:  # a house-sized gap with no ground points
        keep = ~((np.abs(x - (X0 + 20)) < 5) & (np.abs(y - (Y0 - 20)) < 4))
        x, y = x[keep], y[keep]
    vals = sample(img, t, x + shift[0], y + shift[1])
    return np.column_stack([x, y]), vals


def test_rasterize_mean_puts_points_in_the_right_cell():
    xy = np.array([[X0 + 0.1, Y0 - 0.1], [X0 + 0.2, Y0 - 0.2], [X0 + 0.6, Y0 - 0.1]])
    mean, count, t = rasterize_mean(xy, np.array([1.0, 3.0, 10.0]), (X0, Y0 - 1.0, X0 + 1.0, Y0), 0.5)
    assert mean.shape == (2, 2)
    assert mean[0, 0] == pytest.approx(2.0) and count[0, 0] == 2
    assert mean[0, 1] == pytest.approx(10.0)
    assert np.isnan(mean[1, 0])
    assert t.c == pytest.approx(X0) and t.f == pytest.approx(Y0)


@pytest.mark.parametrize("shift", [(0.0, 0.0), (0.3, -0.5)])
def test_recovers_known_shift_through_sparse_points_and_holes(shift):
    img, t = scene()
    xy, vals = lidar_points(img, t, np.array(shift))
    bounds = (X0, Y0 - SIZE * RES, X0 + SIZE * RES, Y0)
    mean, _, ref_t = rasterize_mean(xy, vals, bounds, 0.25)
    ref, valid = filled_edges(mean)
    photo_edges = np.hypot(*gradients(img, sigma_px=0.25 / RES))
    res = search_shift(ref, valid, ref_t, photo_edges, t)
    np.testing.assert_allclose(res.shift, shift, atol=0.06)
    assert res.peak > res.at_zero or shift == (0.0, 0.0)


def test_hole_is_excluded_from_valid_cells():
    img, t = scene()
    xy, vals = lidar_points(img, t, np.zeros(2))
    mean, _, ref_t = rasterize_mean(xy, vals, (X0, Y0 - SIZE * RES, X0 + SIZE * RES, Y0), 0.25)
    _, valid = filled_edges(mean)
    rows, cols = np.nonzero(valid)
    x, y = pixel_to_world(ref_t, cols + 0.5, rows + 0.5)
    assert not ((np.abs(x - (X0 + 20)) < 4.5) & (np.abs(y - (Y0 - 20)) < 3.5)).any()


def test_compare_shifts_prefers_the_true_one():
    from roofphoto.geo.ground import compare_shifts

    img, t = scene()
    xy, vals = lidar_points(img, t, np.zeros(2))
    mean, _, ref_t = rasterize_mean(xy, vals, (X0, Y0 - SIZE * RES, X0 + SIZE * RES, Y0), 0.25)
    ref, valid = filled_edges(mean)
    photo_edges = np.hypot(*gradients(img, sigma_px=0.25 / RES))
    out = compare_shifts(ref, valid, ref_t, photo_edges, t, np.zeros(2), np.array([0.0, 0.6]), n_boot=200)
    assert out["ncc_a"] > out["ncc_b"]
    assert out["a_beats_b_fraction"] > 0.95
