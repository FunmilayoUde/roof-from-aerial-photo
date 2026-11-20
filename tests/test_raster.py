"""Coordinate transforms: if these are wrong, every overlay is wrong by the same amount."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from roofphoto.geo.raster import crop_window, grid_for_bounds, pixel_to_world, sample, snap_bounds, world_to_pixel
from roofphoto.io.pdok import write_geotiff

RES = 0.08
X0, Y0 = 50_150.0, 499_770.0  # top-left corner, at real RD New magnitudes


def test_snap_bounds_grows_outward_to_grid():
    assert snap_bounds((50152.647, 499751.967, 50167.013, 499765.813), RES) == pytest.approx(
        (50152.64, 499751.92, 50167.04, 499765.84)
    )


def test_snap_bounds_keeps_bounds_already_on_grid():
    # 50152.64 / 0.08 is 626908 up to float noise; it must not snap a pixel further out.
    b = (50152.64, 499751.92, 50167.04, 499765.84)
    assert snap_bounds(b, RES) == pytest.approx(b)


def test_grid_covers_snapped_bounds():
    t, w, h = grid_for_bounds((X0, Y0 - 1.0, X0 + 2.0, Y0), RES)
    # 1.0 m is 12.5 pixels, so the bottom edge snaps out to 13 pixels (1.04 m).
    assert (w, h) == (25, 13)
    assert t.c == pytest.approx(X0) and t.f == pytest.approx(Y0)
    assert t.c + w * RES == pytest.approx(X0 + 2.0)
    assert t.f - h * RES == pytest.approx(Y0 - 1.04)


def test_pixel_world_round_trip():
    t = from_origin(X0, Y0, RES, RES)
    col, row = np.array([0.0, 10.5, 123.25]), np.array([0.0, 3.5, 77.75])
    x, y = pixel_to_world(t, col, row)
    c2, r2 = world_to_pixel(t, x, y)
    np.testing.assert_allclose(c2, col, atol=1e-6)
    np.testing.assert_allclose(r2, row, atol=1e-6)


def test_rows_increase_southward():
    t = from_origin(X0, Y0, RES, RES)
    x, y = pixel_to_world(t, 0.5, 0.5)  # centre of the top-left pixel
    assert x == pytest.approx(X0 + RES / 2)
    assert y == pytest.approx(Y0 - RES / 2)
    _, row = world_to_pixel(t, X0, Y0 - 1.0)  # 1 m south
    assert row == pytest.approx(1.0 / RES)


def test_sample_at_pixel_centre_returns_that_pixel():
    t = from_origin(X0, Y0, RES, RES)
    img = np.zeros((20, 30))
    img[7, 11] = 5.0
    x, y = pixel_to_world(t, 11.5, 7.5)
    assert sample(img, t, x, y) == pytest.approx(5.0)
    # Half a pixel east is halfway to the neighbour: bilinear gives half.
    assert sample(img, t, x + RES / 2, y) == pytest.approx(2.5)


def test_sample_outside_is_nan():
    t = from_origin(X0, Y0, RES, RES)
    assert np.isnan(sample(np.ones((5, 5)), t, X0 - 1.0, Y0))


def test_crop_window_transform_maps_to_same_world_points():
    t = from_origin(X0, Y0, RES, RES)
    img = np.arange(100 * 120, dtype=float).reshape(100, 120)
    rows, cols, t2 = crop_window(t, img.shape, (X0 + 1.0, Y0 - 3.0, X0 + 2.0, Y0 - 2.0))
    crop = img[rows, cols]
    # 1 m starting mid-pixel (12.5 px in) covers pixels 12..24 in each direction.
    assert crop.shape == (13, 13)
    assert t2.c == pytest.approx(X0 + 12 * RES) and t2.f == pytest.approx(Y0 - 25 * RES)
    # Any world point inside the crop reads the same value from crop and original.
    x, y = X0 + 1.33, Y0 - 2.41
    assert sample(crop, t2, x, y) == pytest.approx(sample(img, t, x, y))


def test_crop_window_outside_raises():
    t = from_origin(X0, Y0, RES, RES)
    with pytest.raises(ValueError):
        crop_window(t, (10, 10), (X0 + 100, Y0 - 1, X0 + 101, Y0))


def test_geotiff_round_trip_keeps_georeferencing(tmp_path):
    t = from_origin(X0, Y0, RES, RES)
    bands = np.zeros((3, 40, 50), dtype=np.uint8)
    bands[:, 12, 34] = 200
    path = tmp_path / "photo.tif"
    write_geotiff(path, bands, t)
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 28992
        assert src.transform.almost_equals(t)
        x, y = pixel_to_world(src.transform, 34.5, 12.5)
        assert next(src.sample([(float(x), float(y))]))[0] == 200
