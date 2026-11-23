"""Straightening on synthetic noisy shapes with known true outlines."""

import numpy as np
import pytest
from rasterio.features import rasterize
from shapely.affinity import rotate
from shapely.geometry import Polygon, box

from roofphoto.regularise import _snap, axis_angle_px, regularise_facets, regularise_polygon


def noisy(poly: Polygon, rng, amp=2.0, n=400) -> Polygon:
    """Push the outline in and out by smooth waves, like a segmentation boundary (bulges and dents
    a few pixels deep, not pixel-by-pixel jitter), plus a little fine noise."""
    ring = poly.exterior
    t = np.linspace(0, 1, n, endpoint=False)
    pts = np.array([[ring.interpolate(x, normalized=True).x, ring.interpolate(x, normalized=True).y] for x in t])
    offset = sum(amp / 2 * np.sin(2 * np.pi * k * t + rng.uniform(0, 2 * np.pi)) for k in (5, 9, 13))
    offset += rng.normal(0, 0.3, n)
    centre = np.array(poly.centroid.coords[0])
    radial = (pts - centre) / np.linalg.norm(pts - centre, axis=1, keepdims=True)
    return Polygon(pts + offset[:, None] * radial).buffer(0)


def side_angles(poly: Polygon) -> np.ndarray:
    pts = np.asarray(poly.exterior.coords)
    v = np.diff(pts, axis=0)
    return np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 180


def test_snap_to_axis_and_diagonals():
    axis = np.radians(20)
    assert np.degrees(_snap(np.radians(24), axis, np.radians(15))) == pytest.approx(20)
    assert np.degrees(_snap(np.radians(112), axis, np.radians(15))) == pytest.approx(110)
    assert np.degrees(_snap(np.radians(63), axis, np.radians(15))) == pytest.approx(65)  # a hip
    assert _snap(np.radians(40), axis, np.radians(3)) is None


def test_noisy_rotated_rectangle_becomes_a_clean_rectangle():
    rng = np.random.default_rng(0)
    true = rotate(box(100, 100, 250, 180), 20, origin="centroid")
    reg = regularise_polygon(noisy(true, rng), np.radians(20))
    assert len(reg.exterior.coords) - 1 == 4
    angles = side_angles(reg)
    assert all(min(abs(a - 20) % 180, abs(a - 110) % 180) < 0.5 for a in angles)
    assert reg.symmetric_difference(true).area / true.area < 0.08


def test_l_shape_keeps_its_corners():
    rng = np.random.default_rng(1)
    true = Polygon([(100, 100), (300, 100), (300, 160), (180, 160), (180, 260), (100, 260)])
    reg = regularise_polygon(noisy(true, rng), 0.0)
    assert len(reg.exterior.coords) - 1 == 6
    assert reg.symmetric_difference(true).area / true.area < 0.08


def test_axis_from_footprint_in_image_coordinates():
    from rasterio.transform import from_origin

    t = from_origin(0, 100, 1, 1)
    # A footprint whose long side runs 30° north of east in the world: in the image (rows south)
    # that is -30°.
    fp = rotate(box(10, 40, 70, 60), 30, origin="centroid")
    assert np.degrees(axis_angle_px(fp, t)) % 180 == pytest.approx(150, abs=0.5)


def test_facets_stay_adjacent_after_straightening():
    rng = np.random.default_rng(2)
    a = noisy(box(50, 50, 200, 100), rng)
    b = noisy(box(50, 100, 200, 150), rng)
    facets = np.zeros((220, 260), dtype=np.int32)
    facets[rasterize([(a, 1)], out_shape=facets.shape) == 1] = 1
    facets[rasterize([(b, 1)], out_shape=facets.shape) == 1] = 2
    reg = regularise_facets(facets, 0.0)
    assert set(np.unique(reg)) == {0, 1, 2}
    # No gap between them: every roof pixel column through the middle goes straight from 1 to 2.
    col = reg[:, 125]
    roof = col[col > 0]
    assert (np.diff(roof.astype(int)) >= 0).all() and set(roof) == {1, 2}
