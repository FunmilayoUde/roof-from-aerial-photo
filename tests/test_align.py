"""Alignment on synthetic images where the true shift is known."""

from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.transform import from_origin

from roofphoto.geo.align import edge_pieces, fit_shift, gradients, search_offsets
from roofphoto.geo.raster import world_to_pixel

RES = 0.08
X0, Y0 = 50_150.0, 499_780.0
RECT = (50_155.0, 499_765.0, 50_165.0, 499_773.0)  # a 10 x 8 m "roof", world bounds


def rect_edges(bounds, z=6.0):
    xmin, ymin, xmax, ymax = bounds
    c = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    return [
        SimpleNamespace(id=i, type="eave", p0=np.array([*c[i], z]), p1=np.array([*c[(i + 1) % 4], z])) for i in range(4)
    ]


def render_polygon(corners, shift, supersample=4):
    """A bright convex polygon (corners counter-clockwise, world xy) drawn at corners + shift, anti-aliased."""
    t = from_origin(X0, Y0, RES, RES)
    h, w = 300, 300
    sub = (np.arange(supersample) + 0.5) / supersample
    cols, rows = np.meshgrid(np.add.outer(np.arange(w), sub).ravel(), np.add.outer(np.arange(h), sub).ravel())
    x, y = t @ (cols, rows)
    c = np.asarray(corners) + shift
    inside = np.ones_like(x, dtype=bool)
    for a, b in zip(c, np.roll(c, -1, axis=0)):
        inside &= (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) >= 0
    coverage = inside.reshape(h, supersample, w, supersample).mean(axis=(1, 3))
    return (50.0 + 150.0 * coverage)[None].repeat(3, axis=0), t


def render_rect(bounds, shift, supersample=4):
    """A bright rectangle on a dark background, drawn at bounds + shift.

    Each pixel is the average of supersample x supersample sub-samples, so an edge that falls
    inside a pixel gives an intermediate value, as in a real photo, and its position is known
    to well under a pixel.
    """
    t = from_origin(X0, Y0, RES, RES)
    h, w = 300, 300
    sub = (np.arange(supersample) + 0.5) / supersample
    cols, rows = np.meshgrid(np.add.outer(np.arange(w), sub).ravel(), np.add.outer(np.arange(h), sub).ravel())
    x, y = t @ (cols, rows)
    xmin, ymin, xmax, ymax = bounds
    dx, dy = shift
    inside = (x >= xmin + dx) & (x <= xmax + dx) & (y >= ymin + dy) & (y <= ymax + dy)
    coverage = inside.reshape(h, supersample, w, supersample).mean(axis=(1, 3))
    return (50.0 + 150.0 * coverage)[None].repeat(3, axis=0), t


def test_synthetic_image_is_where_we_think():
    img, t = render_rect(RECT, (0, 0))
    col, row = world_to_pixel(t, 50_160.0, 499_769.0)  # rectangle centre
    assert img[0, int(row), int(col)] == 200.0


@pytest.mark.parametrize("shift", [(0.0, 0.0), (0.3, -0.2), (-0.5, 0.45)])
def test_recovers_constant_shift(shift):
    img, t = render_rect(RECT, shift)
    pieces = edge_pieces(rect_edges(RECT))
    res = search_offsets(gradients(img), t, pieces)
    fit = fit_shift(np.array([r.piece.normal for r in res]), np.array([r.offset_m for r in res]))
    np.testing.assert_allclose(fit.s0, shift, atol=0.03)


def test_offsets_are_signed_along_the_normal():
    img, t = render_rect(RECT, (0.4, 0.0))
    pieces = edge_pieces(rect_edges(RECT))
    res = search_offsets(gradients(img), t, pieces)
    # Vertical edges carry the x shift: offset = n . shift, so its sign follows the normal.
    across = [r for r in res if abs(r.piece.normal[0]) > 0.9]
    assert across
    for r in across:
        assert r.offset_m == pytest.approx(0.4 * r.piece.normal[0], abs=0.03)
    # Horizontal edges see no shift across them. Pieces at the corners can lock onto the
    # neighbouring edge when the shift exceeds the end trim; the robust fit drops those.
    along = [r.offset_m for r in res if abs(r.piece.normal[1]) > 0.9]
    assert np.median(along) == pytest.approx(0.0, abs=0.03)


def test_fit_separates_constant_shift_from_height_term():
    rng = np.random.default_rng(1)
    s0, v = np.array([0.05, -0.02]), np.array([0.04, 0.06])
    angles = rng.uniform(0, np.pi, 200)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    h = rng.uniform(3, 9, 200)
    d = np.einsum("ij,ij->i", normals, s0 + h[:, None] * v) + rng.normal(0, 0.02, 200)
    d[:10] += 0.8  # pieces that locked onto the wrong line
    fit = fit_shift(normals, d, h)
    np.testing.assert_allclose(fit.s0, s0, atol=0.03)
    np.testing.assert_allclose(fit.v, v, atol=0.006)
    assert not fit.inliers[:10].any()


def test_recovers_shift_on_diagonal_edges():
    """A diamond: every edge is diagonal (the gradient sign itself is pinned by the test below)."""
    cx, cy, r = 50_160.0, 499_769.0, 5.0
    corners = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    shift = np.array([0.35, -0.25])
    img, t = render_polygon(corners, shift)
    edges = [
        SimpleNamespace(id=i, type="hip", p0=np.array([*corners[i], 6.0]), p1=np.array([*corners[(i + 1) % 4], 6.0]))
        for i in range(4)
    ]
    res = search_offsets(gradients(img), t, edge_pieces(edges))
    fit = fit_shift(np.array([r.piece.normal for r in res]), np.array([r.offset_m for r in res]))
    np.testing.assert_allclose(fit.s0, shift, atol=0.03)


def test_gradient_points_north_when_image_brightens_northward():
    img = np.repeat(np.linspace(200, 0, 50)[:, None], 40, axis=1)  # bright at the top row = north
    gx, gy = gradients(img)
    assert gy[25, 20] > 0 and abs(gx[25, 20]) < 1e-9


def test_fit_through_ground_recovers_v_with_s0_fixed_at_zero():
    rng = np.random.default_rng(2)
    v = np.array([0.02, -0.12])
    # Only two edge directions, as on a rectangular house: the case where the free-s0 fit is unstable.
    normals = np.repeat([[0.36, 0.93], [0.93, -0.36]], [40, 6], axis=0)
    h = rng.uniform(3, 8.5, 46)
    d = np.einsum("ij,ij->i", normals, h[:, None] * v) + rng.normal(0, 0.03, 46)
    fit = fit_shift(normals, d, h, through_ground=True)
    np.testing.assert_allclose(fit.s0, 0.0)
    np.testing.assert_allclose(fit.v, v, atol=0.01)
