"""Label rasterisation: labels must land on exactly the pixels the geometry covers."""

import numpy as np
import pytest
from shapely.geometry import box

from roofphoto.data.chips import (
    EDGE_CLASSES,
    IGNORE,
    chip_grid,
    constant_shift,
    rasterize_labels,
    relief_shift,
)
from roofphoto.geo.raster import world_to_pixel
from roofphoto.io.lidar import Edge, Facet, HouseLabels

RES = 0.08
CX, CY = 50_160.0, 499_760.0


def flat_house(z=5.0, w=8.0, h=4.0):
    """One flat facet, w x h m, centred on (CX, CY), with an eave along its south side."""
    x0, y0 = CX - w / 2, CY - h / 2
    ring = np.array([[x0, y0, z], [x0 + w, y0, z], [x0 + w, y0 + h, z], [x0, y0 + h, z]])
    facet = Facet(id=0, polygon_xyz=ring, pitch_deg=0.0, area_m2=w * h, confidence="normal", fallback=False)
    eave = Edge(id=0, type="eave", p0=ring[0], p1=ring[1])
    return HouseLabels("t", np.zeros(3), box(x0, y0, x0 + w, y0 + h), [facet], [eave])


def test_chip_grid_is_on_the_photo_grid():
    t, n = chip_grid((CX + 0.013, CY - 0.029), 32.0, RES)
    assert n == 400
    assert (t.c / RES) == pytest.approx(round(t.c / RES))
    assert (t.f / RES) == pytest.approx(round(t.f / RES))
    assert abs(t.c + n * RES / 2 - (CX + 0.013)) <= RES / 2


def test_facet_covers_its_area_and_position():
    house = flat_house()
    t, n = chip_grid((CX, CY), 16.0, RES)
    lab = rasterize_labels(house, constant_shift(np.zeros(2)), t, n)
    assert (lab[0] == 1).sum() == round(8.0 / RES) * round(4.0 / RES)  # 100 x 50 pixels
    rows, cols = np.nonzero(lab[0] == 1)
    c0, r0 = world_to_pixel(t, CX - 4.0, CY + 2.0)  # facet's north-west corner
    assert cols.min() == round(float(c0)) and rows.min() == round(float(r0))


def test_constant_shift_moves_labels_by_whole_pixels():
    house = flat_house()
    t, n = chip_grid((CX, CY), 16.0, RES)
    a = rasterize_labels(house, constant_shift(np.zeros(2)), t, n)[0] == 1
    b = rasterize_labels(house, constant_shift(np.array([0.4, -0.8])), t, n)[0] == 1
    # +0.4 m east = +5 columns; -0.8 m north = +10 rows (rows run south).
    assert np.array_equal(np.roll(np.roll(a, 5, axis=1), 10, axis=0), b)


def test_relief_shift_scales_with_height_above_ground():
    t, n = chip_grid((CX, CY), 16.0, RES)
    v = np.array([0.0, 0.04])  # 0.04 m north per m of height
    low = rasterize_labels(flat_house(z=2.0), relief_shift(v, 0.0), t, n)[0] == 1
    high = rasterize_labels(flat_house(z=6.0), relief_shift(v, 0.0), t, n)[0] == 1
    # 2 m -> 0.08 m (1 px), 6 m -> 0.24 m (3 px) north: high sits 2 rows further north. (Whole-pixel
    # shifts: a half-pixel shift puts the boundary on pixel centres, where inside/outside is a tie.)
    assert np.nonzero(low)[0].min() - np.nonzero(high)[0].min() == 2


def test_edge_band_marks_the_eave_with_its_class():
    house = flat_house()
    t, n = chip_grid((CX, CY), 16.0, RES)
    edges = rasterize_labels(house, constant_shift(np.zeros(2)), t, n)[1]
    rows, cols = np.nonzero(edges == EDGE_CLASSES["eave"])
    _, r_eave = world_to_pixel(t, CX, CY - 2.0)
    assert set(np.unique(edges)) <= {0, EDGE_CLASSES["eave"]}
    assert abs(np.median(rows) - float(r_eave)) <= 1
    assert cols.max() - cols.min() + 1 == round(8.0 / RES)


def test_ignore_marks_neighbours_but_never_overwrites_labels():
    house = flat_house()
    t, n = chip_grid((CX, CY), 16.0, RES)
    neighbour = box(CX - 7.9, CY - 7.9, CX - 5.0, CY + 7.9)
    overlapping = box(CX - 3.0, CY - 1.0, CX + 3.0, CY + 1.0)  # inside the house itself
    lab = rasterize_labels(house, constant_shift(np.zeros(2)), t, n, ignore=neighbour.union(overlapping))
    assert (lab[0] == IGNORE).any()
    assert (lab[0] == 1).sum() == 100 * 50  # the house's own pixels are untouched
