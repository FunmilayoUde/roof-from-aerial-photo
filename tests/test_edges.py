"""Edge typing on synthetic roofs whose edges are known."""

import numpy as np

from roofphoto.data.chips import EDGE_CLASSES
from roofphoto.edges import downhill, edge_raster, facets_from_orientation, score_edges

N, E, S, W = 2, 4, 6, 8  # orientation classes facing north, east, south, west
FLAT = 1


def gable(n=60):
    """A gable roof, ridge running east-west through the middle: north half faces N, south half S."""
    orient = np.zeros((n, n), dtype=int)
    orient[10:30, 10:50] = N
    orient[30:50, 10:50] = S
    return orient > 0, orient


def types_at(edges, r, c):
    inv = {v: k for k, v in EDGE_CLASSES.items()}
    return inv.get(int(edges[r, c]))


def test_downhill_directions():
    np.testing.assert_allclose(downhill(N), [-1, 0], atol=1e-12)  # north = up the image
    np.testing.assert_allclose(downhill(E), [0, 1], atol=1e-12)  # east = right
    assert downhill(FLAT) is None


def test_gable_gives_ridge_eaves_and_rakes():
    roof, orient = gable()
    facets, cls = facets_from_orientation(roof, orient, min_px=10)
    assert sorted(cls.values()) == [N, S]
    edges = edge_raster(facets, cls)
    assert types_at(edges, 29, 30) == "ridge"  # between the halves
    assert types_at(edges, 10, 30) == "eave"  # north facet's lower (outer) edge
    assert types_at(edges, 49, 30) == "eave"
    assert types_at(edges, 20, 10) == "rake"  # gable ends run along the slope
    assert types_at(edges, 40, 49) == "rake"


def test_facets_sloping_towards_each_other_meet_in_a_valley():
    orient = np.zeros((60, 60), dtype=int)
    orient[10:30, 10:50] = S  # north half slopes down towards the middle
    orient[30:50, 10:50] = N  # south half too
    facets, cls = facets_from_orientation(orient > 0, orient, min_px=10)
    edges = edge_raster(facets, cls)
    assert types_at(edges, 29, 30) == "valley"


def test_hip_between_facets_ninety_degrees_apart():
    """The south-east corner of a hipped roof: the hip runs from the corner up to the north-west,
    the south facet lies below-left of it and the east facet above-right, both sloping away."""
    orient = np.zeros((60, 60), dtype=int)
    orient[10:50, 10:50] = S
    rr, cc = np.mgrid[0:60, 0:60]
    orient[(orient > 0) & (cc > rr)] = E
    facets, cls = facets_from_orientation(orient > 0, orient, min_px=10)
    edges = edge_raster(facets, cls)
    on_hip = [types_at(edges, r, c) for r in range(15, 45) for c in (r, r + 1)]
    assert on_hip.count("hip") > 20 and "valley" not in on_hip and "break" not in on_hip


def test_flat_roof_edges():
    orient = np.zeros((40, 40), dtype=int)
    orient[10:30, 10:30] = FLAT
    orient[10:30, 30:35] = S  # a south-facing lean-to beside it
    facets, cls = facets_from_orientation(orient > 0, orient, min_px=10)
    edges = edge_raster(facets, cls)
    assert types_at(edges, 10, 20) == "perimeter"
    assert types_at(edges, 20, 29) == "break"


def test_small_fragments_are_merged():
    roof, orient = gable()
    orient[20:23, 20:23] = E  # 9 px of speckle inside the north facet
    facets, cls = facets_from_orientation(roof, orient, min_px=50)
    assert len(cls) == 2


def test_scoring_counts_matches_within_tolerance():
    truth = np.zeros((20, 20), dtype=np.uint8)
    truth[10, 2:18] = EDGE_CLASSES["ridge"]
    pred = np.zeros_like(truth)
    pred[12, 2:18] = EDGE_CLASSES["ridge"]  # 2 px off: within tolerance
    pred[2, 2:6] = EDGE_CLASSES["ridge"]  # far off: a false positive
    s = score_edges(pred, truth, np.zeros(truth.shape, bool), tol_px=3)
    tp_p, n_p, tp_t, n_t = s["ridge"]
    assert (tp_p, n_p) == (16, 20) and (tp_t, n_t) == (16, 16)
