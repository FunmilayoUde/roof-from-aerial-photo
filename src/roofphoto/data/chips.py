"""Training chips: a fixed-size photo crop around one house, and its labels rasterised on the same grid.

Labels are drawn in *photo space*: every LiDAR vertex is moved by the house's fitted shift before
rasterising, so label pixels sit on the roof as the photo shows it. Two bands per label file:

    band 1  facets   0 = background, k = k-th facet (report order), 255 = ignore
    band 2  edges    0 = none, EDGE_CLASSES[t] = edge type t (3 px wide lines), 255 = ignore

Ignore covers other buildings: a chip around one house shows parts of its neighbours, which have
no labels. Calling them background would teach the model that roofs are background.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from rasterio.transform import from_origin
from roofrecon.measure import EDGE_TYPES
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from roofphoto.io.lidar import HouseLabels

IGNORE = 255
EDGE_CLASSES = {t: i + 1 for i, t in enumerate(EDGE_TYPES)}  # 0 is "no edge"
EDGE_WIDTH_PX = 3

# Moves world xyz (N, 3) to where it appears in the photo, as xy (N, 2).
ShiftFn = Callable[[np.ndarray], np.ndarray]


def chip_grid(centre_xy: tuple[float, float], size_m: float, res: float) -> tuple[Affine, int]:
    """A square chip of round(size_m / res) pixels, its corner on the res grid (the photo's pixel grid)."""
    n = round(size_m / res)
    x0 = np.floor((centre_xy[0] - n * res / 2) / res + 0.5) * res
    y1 = np.floor((centre_xy[1] + n * res / 2) / res + 0.5) * res
    return from_origin(float(x0), float(y1), res, res), n


def constant_shift(s0: np.ndarray) -> ShiftFn:
    return lambda p: p[:, :2] + s0


def relief_shift(v: np.ndarray, ground_z: float) -> ShiftFn:
    return lambda p: p[:, :2] + (p[:, 2:3] - ground_z) * v


def _burn(shapes: list[tuple[BaseGeometry, int]], transform: Affine, n: int, out: np.ndarray) -> None:
    """Burn shapes in order (later ones on top) into out, in place."""
    for geom, value in shapes:
        if geom.is_empty:
            continue
        mask = rasterize([(geom, 1)], out_shape=(n, n), transform=transform, fill=0, dtype="uint8")
        out[mask == 1] = value


def rasterize_labels(
    house: HouseLabels, shift: ShiftFn, transform: Affine, n: int, ignore: BaseGeometry | None = None
) -> np.ndarray:
    """(2, n, n) uint8: facet band and edge band, as described in the module docstring."""
    facets = np.zeros((n, n), dtype=np.uint8)
    # Higher facets last: where facets overlap in plan, the label shows the one seen from above.
    order = sorted(range(len(house.facets)), key=lambda i: house.facets[i].polygon_xyz[:, 2].mean())
    shapes = []
    for i in order:
        f = house.facets[i]
        poly = Polygon(shift(f.polygon_xyz), [shift(h) for h in f.holes_xyz]).buffer(0)
        shapes.append((poly, i + 1))
    _burn(shapes, transform, n, facets)

    edges = np.zeros((n, n), dtype=np.uint8)
    half_width = EDGE_WIDTH_PX * transform.a / 2
    lines = []
    for e in house.edges:
        xy = shift(np.vstack([e.p0, e.p1]))
        lines.append((LineString(xy).buffer(half_width, cap_style="flat"), EDGE_CLASSES[e.type]))
    _burn(lines, transform, n, edges)

    if ignore is not None and not ignore.is_empty:
        m = rasterize([(ignore, 1)], out_shape=(n, n), transform=transform, fill=0, dtype="uint8") == 1
        facets[m & (facets == 0)] = IGNORE
        edges[m & (edges == 0)] = IGNORE
    return np.stack([facets, edges])


def neighbour_ignore(neighbours: list[Polygon], buffer_m: float = 2.0) -> BaseGeometry:
    """Other buildings' outlines grown by overhang plus relief displacement (~0.75 m + ~1 m)."""
    from shapely.ops import unary_union

    return unary_union([p.buffer(buffer_m, join_style="mitre") for p in neighbours])
