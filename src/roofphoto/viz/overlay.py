"""Photo + LiDAR roof overlays, drawn in world coordinates so the axes read in metres (RD New)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from affine import Affine
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from roofrecon.export import EDGE_COLORS, FACET_COLORS

from roofphoto.io.lidar import HouseLabels

# Moves world xyz points (N, 3) to where they appear in the photo; identity if None.
ShiftFn = Callable[[np.ndarray], np.ndarray]


def raster_extent(transform: Affine, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """(left, right, bottom, top) for matplotlib's imshow, from a north-up transform."""
    h, w = shape
    left, top = transform.c, transform.f
    return left, left + w * transform.a, top + h * transform.e, top


def draw_overlay(
    ax: Axes,
    rgb: np.ndarray,
    transform: Affine,
    house: HouseLabels,
    shift: ShiftFn | None = None,
    title: str = "",
) -> None:
    """rgb: (3, H, W) uint8. Facets filled, edges coloured by type, BAG walls dashed."""
    move = shift or (lambda p: p[:, :2])
    ax.imshow(np.moveaxis(rgb, 0, -1), extent=raster_extent(transform, rgb.shape[1:]), interpolation="nearest")
    for i, f in enumerate(house.facets):
        color = FACET_COLORS[i % len(FACET_COLORS)]
        ax.add_patch(MplPolygon(move(f.polygon_xyz), closed=True, fc=color, ec="none", alpha=0.3))
    for e in house.edges:
        xy = move(np.vstack([e.p0, e.p1]))
        ax.plot(xy[:, 0], xy[:, 1], color=EDGE_COLORS.get(e.type, "#52514e"), lw=1.8, solid_capstyle="round")
    wx, wy = house.footprint.exterior.xy
    ax.plot(wx, wy, color="white", lw=1.0, ls=(0, (4, 3)))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(useOffset=False, style="plain")


def legend_handles(house: HouseLabels) -> list[Line2D]:
    used = {e.type for e in house.edges}
    handles = [Line2D([], [], color=EDGE_COLORS[t], lw=2, label=t) for t in EDGE_COLORS if t in used]
    handles += [Line2D([], [], color=EDGE_COLORS.get(t, "#52514e"), lw=2, label=t) for t in used - EDGE_COLORS.keys()]
    handles.append(Line2D([], [], color="white", lw=1, ls=(0, (4, 3)), label="BAG walls"))
    return handles
