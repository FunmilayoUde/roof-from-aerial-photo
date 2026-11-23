"""Stage 5: straighten predicted roof geometry before typing edges.

Segmentation gives rounded, wobbly outlines; roofs are made of straight lines at a few angles to
the house's main axis. Each region (the whole roof, and each facet) is turned into a polygon,
simplified, and its sides snapped to the house's main axis or a multiple of 45° from it (hips run
diagonally in plan) when they are within `tol_deg`. Corners are recomputed where consecutive
sides meet. Facets are then redrawn, larger first, and gaps inside the straightened roof outline
go to the nearest facet, so straightening doesn't open false gaps between facets.
All in pixel coordinates (row, col).
"""

from __future__ import annotations

import numpy as np
from rasterio.features import rasterize, shapes
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import Polygon, shape
from shapely.validation import make_valid


def axis_angle_px(footprint, transform) -> float:
    """House main axis as an angle in the image (radians, in col/row space), from its outline."""
    rect = np.asarray(footprint.minimum_rotated_rectangle.exterior.coords)[:3]
    sides = np.diff(rect, axis=0)
    dx, dy = sides[np.argmax(np.hypot(sides[:, 0], sides[:, 1]))]
    # World x east, y north; image col east, row south: (col, row) = (dx / a, -dy / a).
    return float(np.arctan2(-dy, dx))


def _snap(angle: float, axis: float, tol: float) -> float | None:
    """Nearest of axis + k·45° if within tol (all in radians, directions mod 180°)."""
    best, d_best = None, tol
    for k in range(4):
        target = axis + k * np.pi / 4
        d = abs((angle - target + np.pi / 2) % np.pi - np.pi / 2)
        if d <= d_best:
            best, d_best = target, d
    return best


def _intersect(p1, d1, p2, d2):
    """Intersection of lines p1 + t d1 and p2 + s d2, or None if nearly parallel."""
    m = np.array([d1, -d2]).T
    if abs(np.linalg.det(m)) < 0.1:
        return None
    t, _ = np.linalg.solve(m, p2 - p1)
    return p1 + t * d1


def _drop_short_sides(pts: np.ndarray, min_len: float) -> np.ndarray:
    """Merge the shortest side into a point, repeatedly, until every side is at least min_len.

    Segmentation outlines zigzag; after simplification the zigzags remain as short sides whose
    directions are noise. Removing them leaves the long sides that carry the true directions.
    """
    pts = pts.copy()
    while len(pts) > 3:
        nxt = np.roll(pts, -1, axis=0)
        lengths = np.hypot(*(nxt - pts).T)
        i = int(np.argmin(lengths))
        if lengths[i] >= min_len:
            break
        j = (i + 1) % len(pts)
        mid = (pts[i] + pts[j]) / 2
        pts = np.delete(pts, j, axis=0)
        pts[i if i < j else i - 1] = mid
    return pts


def regularise_polygon(
    poly: Polygon, axis: float, simplify_px: float = 2.5, tol_deg: float = 15.0, min_side_px: float = 6.0
) -> Polygon:
    """Simplified polygon with sides snapped to the house axes; falls back to the simplified one."""
    simple = poly.simplify(simplify_px, preserve_topology=True)
    if simple.is_empty or simple.geom_type != "Polygon":
        return poly
    pts = _drop_short_sides(np.asarray(simple.exterior.coords)[:-1], min_side_px)
    if len(pts) < 3:
        return simple
    tol = np.radians(tol_deg)
    # Sides that match none of the house's directions and are short are wiggles: merge them away.
    while len(pts) > 3:
        nxt = np.roll(pts, -1, axis=0)
        v = nxt - pts
        lengths = np.hypot(*v.T)
        free = [i for i in range(len(pts)) if _snap(float(np.arctan2(v[i, 1], v[i, 0])), axis, tol) is None]
        short = [i for i in free if lengths[i] < 2.5 * min_side_px]
        if not short:
            break
        i = min(short, key=lambda k: lengths[k])
        j = (i + 1) % len(pts)
        mid = (pts[i] + pts[j]) / 2
        pts = np.delete(pts, j, axis=0)
        pts[i if i < j else i - 1] = mid
    # One line per side: through its midpoint, along its own or its snapped direction.
    lines = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        v = b - a
        length = float(np.hypot(*v))
        if length < 1e-6:
            continue
        ang = float(np.arctan2(v[1], v[0]))
        snapped = _snap(ang, axis, tol)
        if snapped is not None:
            direction = np.array([np.cos(snapped), np.sin(snapped)])
            if direction @ v < 0:
                direction = -direction
        else:
            direction = v / length
        lines.append(((a + b) / 2, direction, length))
    # Merge consecutive sides that ended up parallel (one straight side split in two).
    merged = []
    for mid, d, ln in lines:
        if merged and abs(merged[-1][1][0] * d[1] - merged[-1][1][1] * d[0]) < 1e-3:
            pm, dm, lm = merged[-1]
            merged[-1] = ((pm * lm + mid * ln) / (lm + ln), dm, lm + ln)
        else:
            merged.append((mid, d, ln))

    # A short side between two parallel sides is a jog the true outline doesn't have: drop it and
    # merge its neighbours into one straight side.
    def parallel(d1, d2) -> bool:
        return abs(d1[0] * d2[1] - d1[1] * d2[0]) < 1e-3

    changed = True
    while changed and len(merged) > 4:
        changed = False
        for i, (_, _, ln) in enumerate(merged):
            prev, nxt = merged[i - 1], merged[(i + 1) % len(merged)]
            if ln < 2.5 * min_side_px and parallel(prev[1], nxt[1]):
                total = prev[2] + nxt[2]
                joined = ((prev[0] * prev[2] + nxt[0] * nxt[2]) / total, prev[1], total)
                keep = [
                    m
                    for k, m in enumerate(merged)
                    if k not in (i - 1 if i else len(merged) - 1, i, (i + 1) % len(merged))
                ]
                insert_at = min(i - 1 if i else len(merged) - 1, (i + 1) % len(merged))
                merged = keep[:insert_at] + [joined] + keep[insert_at:]
                changed = True
                break
    if len(merged) < 3:
        return simple

    def corners_of(lines):
        out = []
        for i, (p1, d1, _) in enumerate(lines):
            p2, d2, _ = lines[(i + 1) % len(lines)]
            x = _intersect(p1, d1, p2, d2)
            out.append(x if x is not None else (p1 + p2) / 2)
        return out

    # The same jog test on the final sides: a side can be long in the input yet end up short
    # between recomputed corners.
    corners = corners_of(merged)
    while len(merged) > 4:
        n = len(merged)
        side = [float(np.hypot(*(corners[i] - corners[i - 1]))) for i in range(n)]  # side i ends at corner i
        jog = next(
            (i for i in range(n) if side[i] < 2.5 * min_side_px and parallel(merged[i - 1][1], merged[(i + 1) % n][1])),
            None,
        )
        if jog is None:
            break
        prev, nxt = merged[jog - 1], merged[(jog + 1) % n]
        total = prev[2] + nxt[2]
        joined = ((prev[0] * prev[2] + nxt[0] * nxt[2]) / total, prev[1], total)
        drop = {(jog - 1) % n, jog, (jog + 1) % n}
        rest = [m for k, m in enumerate(merged) if k not in drop]
        merged = rest[: min(drop)] + [joined] + rest[min(drop) :] if min(drop) <= len(rest) else rest + [joined]
        corners = corners_of(merged)
    out = make_valid(Polygon(corners))
    if out.geom_type != "Polygon":
        out = max(
            (g for g in getattr(out, "geoms", []) if g.geom_type == "Polygon"), key=lambda g: g.area, default=None
        )
    if out is None or out.is_empty or abs(out.area - poly.area) > 0.3 * poly.area:
        return simple
    return out


def _polygons(mask: np.ndarray, smooth_px: float = 1.5) -> list[Polygon]:
    """Polygons of a boolean mask in (col, row) pixel-corner coordinates (holes dropped), after a
    light blur-and-threshold that removes pixel-level jaggedness."""
    if smooth_px:
        mask = gaussian_filter(mask.astype(float), smooth_px) > 0.5
    return [Polygon(shape(g).exterior) for g, v in shapes(mask.astype(np.uint8), mask=mask) if v == 1]


def regularise_facets(facets: np.ndarray, axis: float, simplify_px: float = 2.5, tol_deg: float = 15.0) -> np.ndarray:
    """Facet label map with straightened facets inside a straightened roof outline."""
    roof = facets > 0
    out_shape = facets.shape
    roof_reg = np.zeros(out_shape, dtype=bool)
    for p in _polygons(roof):
        q = regularise_polygon(p, axis, simplify_px, tol_deg)
        roof_reg |= rasterize([(q, 1)], out_shape=out_shape, fill=0, dtype="uint8") == 1
    ids = [i for i in np.unique(facets) if i > 0]
    ids.sort(key=lambda i: -(facets == i).sum())  # larger first, so small facets end up on top
    reg = np.zeros(out_shape, dtype=facets.dtype)
    for i in ids:
        for p in _polygons(facets == i):
            q = regularise_polygon(p, axis, simplify_px, tol_deg)
            reg[rasterize([(q, 1)], out_shape=out_shape, fill=0, dtype="uint8") == 1] = i
    # Inside the straightened roof, fill gaps from the nearest facet; outside it, nothing.
    if (reg > 0).any():
        _, (rr, cc) = distance_transform_edt(reg == 0, return_indices=True)
        reg = np.where(roof_reg, reg[rr, cc], 0)
    return reg
