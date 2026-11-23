"""Stage 5: facets and edge types from a roof mask and per-pixel orientation, by geometry.

Facets are connected regions of one orientation class (flat, or sloped towards one of 8 compass
directions); fragments under `min_px` pixels are merged into the neighbour they share most border
with. Each boundary is then typed from which way the facets on either side slope:

  two sloped facets   both slope away from the line (it is at both tops):
                          facing apart by 135° or more  -> ridge
                          otherwise                     -> hip
                      both slope towards it          -> valley
                      one towards, one away          -> break (a step or pitch change)
  sloped and flat     break
  facet and outside   flat facet                     -> perimeter
                      sloped facet running down to the outline (downhill within 45° of
                      outward)                          -> eave
                      otherwise (running along it)    -> rake

Directions: image rows run south and columns east. Class 2 + k faces 45k° clockwise from north,
so its downhill vector in (row, col) is (-cos 45k°, sin 45k°).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label

from roofphoto.data.chips import EDGE_CLASSES

FLAT = 1
SHIFTS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-neighbours, (row, col)


def downhill(cls: int) -> np.ndarray | None:
    """Unit downhill vector (row, col) of an orientation class; None for flat."""
    if cls == FLAT:
        return None
    az = np.radians(45 * (cls - 2))
    return np.array([-np.cos(az), np.sin(az)])


def facets_from_orientation(roof: np.ndarray, orient: np.ndarray, min_px: int = 150) -> tuple[np.ndarray, dict]:
    """(facet labels (0 = not roof), {facet id: orientation class}).

    orient holds classes 1-9 on roof pixels. Small fragments are merged into their main neighbour,
    repeatedly, so speckle in the prediction doesn't become a swarm of tiny facets.
    """
    cls = np.where(roof, orient, 0).astype(int)
    for _ in range(10):
        lab, n = _components(cls)
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        small = [i for i in range(1, n + 1) if sizes[i] < min_px]
        if not small:
            break
        changed = False
        for i in small:
            m = lab == i
            neighbour = _neighbour_values(m, cls, exclude=cls[m][0])
            if neighbour is not None:
                cls[m] = neighbour
                changed = True
        if not changed:
            break
    lab, n = _components(cls)
    return lab, {i: int(cls[lab == i][0]) for i in range(1, n + 1)}


def _components(cls: np.ndarray) -> tuple[np.ndarray, int]:
    lab = np.zeros(cls.shape, dtype=int)
    n = 0
    for c in np.unique(cls):
        if c == 0:
            continue
        comp, k = label(cls == c)
        lab[comp > 0] = comp[comp > 0] + n
        n += k
    return lab, n


def _neighbour_values(mask: np.ndarray, cls: np.ndarray, exclude: int) -> int | None:
    """Most common roof class bordering mask (4-neighbours), other than its own."""
    vals = []
    for dr, dc in SHIFTS:
        shifted = np.roll(mask, (dr, dc), axis=(0, 1)) & ~mask
        v = cls[shifted]
        vals.append(v[(v != 0) & (v != exclude)])
    v = np.concatenate(vals)
    return int(np.bincount(v).argmax()) if len(v) else None


def edge_raster(facets: np.ndarray, facet_class: dict, eave_cos: float = np.cos(np.radians(45))) -> np.ndarray:
    """Edge types (EDGE_CLASSES values, 0 = none) on the boundary pixels of each facet."""
    h, w = facets.shape
    # For every boundary: which pixels, and the mean step from this facet to the other side.
    steps: dict[tuple[int, int], np.ndarray] = {}
    pixels: dict[tuple[int, int], list[np.ndarray]] = {}
    for dr, dc in SHIFTS:
        other = np.zeros_like(facets)
        rs = slice(max(dr, 0), h + min(dr, 0))
        rd = slice(max(-dr, 0), h + min(-dr, 0))
        cs = slice(max(dc, 0), w + min(dc, 0))
        cd = slice(max(-dc, 0), w + min(-dc, 0))
        other[rd, cd] = facets[rs, cs]  # other[p] = facets[p + (dr, dc)]
        boundary = (facets > 0) & (other != facets)
        for a, b in set(zip(facets[boundary].tolist(), other[boundary].tolist())):
            m = boundary & (facets == a) & (other == b)
            steps[(a, b)] = steps.get((a, b), np.zeros(2)) + m.sum() * np.array([dr, dc], float)
            pixels.setdefault((a, b), []).append(np.argwhere(m))

    out = np.zeros(facets.shape, dtype=np.uint8)
    for (a, b), s in steps.items():
        p = np.concatenate(pixels[(a, b)])
        if b != 0:
            # Between facets: the mean step over the whole shared boundary, which is usually one
            # straight line; averaging also turns a pixel staircase into its true diagonal.
            n = s / max(np.linalg.norm(s), 1e-9)  # unit direction from facet a towards b
            out[p[:, 0], p[:, 1]] = EDGE_CLASSES[_edge_type(facet_class[a], facet_class[b], n, eave_cos)]
            continue
        # Against the outside: a facet's outline turns corners, so use the local outward direction,
        # from the slope of the blurred facet mask (follows edges at any angle, not just the grid).
        g_r, g_c = np.gradient(gaussian_filter((facets == a).astype(float), 2.0))
        for r, c in p:
            n = -np.array([g_r[r, c], g_c[r, c]])
            n = n / max(np.linalg.norm(n), 1e-9)
            out[r, c] = EDGE_CLASSES[_edge_type(facet_class[a], None, n, eave_cos)]
    return out


def _edge_type(ca: int, cb: int | None, n: np.ndarray, eave_cos: float) -> str:
    da = downhill(ca)
    if cb is None:  # roof against outside; n points outward
        if da is None:
            return "perimeter"
        return "eave" if da @ n >= eave_cos else "rake"
    db = downhill(cb)
    if da is None or db is None:
        return "break"
    a_away, b_away = da @ n < 0, db @ (-n) < 0  # sloping away from the line: it is at their top
    if a_away and b_away:
        return "ridge" if da @ db <= np.cos(np.radians(135)) else "hip"
    if not a_away and not b_away:
        return "valley"
    return "break"


def score_edges(pred: np.ndarray, truth: np.ndarray, ignore: np.ndarray, tol_px: float = 3.0) -> dict:
    """Per edge type and for any type: (true positives, predicted, labelled) pixel counts.

    A predicted pixel counts as correct if a labelled pixel of the same type lies within tol_px;
    a labelled pixel counts as found if a predicted one of the same type does. Ignore pixels are
    left out of both. Precision = tp_pred / predicted, recall = tp_truth / labelled.
    """
    out = {}
    types = {"any": None} | {t: v for t, v in EDGE_CLASSES.items() if t != "unclassified"}
    for name, v in types.items():
        p = (pred > 0) if v is None else (pred == v)
        t = ((truth > 0) & (truth != 255)) if v is None else (truth == v)
        p &= ~ignore
        if not p.any() and not t.any():
            continue
        dist_to_t = distance_transform_edt(~t) if t.any() else np.full(t.shape, np.inf)
        dist_to_p = distance_transform_edt(~p) if p.any() else np.full(p.shape, np.inf)
        out[name] = [
            int((p & (dist_to_t <= tol_px)).sum()),
            int(p.sum()),
            int((t & (dist_to_p <= tol_px)).sum()),
            int(t.sum()),
        ]
    return out
