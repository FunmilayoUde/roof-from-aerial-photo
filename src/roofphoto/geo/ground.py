"""Ground-level alignment check: LiDAR intensity of ground points against the aerial photo.

An orthophoto is corrected against a ground model, so features *on the ground* (road paint,
kerbs, paving edges) should sit where they belong even when roofs are displaced. LiDAR records
how strongly each pulse came back (intensity), which shows the same features. Matching an
intensity image of ground points to the photo measures the photo's offset at ground level:
about zero if roof offsets are relief displacement, about the roof offset if they are a
coordinate error.

Paint is bright in both; grass is bright in the near-infrared laser but dark in the photo, so
raw values can correlate either way. Edges are matched instead: gradient magnitude in both
images, compared by normalised cross-correlation (NCC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine
from scipy.ndimage import binary_erosion, gaussian_filter

from roofphoto.geo.align import gradients
from roofphoto.geo.raster import Bounds, grid_for_bounds, pixel_to_world, sample


def rasterize_mean(
    xy: np.ndarray, values: np.ndarray, bounds: Bounds, res: float
) -> tuple[np.ndarray, np.ndarray, Affine]:
    """Mean of values per cell (NaN where empty), the per-cell count, and the grid transform."""
    t, w, h = grid_for_bounds(bounds, res)
    col = np.floor((xy[:, 0] - t.c) / res).astype(int)
    row = np.floor((t.f - xy[:, 1]) / res).astype(int)
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    idx = row[ok] * w + col[ok]
    total = np.bincount(idx, weights=values[ok], minlength=w * h).reshape(h, w)
    count = np.bincount(idx, minlength=w * h).reshape(h, w)
    with np.errstate(invalid="ignore"):
        mean = total / count
    return mean, count, t


def filled_edges(img: np.ndarray, sigma_px: float = 1.0, min_support: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Gradient magnitude of an image with holes, and where it can be trusted.

    Holes (cells with no ground points: under buildings, trees, cars) are filled by normalised
    convolution, which averages only the cells that have data. Gradients next to a hole are
    still unreliable, so the valid mask is eroded by two cells.
    """
    have = np.isfinite(img)
    num = gaussian_filter(np.where(have, img, 0.0), sigma_px)
    den = gaussian_filter(have.astype(float), sigma_px)
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = np.where(den > 1e-6, num / den, 0.0)
    gx, gy = gradients(smooth, sigma_px=0)
    valid = binary_erosion(den >= min_support, iterations=2)
    return np.hypot(gx, gy), valid


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 50:
        return np.nan
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else np.nan


@dataclass
class ShiftSearch:
    shift: np.ndarray  # (2,) photo minus LiDAR, m
    peak: float  # NCC at the best shift
    at_zero: float  # NCC at no shift
    offsets: np.ndarray  # (S,) grid of trial offsets per axis (coarse pass)
    surface: np.ndarray  # (S, S) NCC over (dy, dx) on the coarse grid
    cells: int


def search_shift(
    ref: np.ndarray,
    ref_valid: np.ndarray,
    ref_t: Affine,
    photo_edges: np.ndarray,
    photo_t: Affine,
    search_m: float = 1.6,
    coarse_m: float = 0.08,
    fine_m: float = 0.02,
) -> ShiftSearch:
    """The shift (dx, dy) that maximises NCC between ref at (x, y) and the photo at (x + dx, y + dy)."""
    rows, cols = np.nonzero(ref_valid)
    x, y = pixel_to_world(ref_t, cols + 0.5, rows + 0.5)
    r = ref[rows, cols]

    def score(dx: float, dy: float) -> float:
        return ncc(r, sample(photo_edges, photo_t, x + dx, y + dy))

    offsets = np.arange(-search_m, search_m + coarse_m / 2, coarse_m)
    surface = np.array([[score(dx, dy) for dx in offsets] for dy in offsets])
    iy, ix = np.unravel_index(np.nanargmax(surface), surface.shape)
    best = np.array([offsets[ix], offsets[iy]])
    fine = np.arange(-coarse_m, coarse_m + fine_m / 2, fine_m)
    local = np.array([[score(best[0] + dx, best[1] + dy) for dx in fine] for dy in fine])
    fy, fx = np.unravel_index(np.nanargmax(local), local.shape)
    best = best + np.array([fine[fx], fine[fy]])
    return ShiftSearch(best, float(np.nanmax(local)), score(0.0, 0.0), offsets, surface, int(len(r)))


def compare_shifts(
    ref: np.ndarray,
    ref_valid: np.ndarray,
    ref_t: Affine,
    photo_edges: np.ndarray,
    photo_t: Affine,
    shift_a: np.ndarray,
    shift_b: np.ndarray,
    block_m: float = 5.0,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Which of two candidate shifts matches the ground better, and how sure is that?

    Block bootstrap: the ground is cut into block_m squares, and blocks (not cells) are resampled,
    because neighbouring cells share the same kerb or marking and are not independent evidence.
    Returns each shift's NCC and the fraction of resamples in which shift_a beats shift_b.
    """
    rows, cols = np.nonzero(ref_valid)
    x, y = pixel_to_world(ref_t, cols + 0.5, rows + 0.5)
    r = ref[rows, cols]
    pa = sample(photo_edges, photo_t, x + shift_a[0], y + shift_a[1])
    pb = sample(photo_edges, photo_t, x + shift_b[0], y + shift_b[1])
    ok = np.isfinite(pa) & np.isfinite(pb)
    x, y, r, pa, pb = x[ok], y[ok], r[ok], pa[ok], pb[ok]
    block = np.unique(np.column_stack([np.floor(x / block_m), np.floor(y / block_m)]), axis=0, return_inverse=True)[1]
    block = block.ravel()
    members = [np.nonzero(block == b)[0] for b in range(block.max() + 1)]
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n_boot):
        idx = np.concatenate([members[b] for b in rng.integers(0, len(members), len(members))])
        wins += ncc(r[idx], pa[idx]) > ncc(r[idx], pb[idx])
    return {
        "ncc_a": round(ncc(r, pa), 3),
        "ncc_b": round(ncc(r, pb), 3),
        "a_beats_b_fraction": round(wins / n_boot, 3),
        "blocks": len(members),
    }
