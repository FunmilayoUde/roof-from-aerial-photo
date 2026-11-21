"""Measure how far the LiDAR roof edges sit from the roof edges visible in an aerial photo.

Each LiDAR edge is cut into short pieces. Each piece is slid along its own normal across the
photo's edge-strength image, and the offset where it lines up best is recorded. A piece only
constrains the shift *across* it (sliding along an edge changes nothing), so each piece gives
one equation  n . shift = d.

Two explanations are then compared:
  - coordinate error: one constant shift s0 for the whole house;
  - relief displacement: roofs lean away from the camera's nadir by an amount proportional to
    height above the ground the orthophoto was corrected to, so shift = s0 + h * v, where h is
    height above ground and v the displacement per metre of height.
If relief displacement is the cause, v is clearly non-zero and s0 is close to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine
from scipy.ndimage import gaussian_filter, sobel

from roofphoto.geo.raster import sample


def gradients(rgb: np.ndarray, sigma_px: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """World-oriented gradient (gx east, gy north) of the smoothed grey image; rgb is (3, H, W) or (H, W).

    Image rows run southward, so the northward gradient is minus the gradient along rows.
    """
    grey = rgb.astype(float).mean(axis=0) if rgb.ndim == 3 else rgb.astype(float)
    g = gaussian_filter(grey, sigma_px)
    return sobel(g, axis=1), -sobel(g, axis=0)


def edge_strength(rgb: np.ndarray, sigma_px: float = 1.5) -> np.ndarray:
    """Gradient magnitude, for display."""
    return np.hypot(*gradients(rgb, sigma_px))


@dataclass
class Piece:
    edge_id: int
    edge_type: str
    xy: np.ndarray  # (K, 2) sample points along the piece, world
    normal: np.ndarray  # (2,) unit normal in plan
    z: float  # mean NAP height of the piece


def edge_pieces(edges: list, piece_len_m: float = 1.0, spacing_m: float = 0.04, trim_m: float = 0.2) -> list[Piece]:
    """Cut each edge (with .id, .type, .p0, .p1 in world xyz) into pieces of about piece_len_m.

    trim_m is left off both ends of each edge, where other edges meet it at a corner and the
    photo's edge strength comes from more than one line.
    """
    pieces = []
    for e in edges:
        p0, p1 = np.asarray(e.p0, float), np.asarray(e.p1, float)
        plan = p1[:2] - p0[:2]
        length = float(np.hypot(*plan))
        usable = length - 2 * trim_m
        if usable < spacing_m * 5:
            continue
        direction = plan / length
        normal = np.array([-direction[1], direction[0]])
        n = max(1, round(usable / piece_len_m))
        bounds = np.linspace(trim_m / length, 1 - trim_m / length, n + 1)
        for a, b in zip(bounds[:-1], bounds[1:]):
            k = max(5, round((b - a) * length / spacing_m))
            t = np.linspace(a, b, k)
            pts = p0 + t[:, None] * (p1 - p0)
            pieces.append(Piece(e.id, e.type, pts[:, :2], normal, float(pts[:, 2].mean())))
    return pieces


@dataclass
class PieceOffset:
    piece: Piece
    offset_m: float  # photo edge is at piece + offset * normal
    peak: float  # mean edge strength at the best offset
    contrast: float  # peak / median over the search range
    at_limit: bool  # best offset at the edge of the search range: unreliable


def search_offsets(
    grad: tuple[np.ndarray, np.ndarray],
    transform: Affine,
    pieces: list[Piece],
    search_m: float = 1.2,
    step_m: float = 0.02,
    centres_m: np.ndarray | None = None,
) -> list[PieceOffset]:
    """For each piece, the offset along its normal where the photo has the strongest edge parallel to it.

    The score is the image gradient *across* the piece, |grad . n|, averaged along the piece. A roof
    edge has its gradient across the edge; tree texture and lines at other angles mostly do not.
    centres_m, if given, centres each piece's search window (for a second, narrower pass).
    """
    gx, gy = grad
    offsets = np.arange(-search_m, search_m + step_m / 2, step_m)
    out = []
    for j, p in enumerate(pieces):
        trial = offsets + (0.0 if centres_m is None else centres_m[j])
        shifted = p.xy[None, :, :] + trial[:, None, None] * p.normal  # (S, K, 2)
        across = (
            sample(gx, transform, shifted[..., 0], shifted[..., 1]) * p.normal[0]
            + sample(gy, transform, shifted[..., 0], shifted[..., 1]) * p.normal[1]
        )
        profile = np.nanmean(np.abs(across), axis=1)
        if np.all(np.isnan(profile)):
            continue
        i = int(np.nanargmax(profile))
        best = trial[i]
        if 0 < i < len(trial) - 1:  # parabolic refinement between grid steps
            y0, y1, y2 = profile[i - 1 : i + 2]
            denom = y0 - 2 * y1 + y2
            if denom < 0:
                best += 0.5 * (y0 - y2) / denom * step_m
        out.append(
            PieceOffset(
                p,
                float(best),
                float(profile[i]),
                float(profile[i] / max(np.nanmedian(profile), 1e-9)),
                i in (0, len(trial) - 1),
            )
        )
    return out


@dataclass
class ShiftFit:
    s0: np.ndarray  # (2,) constant shift, m (x east, y north)
    v: np.ndarray | None  # (2,) shift per metre of height above ground; None for the constant model
    rms_m: float  # residual RMS over inliers
    inliers: np.ndarray  # bool mask over the input rows
    n: int

    def shift_at(self, h: float) -> np.ndarray:
        return self.s0 + (0 if self.v is None else h * self.v)


def fit_shift(
    normals: np.ndarray,
    d: np.ndarray,
    h: np.ndarray | None = None,
    through_ground: bool = False,
    iterations: int = 5,
    min_cut_m: float = 0.1,
) -> ShiftFit:
    """Least squares for n . (s0 + h v) = d, refit without outliers.

    h None: constant shift only (n . s0 = d). through_ground: s0 fixed at zero (n . h v = d), i.e.
    pure relief displacement, for when ground-level features are known to be in place.

    Outliers are pieces that locked onto the wrong line in the photo (a gutter, a shadow, a
    neighbour's roof). They are dropped at 3 robust standard deviations, but never closer than
    min_cut_m, so a very consistent fit doesn't start discarding good pieces.
    """
    normals = np.asarray(normals, float)
    d = np.asarray(d, float)
    if h is None:
        A = normals
    elif through_ground:
        A = normals * np.asarray(h, float)[:, None]
    else:
        A = np.column_stack([normals, normals * np.asarray(h, float)[:, None]])
    keep = np.ones(len(d), dtype=bool)
    for _ in range(iterations):
        params, *_ = np.linalg.lstsq(A[keep], d[keep], rcond=None)
        r = d - A @ params
        mad = np.median(np.abs(r[keep] - np.median(r[keep])))
        new_keep = np.abs(r) <= max(3 * 1.4826 * mad, min_cut_m)
        if new_keep.sum() < A.shape[1] + 1 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    params, *_ = np.linalg.lstsq(A[keep], d[keep], rcond=None)
    r = d - A @ params
    if h is not None and through_ground:
        params = np.concatenate([np.zeros(2), params])
    return ShiftFit(
        s0=params[:2],
        v=None if h is None else params[2:],
        rms_m=float(np.sqrt(np.mean(r[keep] ** 2))),
        inliers=keep,
        n=int(keep.sum()),
    )


def bootstrap(normals, d, h, n_boot: int = 500, seed: int = 0) -> np.ndarray:
    """Parameter draws from refitting on resampled pieces: (n_boot, 2 or 4)."""
    rng = np.random.default_rng(seed)
    normals, d = np.asarray(normals), np.asarray(d)
    h = None if h is None else np.asarray(h)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        f = fit_shift(normals[i], d[i], None if h is None else h[i])
        draws.append(f.s0 if f.v is None else np.concatenate([f.s0, f.v]))
    return np.array(draws)
