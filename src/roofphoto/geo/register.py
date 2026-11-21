"""Register one house's LiDAR roof to its aerial photo: the Stage 1 method as a reusable function.

Two-pass edge-piece search (see align.py), then three fits:
  constant      shift = s0                 one offset for the whole roof
  pure relief   shift = h·v                relief displacement, zero at the ground (Stage 1 ground check)
  free relief   shift = s0 + h·v           kept for diagnosis; unstable on rectangular houses
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine

from roofphoto.geo.align import ShiftFit, bootstrap, edge_pieces, fit_shift, gradients, search_offsets
from roofphoto.io.lidar import HouseLabels

AXIS_COS = 0.9  # a piece "constrains" an axis when its normal is within ~25° of it


@dataclass
class Registration:
    normals: np.ndarray  # (N, 2) per piece
    d: np.ndarray  # (N,) measured offset along the normal, m
    h: np.ndarray  # (N,) height above ground, m
    types: np.ndarray  # (N,) edge type per piece
    first_pass: int
    const: ShiftFit
    pure: ShiftFit
    free: ShiftFit
    along_dir: np.ndarray  # house long axis (unit, east-ish)
    across_dir: np.ndarray
    free_ci: tuple[np.ndarray, np.ndarray] | None  # 5th/95th percentile of free-fit params

    def residual(self, fit: ShiftFit) -> np.ndarray:
        shift = np.broadcast_to(fit.s0 + (0 if fit.v is None else self.h[:, None] * fit.v), self.normals.shape)
        return self.d - np.sum(self.normals * shift, axis=1)

    @property
    def common(self) -> np.ndarray:
        """Pieces both the constant and the pure-relief fit keep: the fair set for comparing them."""
        return self.const.inliers & self.pure.inliers

    def axis_support(self, inliers: np.ndarray) -> dict[str, int]:
        return {
            name: int(((np.abs(self.normals @ vec) > AXIS_COS) & inliers).sum())
            for name, vec in (("along", self.along_dir), ("across", self.across_dir))
        }

    def summary(self) -> dict:
        def rms(r: np.ndarray, m: np.ndarray) -> float | None:
            return round(float(np.sqrt(np.mean(r[m] ** 2))), 3) if m.any() else None

        def per_axis(vec: np.ndarray) -> dict:
            return {
                "along_m": round(float(vec @ self.along_dir), 3),
                "across_m": round(float(vec @ self.across_dir), 3),
            }

        r_const, r_pure, r_free = self.residual(self.const), self.residual(self.pure), self.residual(self.free)
        common = self.common
        by_type = {}
        for t in sorted(set(self.types)):
            m = (self.types == t) & common
            if m.any():
                by_type[t] = {
                    "pieces": int(m.sum()),
                    "median_height_m": round(float(np.median(self.h[m])), 2),
                    "rms_residual_constant_m": rms(r_const, m),
                    "rms_residual_pure_relief_m": rms(r_pure, m),
                    "rms_residual_free_relief_m": rms(r_free, m),
                }
        out = {
            "pieces": {
                "first_pass": self.first_pass,
                "searched": len(self.d),
                "inliers_constant": self.const.n,
                "inliers_pure_relief": self.pure.n,
                "inliers_free_relief": self.free.n,
            },
            "constant_model": {
                "shift_m": self.const.s0.round(3).tolist(),
                "magnitude_m": round(float(np.hypot(*self.const.s0)), 3),
                "rms_m": round(self.const.rms_m, 3),
            },
            "pure_relief_model": {
                "v_m_per_m": self.pure.v.round(4).tolist(),
                "v_size_m_per_m": round(float(np.hypot(*self.pure.v)), 4),
                "v_bearing_deg": round(float((90 - np.degrees(np.arctan2(self.pure.v[1], self.pure.v[0]))) % 360), 1),
                "rms_m": round(self.pure.rms_m, 3),
            },
            "model_comparison_same_pieces": {
                "pieces": int(common.sum()),
                "rms_constant_m": rms(r_const, common),
                "rms_pure_relief_m": rms(r_pure, common),
                "rms_free_relief_m": rms(r_free, common),
            },
            "free_relief_model": {
                "s0_m": self.free.s0.round(3).tolist(),
                "v_m_per_m": self.free.v.round(4).tolist(),
                "rms_m": round(self.free.rms_m, 3),
            },
            "house_axes": {
                "along_bearing_deg": round(
                    float((90 - np.degrees(np.arctan2(self.along_dir[1], self.along_dir[0]))) % 360), 1
                ),
                "pieces_constraining_constant": self.axis_support(self.const.inliers),
                "pieces_constraining_pure_relief": self.axis_support(self.pure.inliers),
                "constant_shift": per_axis(self.const.s0),
            },
            "by_edge_type": by_type,
            "heights_above_ground_m": {
                "p10": round(float(np.percentile(self.h, 10)), 2),
                "p90": round(float(np.percentile(self.h, 90)), 2),
            },
        }
        if self.free_ci is not None:
            lo, hi = self.free_ci
            out["free_relief_model"]["s0_90ci_m"] = [lo[:2].round(3).tolist(), hi[:2].round(3).tolist()]
            out["free_relief_model"]["v_90ci"] = [lo[2:].round(4).tolist(), hi[2:].round(4).tolist()]
        return out


def house_axes(house: HouseLabels) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors along and across the BAG outline's minimum rotated rectangle."""
    rect = np.asarray(house.footprint.minimum_rotated_rectangle.exterior.coords)[:3]
    sides = np.diff(rect, axis=0)
    along = sides[np.argmax(np.hypot(sides[:, 0], sides[:, 1]))]
    along = along / np.linalg.norm(along)
    if along[0] < 0:
        along = -along
    return along, np.array([-along[1], along[0]])


def register_house(
    house: HouseLabels,
    rgb: np.ndarray,
    transform: Affine,
    ground_z: float,
    search_m: float = 1.2,
    narrow_m: float = 0.4,
    with_bootstrap: bool = False,
) -> Registration | None:
    """None if too few edge pieces survive to fit the free model (4 parameters)."""
    # 'unclassified' edges come from the pipeline's traced fallback outlines, which it marks
    # approximate; they would measure the pipeline's error, not the photo's offset.
    pieces = edge_pieces([e for e in house.edges if e.type != "unclassified"])
    if len(pieces) < 6:
        return None
    grad = gradients(rgb)
    # Pass 1: wide search, so a large offset can be found at all.
    first = [r for r in search_offsets(grad, transform, pieces, search_m=search_m) if not r.at_limit]
    if len(first) < 6:
        return None
    n1 = np.array([r.piece.normal for r in first])
    h1 = np.array([r.piece.z - ground_z for r in first])
    fit1 = fit_shift(n1, np.array([r.offset_m for r in first]), h1)
    # Pass 2: narrow search around where pass 1 predicts each edge, which keeps pieces from
    # locking onto a stronger line (gutter, chimney, trim) further away.
    h_all = np.array([p.z - ground_z for p in pieces])
    centres = np.einsum("ij,ij->i", np.array([p.normal for p in pieces]), fit1.s0 + h_all[:, None] * fit1.v)
    found = [r for r in search_offsets(grad, transform, pieces, search_m=narrow_m, centres_m=centres) if not r.at_limit]
    if len(found) < 6:
        return None
    normals = np.array([r.piece.normal for r in found])
    d = np.array([r.offset_m for r in found])
    h = np.array([r.piece.z - ground_z for r in found])
    ci = None
    if with_bootstrap:
        draws = bootstrap(normals, d, h)
        ci = tuple(np.percentile(draws, [5, 95], axis=0))
    along, across = house_axes(house)
    return Registration(
        normals=normals,
        d=d,
        h=h,
        types=np.array([r.piece.edge_type for r in found]),
        first_pass=len(first),
        const=fit_shift(normals, d),
        pure=fit_shift(normals, d, h, through_ground=True),
        free=fit_shift(normals, d, h),
        along_dir=along,
        across_dir=across,
        free_ci=ci,
    )
