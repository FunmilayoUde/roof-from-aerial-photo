"""Stage 3: non-ML roof-area baselines, from the BAG footprint alone.

A roof is bigger than the footprint it covers for two reasons: it slopes (true area = plan area /
cos(pitch)), and it overhangs the walls the footprint traces. A single factor absorbs both:

    B0  area = footprint                         flat, no overhang: the floor any method must beat
    B1  area = footprint x k                     k = median(true / footprint) on training houses
    B2  area = footprint x k[age band]           k per BAG construction-year band, same rule

Medians, not means: a few houses with wrong LiDAR truth shouldn't move the factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AGE_BANDS = [(0, 1930), (1930, 1970), (1970, 2000), (2000, 9999)]  # [start, end) construction year
MIN_PER_BAND = 10  # fewer training houses than this: fall back to the overall factor


def age_band(year: int) -> int:
    return next(i for i, (a, b) in enumerate(AGE_BANDS) if a <= year < b)


@dataclass
class Baseline:
    k_all: float
    k_band: dict[int, float]

    def b0(self, footprint: np.ndarray) -> np.ndarray:
        return np.asarray(footprint, float)

    def b1(self, footprint: np.ndarray) -> np.ndarray:
        return np.asarray(footprint, float) * self.k_all

    def b2(self, footprint: np.ndarray, built: np.ndarray) -> np.ndarray:
        k = np.array([self.k_band.get(age_band(int(y)), self.k_all) for y in built])
        return np.asarray(footprint, float) * k


def fit(footprint: np.ndarray, truth: np.ndarray, built: np.ndarray) -> Baseline:
    ratio = np.asarray(truth, float) / np.asarray(footprint, float)
    bands = np.array([age_band(int(y)) for y in built])
    k_band = {
        b: float(np.median(ratio[bands == b])) for b in range(len(AGE_BANDS)) if (bands == b).sum() >= MIN_PER_BAND
    }
    return Baseline(float(np.median(ratio)), k_band)


def errors(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Relative area error, estimate / truth - 1, summarised."""
    e = np.asarray(estimate, float) / np.asarray(truth, float) - 1
    a = np.abs(e)
    return {
        "n": int(len(e)),
        "median_abs_pct": round(100 * float(np.median(a)), 1),
        "p90_abs_pct": round(100 * float(np.percentile(a, 90)), 1),
        "bias_median_pct": round(100 * float(np.median(e)), 1),
        "within_10pct": round(float((a <= 0.10 + 1e-9).mean()), 3),  # tolerance: 110/100 - 1 > 0.1 in floats
    }
