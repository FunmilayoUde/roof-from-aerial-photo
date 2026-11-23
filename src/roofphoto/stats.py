"""Comparing two methods house by house: paired bootstrap interval and one-sided sign test."""

from __future__ import annotations

from math import comb

import numpy as np


def compare(model_err: np.ndarray, base_err: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    """Difference in median |error| (model minus baseline) with a 95% paired-bootstrap interval,
    and how often the model is closer on the same house (one-sided sign test)."""
    rng = np.random.default_rng(seed)
    model_err, base_err = np.asarray(model_err), np.asarray(base_err)
    n = len(model_err)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        diffs.append(np.median(model_err[i]) - np.median(base_err[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    wins = int((model_err < base_err).sum())
    p = sum(comb(n, k) for k in range(wins, n + 1)) / 2**n
    return {
        "median_diff_pts": round(100 * float(np.median(model_err) - np.median(base_err)), 1),
        "ci95_pts": [round(100 * float(lo), 1), round(100 * float(hi), 1)],
        "model_better_houses": f"{wins}/{n}",
        "sign_test_p_one_sided": round(p, 4),
    }
