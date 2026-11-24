"""Comparing two methods house by house: paired bootstrap interval and sign test (both sides)."""

from __future__ import annotations

from math import comb

import numpy as np


def compare(model_err: np.ndarray, base_err: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    """Difference in median |error| (model minus baseline) with a 95% paired-bootstrap interval,
    and how often the model is closer on the same house, with an exact sign test: one-sided
    (H1: the model is closer more often) and two-sided (H1: either method is closer more often).
    Report the two-sided value unless the direction was fixed in advance."""
    rng = np.random.default_rng(seed)
    model_err, base_err = np.asarray(model_err), np.asarray(base_err)
    n = len(model_err)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        diffs.append(np.median(model_err[i]) - np.median(base_err[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    wins = int((model_err < base_err).sum())
    p = sign_test_one_sided(wins, n)
    return {
        "median_diff_pts": round(100 * float(np.median(model_err) - np.median(base_err)), 1),
        "ci95_pts": [round(100 * float(lo), 1), round(100 * float(hi), 1)],
        "model_better_houses": f"{wins}/{n}",
        "sign_test_p_one_sided": round(p, 4),
        "sign_test_p_two_sided": round(sign_test_two_sided(wins, n), 4),
    }


def compare_blocked(
    model_err: np.ndarray, base_err: np.ndarray, blocks: np.ndarray, n_boot: int = 5000, seed: int = 0
) -> dict:
    """compare(), with the bootstrap resampling whole blocks (e.g. sub-tiles) instead of houses.

    Houses in one block share builders, eras and imaging conditions, so their errors are not
    independent; a house-level bootstrap then understates the uncertainty. Each replicate draws
    as many blocks as there are, with replacement, and pools their houses. With few blocks (5-9
    here) the percentile interval is itself approximate. Also returns how many blocks the model
    wins on (median error lower than the reference's within the block).
    """
    rng = np.random.default_rng(seed)
    model_err, base_err, blocks = np.asarray(model_err), np.asarray(base_err), np.asarray(blocks)
    names = np.unique(blocks)
    members = [np.flatnonzero(blocks == b) for b in names]
    diffs = []
    for _ in range(n_boot):
        i = np.concatenate([members[j] for j in rng.integers(0, len(names), len(names))])
        diffs.append(np.median(model_err[i]) - np.median(base_err[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    wins = sum(int(np.median(model_err[m]) < np.median(base_err[m])) for m in members)
    # With few blocks the percentile interval tends to be too narrow (every replicate mixes the
    # same few blocks); the exact sign test over blocks is the conservative check.
    return {
        "median_diff_pts": round(100 * float(np.median(model_err) - np.median(base_err)), 2),
        "ci95_pts": [round(100 * float(lo), 2), round(100 * float(hi), 2)],
        "blocks": len(names),
        "block_sizes": [len(m) for m in members],
        "model_better_blocks": f"{wins}/{len(names)}",
        "block_sign_test_p_two_sided": round(sign_test_two_sided(wins, len(names)), 4),
    }


def sign_test_one_sided(wins: int, n: int) -> float:
    """P(at least `wins` successes in n fair coin flips)."""
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2**n


def sign_test_two_sided(wins: int, n: int) -> float:
    """Exact two-sided sign test: twice the smaller tail, capped at 1."""
    lower = sum(comb(n, k) for k in range(0, wins + 1)) / 2**n
    return min(1.0, 2 * min(sign_test_one_sided(wins, n), lower))
