"""Pool Stage 4 cross-validation folds: every house tested once, by a model that never saw its area.

    uv run python scripts/stage4_cv_summary.py cv_west cv_centre cv_east

Reads outputs/stage4/<run>/per_house.csv for each fold. If the fold's validation houses were
scored too (stage4_evaluate.py --on val), it adds a calibrated model: the photo model's area x c,
with c = median(LiDAR area / model area) over that fold's validation houses, which neither the
model nor c ever saw at test time. Reports, for every area method, the
error over all houses and the clean subset; the model-minus-baseline difference in median error
with a paired bootstrap interval; a one-sided sign test (how often the model beats the baseline
on the same house); per-fold results; and an error analysis.
Writes outputs/stage4/cv_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import comb
from pathlib import Path

import numpy as np

from roofphoto.baseline import errors

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["model_pitch_m2", "model_pitch_cal_m2", "model_fallback_m2", "hybrid_m2", "b1_m2"]
CLEAN_MAX_UNCERTAIN = 0.10


def compare(model_err: np.ndarray, base_err: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    rows, calibration = [], {}
    for run in args.runs:
        folder = ROOT / "outputs" / "stage4" / run
        c = 1.0
        if (folder / "per_house_val.csv").exists():
            val = list(csv.DictReader(open(folder / "per_house_val.csv")))
            c = float(np.median([float(v["truth_m2"]) / float(v["model_pitch_m2"]) for v in val]))
            calibration[run] = {"factor": round(c, 4), "validation_houses": len(val)}
        for r in csv.DictReader(open(folder / "per_house.csv")):
            rows.append(r | {"fold": run, "model_pitch_cal_m2": float(r["model_pitch_m2"]) * c})
    truth = np.array([float(r["truth_m2"]) for r in rows])
    clean = np.array([float(r["uncertain_share"]) <= CLEAN_MAX_UNCERTAIN for r in rows])
    err = {m: np.abs(np.array([float(r[m]) for r in rows]) / truth - 1) for m in METHODS}
    est = {m: np.array([float(r[m]) for r in rows]) for m in METHODS}

    out = {"folds": args.runs, "houses": len(rows), "clean_houses": int(clean.sum()), "calibration": calibration}
    out["area"] = {
        name: {m: errors(est[m][mask], truth[mask]) for m in METHODS}
        for name, mask in (("all", np.ones(len(rows), bool)), ("clean", clean))
    }
    out["vs_baseline"] = {
        name: {m: compare(err[m][mask], err["b1_m2"][mask]) for m in METHODS if m != "b1_m2"}
        for name, mask in (("all", np.ones(len(rows), bool)), ("clean", clean))
    }
    out["per_fold"] = {
        run: {m: round(100 * float(np.median(err[m][np.array([r["fold"] == run for r in rows])])), 1) for m in METHODS}
        | {"houses": sum(r["fold"] == run for r in rows)}
        for run in args.runs
    }
    best = min(
        ("model_pitch_m2", "model_pitch_cal_m2", "model_fallback_m2", "hybrid_m2"), key=lambda m: np.median(err[m])
    )
    analysis = {}
    for name, key, edges in (
        ("flat share of labelled roof", "flat_share", [0, 0.05, 0.5, 1.01]),
        ("ignored (uncertain) share", "uncertain_share", [0, 0.1, 0.3, 0.51]),
        ("unlabelled points inside walls", "unlabelled_interior_share", [0, 0.02, 0.05, 1.0]),
        ("roof area, m²", "truth_m2", [0, 150, 250, 1000]),
    ):
        v = np.array([float(r[key]) for r in rows])
        analysis[name] = {
            f"{lo}-{hi}": {
                "houses": int(m.sum()),
                "model_median_abs_pct": round(100 * float(np.median(err[best][m])), 1),
                "b1_median_abs_pct": round(100 * float(np.median(err["b1_m2"][m])), 1),
            }
            for lo, hi in zip(edges[:-1], edges[1:])
            if (m := (v >= lo) & (v < hi)).any()
        }
    out["error_analysis"] = {"model_variant": best, "by": analysis}
    (ROOT / "outputs" / "stage4" / "cv_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
