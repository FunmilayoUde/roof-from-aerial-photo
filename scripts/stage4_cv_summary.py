"""Pool Stage 4 cross-validation folds: every house tested once, by a model that never saw its area.

    uv run python scripts/stage4_cv_summary.py cv_west cv_centre cv_east
    uv run python scripts/stage4_cv_summary.py cv_west,cv_west_s1,cv_west_s2 cv_centre,... cv_east,...

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
from pathlib import Path

import numpy as np

from roofphoto.baseline import errors
from roofphoto.stats import compare

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["model_pitch_m2", "model_pitch_xcal_m2", "model_pitch_cal_m2", "model_fallback_m2", "hybrid_m2", "b1_m2"]
CLEAN_MAX_UNCERTAIN = 0.10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    rows, calibration = [], {}
    # A fold is one run or several seeds of the same split, comma-separated: cv_west,cv_west_s1,...
    # With several seeds, each house's model area is the mean over the seeds' predictions.
    per_seed = {}
    for group in args.runs:
        runs = group.split(",")
        fold = runs[0]
        tables = {}
        for run in runs:
            folder = ROOT / "outputs" / "stage4" / run
            if (folder / "per_house.csv").exists():
                tables[run] = {r["house_id"]: r for r in csv.DictReader(open(folder / "per_house.csv"))}
        if not tables:
            raise SystemExit(f"{group}: no per_house.csv")
        first = next(iter(tables.values()))
        for run, t in tables.items():
            per_seed.setdefault(run.split("_s")[-1] if "_s" in run else "0", {}).update(
                {h: float(r["model_pitch_m2"]) for h, r in t.items()}
            )
        # Validation-fitted factor (kept for comparison): mean of the seeds' factors.
        cs = []
        for run in tables:
            v = ROOT / "outputs" / "stage4" / run / "per_house_val.csv"
            if v.exists():
                val = list(csv.DictReader(open(v)))
                cs.append(float(np.median([float(x["truth_m2"]) / float(x["model_pitch_m2"]) for x in val])))
        c = float(np.mean(cs)) if cs else 1.0
        calibration[fold] = {"validation_factor": round(c, 4), "seeds": len(tables)}
        for h, r in first.items():
            pred = float(np.mean([float(t[h]["model_pitch_m2"]) for t in tables.values()]))
            rows.append(r | {"fold": fold, "model_pitch_m2": pred, "model_pitch_cal_m2": pred * c})

    # Cross-fold calibration: fold F's factor comes from the OTHER folds' test houses, each predicted
    # by a model that never saw it; none of them is in F's test set.
    folds = [g.split(",")[0] for g in args.runs]
    for f in folds:
        other = [r for r in rows if r["fold"] != f]
        c = float(np.median([float(r["truth_m2"]) / r["model_pitch_m2"] for r in other])) if other else 1.0
        calibration[f]["cross_fold_factor"] = round(c, 4)
        calibration[f]["cross_fold_houses"] = len(other)
        for r in rows:
            if r["fold"] == f:
                r["model_pitch_xcal_m2"] = r["model_pitch_m2"] * c
    truth = np.array([float(r["truth_m2"]) for r in rows])
    clean = np.array([float(r["uncertain_share"]) <= CLEAN_MAX_UNCERTAIN for r in rows])
    err = {m: np.abs(np.array([float(r[m]) for r in rows]) / truth - 1) for m in METHODS}
    est = {m: np.array([float(r[m]) for r in rows]) for m in METHODS}

    out = {"folds": args.runs, "houses": len(rows), "clean_houses": int(clean.sum()), "calibration": calibration}
    # Each seed on its own (houses it predicted), to show how much training randomness moves the result.
    tr = {r["house_id"]: float(r["truth_m2"]) for r in rows}
    out["per_seed_median_abs_pct"] = {
        seed: {
            "houses": len(p),
            "median_abs_pct": round(100 * float(np.median([abs(v / tr[h] - 1) for h, v in p.items()])), 1),
        }
        for seed, p in sorted(per_seed.items())
    }
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
        for run in folds
    }
    best = min(
        ("model_pitch_m2", "model_pitch_xcal_m2", "model_fallback_m2", "hybrid_m2"), key=lambda m: np.median(err[m])
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
