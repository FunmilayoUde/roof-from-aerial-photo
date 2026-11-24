"""Per-house area results of the saved evaluations, as arrays, for the paper's tables and figures.

Each loader returns a dict of equal-length arrays: house_id, tile, truth (LiDAR roof area, m²),
and the predictions it compares. Nothing is recomputed from models; these read the per-house CSV
files written by stage4_evaluate.py and new_town_eval.py, and apply the same ensembling and
calibration as stage4_cv_summary.py and two_town_compare.py.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

BILTHOVEN_FOLDS = {
    "west": ["32CN1_03", "32CN1_08", "32CN1_13"],
    "centre": ["32CN1_04", "32CN1_09"],
    "east": ["32CN1_05", "32CN1_10", "32CN1_15", "32CN1_20"],
}
HEEMSTEDE_FOLDS = {"west": ["25CN1_01", "25CN1_06"], "centre": ["25CN1_02", "25CN1_07"], "east": ["25CN1_13"]}


def _rows(root: Path, run: str, name: str = "per_house.csv") -> list[dict]:
    with open(root / "outputs/stage4" / run / name) as fh:
        return list(csv.DictReader(fh))


def bilthoven_area(root: Path) -> dict[str, np.ndarray]:
    """Stage 4 final: mean of each fold's three seeds, times the cross-fold factor; footprint x k."""
    cal = json.loads((root / "outputs/stage4/cv_summary.json").read_text())["calibration"]
    out: dict[str, list] = {k: [] for k in ("house_id", "tile", "fold", "truth", "model", "b1", "flat", "unc")}
    for fold in BILTHOVEN_FOLDS:
        tables = [
            {r["house_id"]: r for r in _rows(root, run)} for run in (f"cv_{fold}", f"cv_{fold}_s1", f"cv_{fold}_s2")
        ]
        factor = cal[f"cv_{fold}"]["cross_fold_factor"]
        for hid, r in tables[0].items():
            out["house_id"].append(hid)
            out["tile"].append(r["tile"])
            out["fold"].append(fold)
            out["truth"].append(float(r["truth_m2"]))
            out["model"].append(np.mean([float(t[hid]["model_pitch_m2"]) for t in tables]) * factor)
            out["b1"].append(float(r["b1_m2"]))
            out["flat"].append(float(r["flat_share"]))
            out["unc"].append(float(r["uncertain_share"]))
    return {k: np.array(v) for k, v in out.items()}


def heemstede_untrained(root: Path) -> dict[str, np.ndarray]:
    """Bilthoven's nine area models on Heemstede, calibrated on Bilthoven (new_town_eval.py)."""
    with open(root / "outputs/new_town/heemstede/per_house.csv") as fh:
        rows = list(csv.DictReader(fh))
    return {
        "house_id": np.array([r["house_id"] for r in rows]),
        "tile": np.array([r["tile"] for r in rows]),
        "truth": np.array([float(r["truth_m2"]) for r in rows]),
        "model": np.array([float(r["model_cal_m2"]) for r in rows]),
        "b1": np.array([float(r["b1_m2"]) for r in rows]),
        "unc": np.array([float(r["uncertain_share"]) for r in rows]),
    }


SEEDS = ("", "_s1", "_s2")  # run-name suffixes of seeds 0, 1, 2


def two_town_area(root: Path, town: str, seeds: tuple[str, ...] = SEEDS) -> dict[str, np.ndarray]:
    """Two-town (bt_) and Bilthoven-only (cv_) models on one town, same folds and seeds, each
    calibrated by the other folds' houses and compared with its own footprint baseline
    (two_town_compare.py). With several seeds, each house's area is the mean over the fold's
    seeds before calibration, as for the Stage 4 ensemble; seeds=("",) gives seed 0 alone."""
    tags = {"bilthoven": ("bilt", ""), "heemstede": ("heem", "heem")}[town]
    out: dict[str, np.ndarray] = {}
    for label, prefix, tag in (("two_town", "bt_", tags[0]), ("bilthoven_only", "cv_", tags[1])):
        name = f"per_house_{tag}.csv" if tag else "per_house.csv"
        per_fold = {}
        for f in BILTHOVEN_FOLDS:
            tables = [_rows(root, f"{prefix}{f}{s}", name) for s in seeds]
            ids = [r["house_id"] for r in tables[0]]
            assert all([r["house_id"] for r in t] == ids for t in tables), f"{prefix}{f}: seeds disagree on houses"
            per_fold[f] = [
                dict(r, model_pitch_m2=float(np.mean([float(t[i]["model_pitch_m2"]) for t in tables])))
                for i, r in enumerate(tables[0])
            ]
        cal = []
        for f, rs in per_fold.items():
            # A house with no predicted roof still counts (as 100% off) but gives no ratio.
            others = [r for g, rr in per_fold.items() if g != f for r in rr if r["model_pitch_m2"] > 0]
            c = float(np.median([float(r["truth_m2"]) / r["model_pitch_m2"] for r in others]))
            cal += [r["model_pitch_m2"] * c for r in rs]
        rs = [r for f in BILTHOVEN_FOLDS for r in per_fold[f]]
        ids = np.array([r["house_id"] for r in rs])
        if "house_id" in out:
            assert (out["house_id"] == ids).all(), "both runs must score the same houses in the same order"
        out["house_id"] = ids
        out["tile"] = np.array([r["tile"] for r in rs])
        out["truth"] = np.array([float(r["truth_m2"]) for r in rs])
        out[label] = np.array(cal)
        out[f"{label}_raw"] = np.array([r["model_pitch_m2"] for r in rs])  # before calibration
        out[f"{label}_b1"] = np.array([float(r["b1_m2"]) for r in rs])
    return out


def abs_err(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """|pred / truth - 1| as a fraction."""
    return np.abs(np.asarray(pred) / np.asarray(truth) - 1)
