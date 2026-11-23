"""Does training on both towns help? Two-town models (bt_<fold>) against the Bilthoven-only models
(cv_<fold>, seed 0), on the same held-out houses of each fold, per town.

    uv run python scripts/two_town_compare.py

Both are seed 0 with the Stage 4 recipe; the only difference is Heemstede's training houses. Per
town: median |area error| and bias (uncalibrated, and calibrated by the other folds' houses as in
Stage 4), each model against its own footprint baseline, and two-town against Bilthoven-only on
the same houses. Then pitch per facet on Heemstede, the quantity that moved most between towns.
Writes outputs/two_town/summary.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from roofphoto.baseline import errors
from roofphoto.models import unet
from roofphoto.models.dataset import kept_chips
from roofphoto.stats import compare

ROOT = Path(__file__).resolve().parents[1]
FOLDS = {
    "west": ["25CN1_01", "25CN1_06"],
    "centre": ["25CN1_02", "25CN1_07"],
    "east": ["25CN1_13"],
}


def rows(run: str, tag: str) -> list[dict]:
    name = "per_house.csv" if not tag else f"per_house_{tag}.csv"
    return list(csv.DictReader(open(ROOT / "outputs/stage4" / run / name)))


def calibrated(per_fold: dict[str, list[dict]]) -> dict[str, np.ndarray]:
    """Area x factor from the other folds' houses (median LiDAR / model), as in Stage 4."""
    out = {}
    for f, rs in per_fold.items():
        others = [r for g, rr in per_fold.items() if g != f for r in rr]
        c = float(np.median([float(r["truth_m2"]) / float(r["model_pitch_m2"]) for r in others]))
        out[f] = np.array([float(r["model_pitch_m2"]) for r in rs]) * c
    return out


def town_summary(per_fold_bt: dict, per_fold_one: dict) -> dict:
    res = {}
    for name, per_fold in (("two_town", per_fold_bt), ("bilthoven_only", per_fold_one)):
        rs = [r for f in FOLDS for r in per_fold[f]]
        truth = np.array([float(r["truth_m2"]) for r in rs])
        raw = np.array([float(r["model_pitch_m2"]) for r in rs])
        cal = np.concatenate([calibrated(per_fold)[f] for f in FOLDS])
        b1 = np.array([float(r["b1_m2"]) for r in rs])
        res[name] = {
            "houses": len(rs),
            "model_uncalibrated": errors(raw, truth),
            "model_calibrated": errors(cal, truth),
            "baseline_b1": errors(b1, truth),
            "model_cal_vs_baseline": compare(np.abs(cal / truth - 1), np.abs(b1 / truth - 1)),
            "_cal_err": np.abs(cal / truth - 1),
            "_ids": [r["house_id"] for r in rs],
        }
    a, b = res["two_town"], res["bilthoven_only"]
    assert a["_ids"] == b["_ids"], "the two runs must score the same houses in the same order"
    res["two_town_vs_bilthoven_only"] = compare(a.pop("_cal_err"), b.pop("_cal_err"))
    a.pop("_ids")
    b.pop("_ids")
    return res


def pitch_heemstede(device) -> dict:
    """Area-weighted pitch error and bias per facet on each fold's Heemstede test houses."""
    out = {}
    for label, prefix in (("two_town", "bt_"), ("bilthoven_only", "cv_")):
        t_all, p_all, w_all = [], [], []
        for fold, tiles in FOLDS.items():
            m = unet.build(pretrained=False)
            m.load_state_dict(torch.load(ROOT / "outputs/stage4" / f"{prefix}{fold}" / "model.pt", map_location="cpu"))
            m.to(device).eval()
            import rasterio

            from roofphoto.models.dataset import load_targets

            for chip in kept_chips(ROOT, tiles):
                image, _, _, _ = load_targets(chip)
                with rasterio.open(chip.folder / f"labels_{chip.labels}.tif") as src:
                    band = src.read(1)
                with torch.no_grad():
                    pitch = unet.split(m(torch.from_numpy(image)[None].to(device)))[1][0].cpu().numpy()
                for f in json.loads((chip.folder / "facets.json").read_text()):
                    sel = band == f["class"]
                    if f["ignored_in_labels"] or sel.sum() < 20:
                        continue
                    t_all.append(f["pitch_deg"])
                    p_all.append(float(np.clip(pitch[sel], 0, 70).mean()))
                    w_all.append(f["area_m2"])
        t, p, w = np.array(t_all), np.array(p_all), np.array(w_all)
        steep = (t >= 40) & (t < 50)
        out[label] = {
            "facets": len(t),
            "mae_deg": round(float(np.average(np.abs(p - t), weights=w)), 2),
            "bias_deg": round(float(np.average(p - t, weights=w)), 2),
            "pred_at_true_40_50": round(float(np.average(p[steep], weights=w[steep])), 1),
            "true_at_40_50": round(float(np.average(t[steep], weights=w[steep])), 1),
        }
    return out


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = ROOT / "outputs/two_town"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "bilthoven": town_summary({f: rows(f"bt_{f}", "bilt") for f in FOLDS}, {f: rows(f"cv_{f}", "") for f in FOLDS}),
        "heemstede": town_summary(
            {f: rows(f"bt_{f}", "heem") for f in FOLDS}, {f: rows(f"cv_{f}", "heem") for f in FOLDS}
        ),
        "heemstede_pitch_per_facet": pitch_heemstede(device),
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
