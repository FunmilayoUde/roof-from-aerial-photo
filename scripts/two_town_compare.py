"""Does training on both towns help? Two-town models (bt_<fold>) against Bilthoven-only models
(cv_<fold>), same folds and seeds, on the same held-out houses, per town.

    uv run python scripts/two_town_compare.py

Both use the Stage 4 recipe; the only difference is Heemstede's training houses. Main result:
three-seed ensembles on both sides (seeds 0-2, area averaged per house, then calibrated by the
other folds' houses as in Stage 4). Also reported: seed 0 alone (the earlier single-seed result)
and each seed's own median error, to show the spread. Per town: median |area error| and bias,
each model against its own footprint baseline, and two-town against Bilthoven-only. Then pitch
per facet on Heemstede, the quantity that moved most between towns, from the three-seed mean.
Needs per_house_<tag>.csv for every run (stage4_evaluate.py --tiles ... --tag bilt|heem).
Writes outputs/two_town/summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from roofphoto.baseline import errors
from roofphoto.models import unet
from roofphoto.models.dataset import kept_chips, load_targets
from roofphoto.results import HEEMSTEDE_FOLDS, SEEDS, abs_err, two_town_area
from roofphoto.stats import compare

ROOT = Path(__file__).resolve().parents[1]


def town_summary(town: str, seeds: tuple[str, ...]) -> dict:
    d = two_town_area(ROOT, town, seeds)
    res = {"seeds": len(seeds), "houses": len(d["truth"])}
    for label in ("two_town", "bilthoven_only"):
        res[label] = {
            "model_calibrated": errors(d[label], d["truth"]),
            "model_uncalibrated": errors(d[f"{label}_raw"], d["truth"]),
            "baseline_b1": errors(d[f"{label}_b1"], d["truth"]),
            "model_vs_baseline": compare(abs_err(d[label], d["truth"]), abs_err(d[f"{label}_b1"], d["truth"])),
        }
    res["two_town_vs_bilthoven_only"] = compare(
        abs_err(d["two_town"], d["truth"]), abs_err(d["bilthoven_only"], d["truth"])
    )
    return res


def per_seed_medians(town: str) -> dict:
    out = {}
    for s in SEEDS:
        d = two_town_area(ROOT, town, (s,))
        out[f"seed{SEEDS.index(s)}"] = {
            k: round(100 * float(np.median(abs_err(d[k], d["truth"]))), 2) for k in ("two_town", "bilthoven_only")
        }
    return out


def pitch_heemstede(device) -> dict:
    """Area-weighted pitch error and bias per facet on each fold's Heemstede test houses, from the
    mean predicted pitch of the fold's three seeds."""
    out = {}
    for label, prefix in (("two_town", "bt_"), ("bilthoven_only", "cv_")):
        t_all, p_all, w_all = [], [], []
        for fold, tiles in HEEMSTEDE_FOLDS.items():
            models = []
            for s in SEEDS:
                m = unet.build(pretrained=False)
                weights = ROOT / "outputs/stage4" / f"{prefix}{fold}{s}" / "model.pt"
                m.load_state_dict(torch.load(weights, map_location="cpu"))
                models.append(m.to(device).eval())
            for chip in kept_chips(ROOT, tiles):
                image, _, _, _ = load_targets(chip)
                with rasterio.open(chip.folder / f"labels_{chip.labels}.tif") as src:
                    band = src.read(1)
                x = torch.from_numpy(image)[None].to(device)
                with torch.no_grad():
                    pitch = np.mean([unet.split(m(x))[1][0].cpu().numpy() for m in models], 0)
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
        "three_seeds": {t: town_summary(t, SEEDS) for t in ("heemstede", "bilthoven")},
        "seed0_only": {t: town_summary(t, ("",)) for t in ("heemstede", "bilthoven")},
        "per_seed_median_abs_pct": {t: per_seed_medians(t) for t in ("heemstede", "bilthoven")},
        "heemstede_pitch_per_facet_three_seeds": pitch_heemstede(device),
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
