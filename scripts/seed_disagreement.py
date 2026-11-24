"""Does disagreement between training seeds predict error? (review item 29)

    uv run python scripts/seed_disagreement.py            # area (from saved files) and pitch (runs models)
    uv run python scripts/seed_disagreement.py --area-only

Area, per house: the spread of the three seeds' area estimates (standard deviation / mean)
against the ensemble's |area error|, Bilthoven out of fold (cv_<fold>, _s1, _s2).
Pitch, per reliable facet: the spread of the three seeds' facet-mean pitch against the
ensemble's |pitch error|, Bilthoven out of fold.
For each: Spearman rank correlation with a bootstrap 95% interval over houses, and the median
error by tertile of disagreement. A useful uncertainty signal would show error rising with
disagreement. Writes outputs/seed_disagreement/summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from roofphoto.results import BILTHOVEN_FOLDS, SEEDS

ROOT = Path(__file__).resolve().parents[1]


def spearman_ci(x, y, groups, n_boot=2000, seed=0):
    """Spearman's rho with a bootstrap interval resampling groups (houses), since facets of one
    house are not independent."""
    rng = np.random.default_rng(seed)
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    names = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in names]
    rho = float(spearmanr(x, y).statistic)
    boots = []
    for _ in range(n_boot):
        i = np.concatenate([members[j] for j in rng.integers(0, len(names), len(names))])
        boots.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(rho, 3), [round(float(lo), 3), round(float(hi), 3)]


def tertiles(disagree, err):
    edges = np.percentile(disagree, [100 / 3, 200 / 3])
    bins = np.digitize(disagree, edges)
    return {
        name: {"n": int((bins == b).sum()), "median_abs_error": round(float(np.median(err[bins == b])), 2)}
        for b, name in enumerate(("low", "middle", "high"))
    }


def area_part() -> dict:
    cal = json.loads((ROOT / "outputs/stage4/cv_summary.json").read_text())["calibration"]
    ids, spread, err = [], [], []
    for fold in BILTHOVEN_FOLDS:
        files = [ROOT / "outputs/stage4" / f"cv_{fold}{s}" / "per_house.csv" for s in SEEDS]
        tables = [{r["house_id"]: r for r in csv.DictReader(open(f))} for f in files]
        c = cal[f"cv_{fold}"]["cross_fold_factor"]
        for hid, r in tables[0].items():
            a = np.array([float(t[hid]["model_pitch_m2"]) for t in tables])
            truth = float(r["truth_m2"])
            ids.append(hid)
            spread.append(100 * a.std() / a.mean())
            err.append(100 * abs(a.mean() * c / truth - 1))
    spread, err = np.array(spread), np.array(err)
    rho, ci = spearman_ci(spread, err, ids)
    return {
        "houses": len(err),
        "spread_definition": "std/mean of the three seeds' area, %",
        "median_spread_pct": round(float(np.median(spread)), 2),
        "spearman_rho": rho,
        "rho_ci95": ci,
        "error_by_spread_tertile_pct": tertiles(spread, err),
    }


def pitch_part() -> dict:
    import rasterio
    import torch

    from roofphoto.models import unet
    from roofphoto.models.dataset import kept_chips, load_targets

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    houses, spread, err, area = [], [], [], []
    for fold, tiles in BILTHOVEN_FOLDS.items():
        models = []
        for s in SEEDS:
            m = unet.build(pretrained=False)
            m.load_state_dict(torch.load(ROOT / "outputs/stage4" / f"cv_{fold}{s}" / "model.pt", map_location="cpu"))
            models.append(m.to(device).eval())
        for chip in kept_chips(ROOT, tiles):
            image, _, _, _ = load_targets(chip)
            with rasterio.open(chip.folder / f"labels_{chip.labels}.tif") as src:
                band = src.read(1)
            x = torch.from_numpy(image)[None].to(device)
            with torch.no_grad():
                pitches = [np.clip(unet.split(m(x))[1][0].cpu().numpy(), 0, 70) for m in models]
            for f in json.loads((chip.folder / "facets.json").read_text()):
                sel = band == f["class"]
                if f["ignored_in_labels"] or sel.sum() < 20:
                    continue
                per_seed = np.array([p[sel].mean() for p in pitches])
                houses.append(chip.house_id)
                spread.append(float(per_seed.std()))
                err.append(abs(float(per_seed.mean()) - f["pitch_deg"]))
                area.append(f["area_m2"])
        print(f"{fold}: {len(err)} facets so far", flush=True)
    spread, err = np.array(spread), np.array(err)
    rho, ci = spearman_ci(spread, err, houses)
    return {
        "facets": len(err),
        "spread_definition": "std of the three seeds' facet-mean pitch, degrees",
        "median_spread_deg": round(float(np.median(spread)), 2),
        "spearman_rho": rho,
        "rho_ci95_resampling_houses": ci,
        "error_by_spread_tertile_deg": tertiles(spread, err),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--area-only", action="store_true")
    args = ap.parse_args()
    out_dir = ROOT / "outputs/seed_disagreement"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"area": area_part()}
    if not args.area_only:
        result["pitch"] = pitch_part()
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
