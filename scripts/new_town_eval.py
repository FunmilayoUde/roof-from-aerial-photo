"""Test the Bilthoven models on another town, with nothing about that town used to set anything.

    uv run python scripts/new_town_eval.py --name heemstede 25CN1_01 25CN1_02 25CN1_06 25CN1_07 25CN1_13

Needs Stage 2 run on those tiles. Per kept house in the new town:
  area         each of the 9 Bilthoven area models (3 folds x 3 seeds; none saw this town) gives a
               roof area as in Stage 4 (predicted roof touching the BAG outline + 1.5 m, per-pixel
               area / cos(predicted pitch)); the 9 are averaged, then multiplied by the calibration
               factor from all 351 Bilthoven out-of-fold predictions.
  baseline     footprint x the factor fitted on all Bilthoven houses (B1).
  reference    footprint x a factor fitted on this town itself. Not a fair competitor: it needs
               local LiDAR. It shows how much of any gap is this town's roofs being different.
  pixels       roof IoU and pitch error of the area ensemble; orientation accuracy of the 9
               orientation models averaged.
  edges        rule-based edges from the ensemble roof and orientation, scored as in Stage 5.
Writes outputs/new_town/<name>/results.json and per_house.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, label

from roofphoto.baseline import errors, fit
from roofphoto.data.lookup import footprints, manifest
from roofphoto.edges import edge_raster, facets_from_orientation, score_edges
from roofphoto.models import unet
from roofphoto.models.dataset import IGNORE, kept_chips, load_targets
from roofphoto.stats import compare

ROOT = Path(__file__).resolve().parents[1]
BILTHOVEN = [f"32CN1_{k}" for k in ("03", "04", "05", "08", "09", "10", "13", "15", "20")]
HEEMSTEDE = ["25CN1_01", "25CN1_02", "25CN1_06", "25CN1_07", "25CN1_13"]
FOLDS = ("west", "centre", "east")
AREA_RUNS = [f"cv_{f}{s}" for f in FOLDS for s in ("", "_s1", "_s2")]
TWO_TOWN_RUNS = [f"bt_{f}" for f in FOLDS]  # trained on Bilthoven + Heemstede, seed 0
ORIENT_RUNS = [f"s5w_{f}_geom_s{s}" for f in FOLDS for s in range(3)]
RES = 0.08
MAX_PITCH_DEG = 70.0
CLEAN_MAX_UNCERTAIN = 0.10


def load(run: str, device) -> torch.nn.Module:
    m = unet.build(pretrained=False)
    m.load_state_dict(torch.load(ROOT / "outputs/stage4" / run / "model.pt", map_location="cpu"))
    return m.to(device).eval()


def two_town_calibration() -> float:
    """median(LiDAR area / model area) over both towns' out-of-fold houses of the two-town models."""
    ratios = []
    for f in FOLDS:
        for tag in ("bilt", "heem"):
            for r in csv.DictReader(open(ROOT / "outputs/stage4" / f"bt_{f}" / f"per_house_{tag}.csv")):
                if float(r["model_pitch_m2"]) > 0:
                    ratios.append(float(r["truth_m2"]) / float(r["model_pitch_m2"]))
    return float(np.median(ratios))


def bilthoven_calibration() -> float:
    """median(LiDAR area / model area) over all Bilthoven houses, each predicted out of fold
    (mean over its fold's 3 seeds), as in the Stage 4 cross-fold calibration."""
    ratios = []
    for f in FOLDS:
        tables = [
            {r["house_id"]: r for r in csv.DictReader(open(ROOT / "outputs/stage4" / f"cv_{f}{s}" / "per_house.csv"))}
            for s in ("", "_s1", "_s2")
        ]
        for h, r in tables[0].items():
            pred = np.mean([float(t[h]["model_pitch_m2"]) for t in tables])
            ratios.append(float(r["truth_m2"]) / pred)
    return float(np.median(ratios))


def house_region(roof: np.ndarray, seed: np.ndarray) -> np.ndarray:
    comp, _ = label(roof, structure=np.ones((3, 3)))
    return np.isin(comp, np.unique(comp[seed & (comp > 0)])) & (comp > 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tiles", nargs="+")
    ap.add_argument("--name", required=True)
    ap.add_argument(
        "--models",
        choices=["bilthoven", "two-town"],
        default="bilthoven",
        help="bilthoven: 9 Bilthoven-only area models; two-town: the 3 models trained on Bilthoven + Heemstede",
    )
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = ROOT / "outputs/new_town" / (args.name + ("" if args.models == "bilthoven" else "_two_town"))
    out.mkdir(parents=True, exist_ok=True)

    train_towns = BILTHOVEN + (HEEMSTEDE if args.models == "two-town" else [])
    bilt = [r for t in train_towns for r in manifest(ROOT, t).values() if r["status"] == "kept"]
    b1 = fit(
        np.array([float(r["footprint_m2"]) for r in bilt]),
        np.array([float(r["roof_area_m2"]) for r in bilt]),
        np.array([int(r["built"]) for r in bilt]),
    )
    cal = bilthoven_calibration() if args.models == "bilthoven" else two_town_calibration()
    area_models = [load(r, device) for r in (AREA_RUNS if args.models == "bilthoven" else TWO_TOWN_RUNS)]
    orient_models = [load(r, device) for r in ORIENT_RUNS]

    rows, inter, union, o_hit, o_n, edge_counts = [], 0, 0, 0, 0, {}
    for tile in args.tiles:
        fps, man = footprints(ROOT, tile), manifest(ROOT, tile)
        for chip in kept_chips(ROOT, [tile]):
            image, orient_t, pitch_t, roof_t = load_targets(chip)
            with rasterio.open(chip.folder / "photo.tif") as src:
                t = src.transform
            with rasterio.open(chip.folder / f"labels_{chip.labels}.tif") as src:
                edges_t = src.read(2)
            seed = rasterize([(fps[chip.house_id].buffer(1.5), 1)], out_shape=orient_t.shape, transform=t) == 1
            x = torch.from_numpy(image)[None].to(device)
            areas, roofs, pitches = [], [], []
            with torch.no_grad():
                for m in area_models:
                    _, pitch_p, roof_l = unet.split(m(x))
                    roof_p = torch.sigmoid(roof_l[0]).cpu().numpy()
                    pp = np.clip(pitch_p[0].cpu().numpy(), 0, MAX_PITCH_DEG)
                    house = house_region(roof_p > 0.5, seed)
                    areas.append(float((RES * RES / np.cos(np.radians(pp[house]))).sum()))
                    roofs.append(roof_p)
                    pitches.append(pp)
                probs = np.mean([torch.softmax(unet.split(m(x))[0][0, 1:], 0).cpu().numpy() for m in orient_models], 0)
            roof_e = np.mean(roofs, 0) > 0.5
            house = house_region(roof_e, seed)
            known = roof_t != IGNORE
            inter += int((roof_e & (roof_t == 1) & known).sum())
            union += int(((roof_e | (roof_t == 1)) & known).sum())
            orient_p = probs.argmax(0) + 1
            oriented = (orient_t >= 1) & (orient_t != IGNORE)
            o_hit += int((orient_p[oriented] == orient_t[oriented]).sum())
            o_n += int(oriented.sum())
            pitch_e = np.mean(pitches, 0)
            has_pitch = np.isfinite(pitch_t)
            facets, cls = facets_from_orientation(house, orient_p, 150)
            ignore = binary_dilation(edges_t == IGNORE, iterations=3)
            for k, v in score_edges(edge_raster(facets, cls), edges_t, ignore, 3.0).items():
                edge_counts[k] = [a + b for a, b in zip(edge_counts.get(k, [0, 0, 0, 0]), v)]
            r = man[chip.house_id]
            model = float(np.mean(areas))
            rows.append(
                {
                    "tile": tile,
                    "house_id": chip.house_id,
                    "truth_m2": float(r["roof_area_m2"]),
                    "footprint_m2": float(r["footprint_m2"]),
                    "built": int(r["built"]),
                    "uncertain_share": float(r["uncertain_area_share"]),
                    "model_m2": round(model, 1),
                    "model_cal_m2": round(model * cal, 1),
                    "b1_m2": round(float(r["footprint_m2"]) * b1.k_all, 1),
                    "pitch_mae_deg": round(float(np.abs(pitch_e - pitch_t)[has_pitch].mean()), 2)
                    if has_pitch.any()
                    else None,
                }
            )
            print(
                f"  {chip.house_id}: LiDAR {rows[-1]['truth_m2']}, "
                f"model {rows[-1]['model_cal_m2']}, B1 {rows[-1]['b1_m2']}"
            )

    with (out / "per_house.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    truth = np.array([r["truth_m2"] for r in rows])
    fp = np.array([r["footprint_m2"] for r in rows])
    local = fit(fp, truth, np.array([r["built"] for r in rows]))
    est = {
        "model_calibrated": np.array([r["model_cal_m2"] for r in rows]),
        "model_uncalibrated": np.array([r["model_m2"] for r in rows]),
        "baseline_b1": np.array([r["b1_m2"] for r in rows]),
        "reference_local_factor": fp * local.k_all,
    }
    clean = np.array([r["uncertain_share"] <= CLEAN_MAX_UNCERTAIN for r in rows])
    err = {k: np.abs(v / truth - 1) for k, v in est.items()}
    result = {
        "town": args.name,
        "models": args.models,
        "tiles": args.tiles,
        "houses": len(rows),
        "clean_houses": int(clean.sum()),
        "factors": {
            "baseline_b1_from_training_towns": round(b1.k_all, 3),
            "calibration_from_training_towns": round(cal, 4),
            "local_reference": round(local.k_all, 3),
        },
        "area": {
            name: {k: errors(v[mask], truth[mask]) for k, v in est.items()}
            for name, mask in (("all", np.ones(len(rows), bool)), ("clean", clean))
        },
        "vs_baseline": {
            name: compare(err["model_calibrated"][mask], err["baseline_b1"][mask])
            for name, mask in (("all", np.ones(len(rows), bool)), ("clean", clean))
        },
        "pixels": {
            "roof_iou": round(inter / max(union, 1), 3),
            "orientation_accuracy": round(o_hit / max(o_n, 1), 3),
            "pitch_mae_deg_median_house": round(
                float(np.median([r["pitch_mae_deg"] for r in rows if r["pitch_mae_deg"] is not None])), 2
            ),
        },
        "edges": {
            k: {"precision": round(a / b, 3) if b else None, "recall": round(c / d, 3) if d else None}
            for k, (a, b, c, d) in edge_counts.items()
        },
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
