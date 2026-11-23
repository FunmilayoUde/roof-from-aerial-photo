"""Stage 4: evaluate a trained U-Net on its held-out area, against the Stage 3 baseline.

    uv run python scripts/stage4_evaluate.py --run east

Per test house:
  pixels   roof-mask IoU (predicted roof vs labelled roof) and orientation accuracy on labelled
           roof pixels, skipping ignore pixels;
  area     the roof output's regions touching the house's BAG outline + 1.5 m (room for the
           lean); within them the orientation output says which pixels are flat. Sloped area:
             pitch     each pixel / cos(predicted pitch)
             fallback  flat pixels as-is, sloped pixels / cos(median training pitch)
             hybrid    footprint x (k_flat x flat share + k_pitched x (1 - flat share)): the
                       footprint gives the size, the photo only whether the roof is flat. k_flat
                       and k_pitched are fitted on training houses with their true flat share;
                       test houses use the predicted one. (Uses the footprint: photo + footprint.)
           compared with the LiDAR roof area, next to baseline B1 (footprint x k, k fitted on the
           same training houses).
Writes outputs/stage4/<run>/eval.json, per_house.csv and previews/. With --on val it scores the
validation sub-tile instead (eval_val.json, per_house_val.csv, no previews), for calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from scipy.ndimage import label  # noqa: E402
from shapely.geometry import shape  # noqa: E402

from roofphoto.baseline import errors, fit  # noqa: E402
from roofphoto.models import unet  # noqa: E402
from roofphoto.models.dataset import IGNORE, N_CLASSES, kept_chips, load_targets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = 0.08
CLEAN_MAX_UNCERTAIN = 0.10
MAX_PITCH_DEG = 70.0  # cap: 1/cos explodes near 90°


def footprints(tile: str) -> dict[str, object]:
    priv = {
        r["house_id"]: r["bag_id"]
        for r in csv.DictReader(open(ROOT / "data/processed/stage2" / tile / "houses_private.csv"))
    }
    feats = {
        f["properties"]["identificatie"]: f for f in json.loads((ROOT / "data/raw/bag" / f"{tile}.json").read_text())
    }
    return {h: shape(feats[b]["geometry"]) for h, b in priv.items()}


def manifest(tile: str) -> dict[str, dict]:
    return {r["house_id"]: r for r in csv.DictReader(open(ROOT / "outputs/stage2" / tile / "manifest.csv"))}


def flat_share_true(chip_folder: Path) -> float:
    """Flat share of a house's reliable roof area, from its facets (pitch under 5°)."""
    f = [x for x in json.loads((chip_folder / "facets.json").read_text()) if not x["ignored_in_labels"]]
    total = sum(x["area_m2"] for x in f)
    return sum(x["area_m2"] for x in f if x["pitch_deg"] < 5) / total if total else 0.0


def fit_hybrid(train_rows: list[dict], train_tiles: list[str]) -> tuple[float, float]:
    """Least squares for roof / footprint = k_flat x flat share + k_pitched x (1 - flat share)."""
    folders = {c.house_id: c.folder for c in kept_chips(ROOT, train_tiles)}
    f = np.array([flat_share_true(folders[r["house_id"]]) for r in train_rows])
    y = np.array([float(r["roof_area_m2"]) / float(r["footprint_m2"]) for r in train_rows])
    (k_flat, k_pitched), *_ = np.linalg.lstsq(np.column_stack([f, 1 - f]), y, rcond=None)
    return float(k_flat), float(k_pitched)


def median_training_pitch(train_tiles: list[str]) -> float:
    """Area-weighted median pitch of reliable sloped facets in the training houses."""
    p, w = [], []
    for c in kept_chips(ROOT, train_tiles):
        for f in json.loads((c.folder / "facets.json").read_text()):
            if not f["ignored_in_labels"] and f["pitch_deg"] >= 5:
                p.append(f["pitch_deg"])
                w.append(f["area_m2"])
    order = np.argsort(p)
    cum = np.cumsum(np.array(w)[order])
    return float(np.array(p)[order][np.searchsorted(cum, cum[-1] / 2)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--previews", type=int, default=8)
    ap.add_argument("--on", choices=["test", "val"], default="test", help="which held-out houses to score")
    ap.add_argument("--tiles", nargs="+", help="score these tiles instead (written as per_house_<tag>.csv)")
    ap.add_argument("--tag", default="", help="with --tiles: suffix for the output files")
    args = ap.parse_args()
    run = ROOT / "outputs" / "stage4" / args.run
    split = json.loads((run / "split.json").read_text())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = unet.build(pretrained=False)
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    model.to(device).eval()

    # Baseline B1 and the fallback pitch, both from the training houses only.
    train_rows = [r for t in split["train"] for r in manifest(t).values() if r["status"] == "kept"]
    b1 = fit(
        np.array([float(r["footprint_m2"]) for r in train_rows]),
        np.array([float(r["roof_area_m2"]) for r in train_rows]),
        np.array([int(r["built"]) for r in train_rows]),
    )
    fallback_pitch = median_training_pitch(split["train"])
    k_flat, k_pitched = fit_hybrid(train_rows, split["train"])

    rows, inter, union, correct, labelled = [], 0, 0, 0, 0
    (run / "previews").mkdir(exist_ok=True)
    suffix = "" if args.on == "test" else "_val"
    tiles = split[args.on]
    if args.tiles:
        suffix, tiles = f"_{args.tag or 'tiles'}", args.tiles
    if args.on == "val" or args.tiles:
        args.previews = 0
    for tile in tiles:
        fps, man = footprints(tile), manifest(tile)
        for chip in kept_chips(ROOT, [tile]):
            image, orient, pitch_t, roof_t = load_targets(chip)
            with torch.no_grad():
                out = model(torch.from_numpy(image)[None].to(device))
            logits, pitch_p, roof_l = unet.split(out)
            # Within the roof, the most likely non-background orientation: flat (1) or a slope (2-9).
            cls_roof = logits[0, 1:].argmax(0).cpu().numpy() + 1
            roof_p = torch.sigmoid(roof_l[0]).cpu().numpy() > 0.5
            cls = np.where(roof_p, cls_roof, 0)
            pp = np.clip(pitch_p[0].cpu().numpy(), 0, MAX_PITCH_DEG)

            known = roof_t != IGNORE
            pr, tr = roof_p & known, (roof_t == 1) & known
            inter += int((pr & tr).sum())
            union += int((pr | tr).sum())
            oriented = (orient >= 1) & (orient != IGNORE)
            correct += int(((cls == orient) & oriented).sum())
            labelled += int(oriented.sum())

            with rasterio.open(chip.folder / "photo.tif") as src:
                t = src.transform
            seed = rasterize([(fps[chip.house_id].buffer(1.5), 1)], out_shape=cls.shape, transform=t) == 1
            comp, _ = label(roof_p, structure=np.ones((3, 3)))
            ids = np.unique(comp[seed & (comp > 0)])
            house = np.isin(comp, ids) & (comp > 0)
            px = RES * RES
            sloped = house & (cls >= 2)
            r = man[chip.house_id]
            truth = float(r["roof_area_m2"])
            rows.append(
                {
                    "tile": tile,
                    "house_id": chip.house_id,
                    "truth_m2": truth,
                    "plan_m2": round(float(house.sum() * px), 1),
                    "model_pitch_m2": round(float((px / np.cos(np.radians(pp[house]))).sum()), 1),
                    "model_fallback_m2": round(
                        float((house & ~sloped).sum() * px + sloped.sum() * px / np.cos(np.radians(fallback_pitch))), 1
                    ),
                    "b1_m2": round(float(b1.b1([float(r["footprint_m2"])])[0]), 1),
                    "hybrid_m2": round(
                        float(r["footprint_m2"])
                        * (
                            k_flat * (flat_pred := float((house & ~sloped).sum() / max(house.sum(), 1)))
                            + k_pitched * (1 - flat_pred)
                        ),
                        1,
                    ),
                    "flat_share_pred": round(flat_pred, 3),
                    "house_iou": round(float((pr & tr & house).sum() / max(((pr & house) | tr).sum(), 1)), 3),
                    "uncertain_share": float(r["uncertain_area_share"]),
                    "unlabelled_interior_share": float(r["unlabelled_interior_share"]),
                    "flat_share": round(float((orient == 1).sum() / max(oriented.sum(), 1)), 3),
                    "pitch_mae_deg": round(float(np.nanmean(np.abs(pp - pitch_t)[np.isfinite(pitch_t)])), 2)
                    if np.isfinite(pitch_t).any()
                    else None,
                }
            )
            if len(rows) <= args.previews:
                preview(image, orient, cls, house, rows[-1], run / "previews" / f"{chip.house_id}.png")

    with (run / f"per_house{suffix}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    truth = np.array([r["truth_m2"] for r in rows])
    clean = np.array([r["uncertain_share"] <= CLEAN_MAX_UNCERTAIN for r in rows])
    methods = ["model_pitch_m2", "model_fallback_m2", "hybrid_m2", "b1_m2", "plan_m2"]
    result = {
        "run": args.run,
        "scored": tiles,
        "on": args.on,
        "houses": len(rows),
        "b1_factor": round(b1.k_all, 3),
        "fallback_pitch_deg": round(fallback_pitch, 1),
        "hybrid_factors": {"k_flat": round(k_flat, 3), "k_pitched": round(k_pitched, 3)},
        "pixels": {
            "roof_iou": round(inter / max(union, 1), 3),
            "orientation_accuracy": round(correct / max(labelled, 1), 3),
        },
        "pitch_mae_deg_median_house": round(
            float(np.median([r["pitch_mae_deg"] for r in rows if r["pitch_mae_deg"] is not None])), 2
        ),
        "area": {
            subset: {m: errors(np.array([r[m] for r in rows])[mask], truth[mask]) for m in methods}
            for subset, mask in (("all", np.ones(len(rows), bool)), ("clean", clean))
        },
    }
    # Error analysis: where does the better model variant beat or lose to B1?
    best = min(
        ("model_pitch_m2", "model_fallback_m2", "hybrid_m2"), key=lambda m: result["area"]["all"][m]["median_abs_pct"]
    )
    err_m = np.abs(np.array([r[best] for r in rows]) / truth - 1)
    err_b = np.abs(np.array([r["b1_m2"] for r in rows]) / truth - 1)
    analysis = {}
    for name, key, edges in (
        ("flat share of labelled roof", "flat_share", [0, 0.05, 0.5, 1.01]),
        ("ignored (uncertain) share", "uncertain_share", [0, 0.1, 0.3, 0.51]),
        ("unlabelled points inside walls", "unlabelled_interior_share", [0, 0.02, 0.05, 1.0]),
        ("roof area, m²", "truth_m2", [0, 150, 250, 1000]),
    ):
        v = np.array([r[key] for r in rows])
        analysis[name] = {
            f"{lo}-{hi}": {
                "houses": int(m.sum()),
                "model_median_abs_pct": round(100 * float(np.median(err_m[m])), 1),
                "b1_median_abs_pct": round(100 * float(np.median(err_b[m])), 1),
            }
            for lo, hi in zip(edges[:-1], edges[1:])
            if (m := (v >= lo) & (v < hi)).any()
        }
    result["error_analysis"] = {"model_variant": best, "by": analysis}
    result["model_beats_b1_houses"] = int((err_m < err_b).sum())
    (run / f"eval{suffix}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def preview(image, orient, cls, house, row, path) -> None:
    from roofphoto.models.dataset import IMAGENET_MEAN, IMAGENET_STD

    rgb = np.clip(np.moveaxis(image * IMAGENET_STD + IMAGENET_MEAN, 0, -1), 0, 1)
    cmap = plt.get_cmap("tab10", N_CLASSES)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    axs[0].imshow(rgb)
    axs[0].set_title(f"{row['house_id']}", fontsize=9)
    lab = np.ma.masked_where((orient == 0) | (orient == IGNORE), orient)
    axs[1].imshow(rgb)
    axs[1].imshow(lab, cmap=cmap, vmin=0, vmax=N_CLASSES - 1, alpha=0.55, interpolation="nearest")
    axs[1].set_title(f"label: orientation (LiDAR {row['truth_m2']} m²)", fontsize=9)
    axs[2].imshow(rgb)
    axs[2].imshow(
        np.ma.masked_where(cls == 0, cls), cmap=cmap, vmin=0, vmax=N_CLASSES - 1, alpha=0.55, interpolation="nearest"
    )
    axs[2].contour(house, levels=[0.5], colors="white", linewidths=1)
    axs[2].set_title(f"prediction (white: this house) {row['model_fallback_m2']} m², B1 {row['b1_m2']} m²", fontsize=9)
    for ax in axs:
        ax.set_axis_off()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
