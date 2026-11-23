"""Stage 5: pitch per facet, from the photo, against the LiDAR pitch.

    uv run python scripts/pitch_per_facet.py

For every reliable facet of every kept house: the mean of the models' predicted pitch over the
facet's pixels (its true outline from the labels, so this measures pitch, not segmentation).
  Bilthoven  each house by its own fold's 3 area models (never trained on it)
  Heemstede  all 9 Bilthoven area models
Baseline: one constant pitch for every facet, the area-weighted median of all Bilthoven facets.
Reported overall, by town, and by band of true pitch: if the model pulls pitch towards what it
learned, steep facets read too flat and shallow ones too steep.
Writes outputs/pitch/facets.csv, summary.json and pitch.png.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402

from roofphoto.models import unet  # noqa: E402
from roofphoto.models.dataset import kept_chips, load_targets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ("west", "centre", "east")
HEEMSTEDE = ["25CN1_01", "25CN1_02", "25CN1_06", "25CN1_07", "25CN1_13"]
BANDS = [0, 5, 20, 30, 40, 50, 90]


def load(run: str, device) -> torch.nn.Module:
    m = unet.build(pretrained=False)
    m.load_state_dict(torch.load(ROOT / "outputs/stage4" / run / "model.pt", map_location="cpu"))
    return m.to(device).eval()


def facets_of(chip, models, device, town) -> list[dict]:
    image, _, _, _ = load_targets(chip)
    with rasterio.open(chip.folder / f"labels_{chip.labels}.tif") as src:
        facet_band = src.read(1)
    x = torch.from_numpy(image)[None].to(device)
    with torch.no_grad():
        pitch = np.mean([unet.split(m(x))[1][0].cpu().numpy() for m in models], 0)
    out = []
    for f in json.loads((chip.folder / "facets.json").read_text()):
        if f["ignored_in_labels"]:
            continue
        m = facet_band == f["class"]
        if m.sum() < 20:
            continue
        out.append(
            {
                "town": town,
                "house_id": chip.house_id,
                "facet_id": f["facet_id"],
                "area_m2": f["area_m2"],
                "true_pitch": f["pitch_deg"],
                "pred_pitch": round(float(np.clip(pitch[m], 0, 70).mean()), 2),
            }
        )
    return out


def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, c[-1] / 2)])


def summary(rows: list[dict], constant: float) -> dict:
    t = np.array([r["true_pitch"] for r in rows])
    p = np.array([r["pred_pitch"] for r in rows])
    w = np.array([r["area_m2"] for r in rows])
    slope, intercept = np.polyfit(t, p, 1, w=np.sqrt(w))
    out = {
        "facets": len(rows),
        "model_mae_deg": round(float(np.average(np.abs(p - t), weights=w)), 2),
        "model_bias_deg": round(float(np.average(p - t, weights=w)), 2),
        "constant_mae_deg": round(float(np.average(np.abs(constant - t), weights=w)), 2),
        "fit_pred_vs_true": {"slope": round(float(slope), 3), "intercept_deg": round(float(intercept), 2)},
        "by_true_pitch": {},
    }
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        m = (t >= lo) & (t < hi)
        if m.sum() >= 5:
            out["by_true_pitch"][f"{lo}-{hi}"] = {
                "facets": int(m.sum()),
                "true_mean": round(float(np.average(t[m], weights=w[m])), 1),
                "pred_mean": round(float(np.average(p[m], weights=w[m])), 1),
                "mae": round(float(np.average(np.abs(p[m] - t[m]), weights=w[m])), 1),
            }
    return out


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = ROOT / "outputs/pitch"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold in FOLDS:
        split = json.loads((ROOT / "outputs/stage4" / f"cv_{fold}" / "split.json").read_text())
        models = [load(f"cv_{fold}{s}", device) for s in ("", "_s1", "_s2")]
        for chip in kept_chips(ROOT, split["test"]):
            rows += facets_of(chip, models, device, "Bilthoven")
        print(f"Bilthoven {fold}: {sum(r['town'] == 'Bilthoven' for r in rows)} facets so far", flush=True)
    models = [load(f"cv_{f}{s}", device) for f in FOLDS for s in ("", "_s1", "_s2")]
    for chip in kept_chips(ROOT, HEEMSTEDE):
        rows += facets_of(chip, models, device, "Heemstede")
    with (out_dir / "facets.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    bilt = [r for r in rows if r["town"] == "Bilthoven"]
    constant = weighted_median(np.array([r["true_pitch"] for r in bilt]), np.array([r["area_m2"] for r in bilt]))
    result = {"constant_baseline_deg": round(constant, 1)}
    for town in ("Bilthoven", "Heemstede"):
        result[town] = summary([r for r in rows if r["town"] == town], constant)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    fig, axs = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True, sharey=True)
    for ax, town in zip(axs, ("Bilthoven", "Heemstede")):
        sel = [r for r in rows if r["town"] == town]
        t = np.array([r["true_pitch"] for r in sel])
        p = np.array([r["pred_pitch"] for r in sel])
        ax.scatter(t, p, s=6, alpha=0.25, color="#2a78d6", label="facets")
        b = result[town]["by_true_pitch"]
        ax.plot(
            [v["true_mean"] for v in b.values()],
            [v["pred_mean"] for v in b.values()],
            "o-",
            color="#d6572a",
            label="mean per band of true pitch",
        )
        ax.plot([0, 70], [0, 70], color="#555", lw=0.8, label="perfect")
        ax.axhline(constant, color="#999", lw=0.8, ls="--", label=f"constant baseline {constant:.0f}°")
        ax.set_xlim(0, 70)
        ax.set_ylim(0, 70)
        ax.set_aspect("equal")
        ax.set_xlabel("LiDAR pitch, degrees")
        ax.set_title(f"{town}: {len(sel)} facets, MAE {result[town]['model_mae_deg']}°", fontsize=10)
    axs[0].set_ylabel("predicted pitch (mean over facet), degrees")
    axs[0].legend(fontsize=7, loc="upper left")
    fig.savefig(out_dir / "pitch.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
