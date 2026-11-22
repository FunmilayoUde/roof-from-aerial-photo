"""Stage 5, step 1: why is orientation weak? Diagnose the Stage 4 models without training anything.

    uv run python scripts/stage5_orientation_diagnosis.py cv_west cv_centre cv_east

For each fold's model, on its test, validation and training houses, over labelled roof pixels
(reliable facets only):
  - accuracy of the orientation output (flat + 8 directions), restricted to roof pixels;
  - confusion matrix, true class against predicted;
  - for sloped pixels predicted sloped: how many compass steps off (0 = right, 1 = neighbouring
    octant, 4 = opposite);
  - accuracy by facet pitch.
Low accuracy on training houses too would mean the model never learned orientation (a setup or
learning problem); high on training and low on test would mean it doesn't generalise.
Writes outputs/stage5/orientation_diagnosis.json and confusion.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from roofphoto.models import unet  # noqa: E402
from roofphoto.models.dataset import IGNORE, N_CLASSES, kept_chips, load_targets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["bg", "flat", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]
PITCH_BINS = [0, 5, 20, 35, 50, 90]


def diagnose(model, chips, device) -> dict:
    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    steps = np.zeros(5, dtype=np.int64)
    by_pitch = {f"{a}-{b}": [0, 0] for a, b in zip(PITCH_BINS[:-1], PITCH_BINS[1:])}
    for chip in chips:
        image, orient, pitch, _ = load_targets(chip)
        with torch.no_grad():
            logits, _, _ = unet.split(model(torch.from_numpy(image)[None].to(device)))
        pred = logits[0, 1:].argmax(0).cpu().numpy() + 1  # orientation within the roof: 1-9
        m = (orient >= 1) & (orient != IGNORE)
        t, p = orient[m].astype(int), pred[m]
        np.add.at(conf, (t, p), 1)
        both = (t >= 2) & (p >= 2)
        d = np.abs((t[both] - 2) - (p[both] - 2)) % 8
        steps += np.bincount(np.minimum(d, 8 - d), minlength=5)
        pp = pitch[m]
        for a, b in zip(PITCH_BINS[:-1], PITCH_BINS[1:]):
            k = (pp >= a) & (pp < b)
            by_pitch[f"{a}-{b}"][0] += int((t[k] == p[k]).sum())
            by_pitch[f"{a}-{b}"][1] += int(k.sum())
    roof = conf[1:, 1:]
    return {
        "pixels": int(roof.sum()),
        "accuracy": round(float(np.trace(roof) / max(roof.sum(), 1)), 3),
        "flat_vs_sloped_accuracy": round(float((roof[0, 0] + roof[1:, 1:].sum()) / max(roof.sum(), 1)), 3),
        "sloped_steps_off": {str(i): round(float(steps[i] / max(steps.sum(), 1)), 3) for i in range(5)},
        "accuracy_by_pitch_deg": {k: (round(c / n, 3) if n else None) for k, (c, n) in by_pitch.items()},
        "confusion": conf[1:, 1:].tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--train-sample", type=int, default=60, help="training houses scored per fold")
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = ROOT / "outputs" / "stage5"
    out.mkdir(parents=True, exist_ok=True)
    result, pooled = {}, {s: np.zeros((N_CLASSES - 1, N_CLASSES - 1), dtype=np.int64) for s in ("test", "train")}
    for run in args.runs:
        folder = ROOT / "outputs" / "stage4" / run
        split = json.loads((folder / "split.json").read_text())
        model = unet.build(pretrained=False)
        model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
        model.to(device).eval()
        rng = np.random.default_rng(0)
        train = kept_chips(ROOT, split["train"])
        train = [train[i] for i in rng.choice(len(train), min(args.train_sample, len(train)), replace=False)]
        result[run] = {
            "test": diagnose(model, kept_chips(ROOT, split["test"]), device),
            "val": diagnose(model, kept_chips(ROOT, split["val"]), device),
            "train_sample": diagnose(model, train, device),
        }
        pooled["test"] += np.array(result[run]["test"]["confusion"])
        pooled["train"] += np.array(result[run]["train_sample"]["confusion"])
        for k in ("test", "val", "train_sample"):
            r = result[run][k]
            print(
                f"{run} {k:12s} accuracy {r['accuracy']:.3f}  flat/sloped {r['flat_vs_sloped_accuracy']:.3f}  "
                f"steps off {r['sloped_steps_off']}"
            )
    (out / "orientation_diagnosis.json").write_text(json.dumps(result, indent=2))

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    for ax, (name, c) in zip(axs, pooled.items()):
        rows = c / np.maximum(c.sum(1, keepdims=True), 1)
        ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(rows)):
            for j in range(len(rows)):
                if rows[i, j] >= 0.05:
                    ax.text(
                        j,
                        i,
                        f"{rows[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if rows[i, j] > 0.5 else "black",
                    )
        ax.set_xticks(range(9), NAMES[1:])
        ax.set_yticks(range(9), NAMES[1:])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name} houses: share of each true class (rows sum to 1)", fontsize=9)
    fig.savefig(out / "confusion.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
