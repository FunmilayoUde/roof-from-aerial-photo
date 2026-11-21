"""Stage 2 summary across tiles: what was kept and why, how well labels register, and which
correction model fits better, pooled over every house.

    uv run python scripts/stage2_summary.py 32CN1_03 32CN1_04 ...      # tile names under outputs/stage2/

Writes outputs/stage2/summary_all.json and figures in outputs/stage2/figures/.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def num(r: dict, k: str) -> float | None:
    v = r.get(k, "")
    return None if v in ("", None) else float(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tiles", nargs="+")
    args = ap.parse_args()
    out = ROOT / "outputs" / "stage2"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    rows, by_height = [], {}
    for t in args.tiles:
        with (out / t / "manifest.csv").open() as fh:
            for r in csv.DictReader(fh):
                r["tile"] = t
                rep = ROOT / "data" / "processed" / "stage2" / t / r["house_id"] / "report.json"
                if rep.exists():
                    off = json.loads(rep.read_text())["source"]["offset_xyz"]
                    r["x"], r["y"] = off[0], off[1]
                rows.append(r)
        s = json.loads((out / t / "summary.json").read_text())
        for b, v in s.get("residual_by_height_all_registered_pieces", {}).items():
            acc = by_height.setdefault(b, {"pieces": 0, "sq_c": 0.0, "sq_r": 0.0})
            acc["pieces"] += v["pieces"]
            acc["sq_c"] += v["pieces"] * v["rms_constant_m"] ** 2
            acc["sq_r"] += v["pieces"] * v["rms_pure_relief_m"] ** 2

    status = Counter(r["status"] for r in rows)
    failed = Counter(f for r in rows for f in r["failed"].split(";") if f)
    only = Counter(r["failed"] for r in rows if r["failed"] and ";" not in r["failed"])
    kept = [r for r in rows if r["status"] == "kept"]
    reliable = [
        r
        for r in rows
        if r.get("relief_labels_reliable") == "True"
        and "repeat_diff_constant_m" not in r["failed"]
        and num(r, "rms_constant_m") is not None
        and num(r, "rms_pure_relief_m") is not None
    ]
    better = sum(num(r, "rms_pure_relief_m") < num(r, "rms_constant_m") for r in reliable)
    summary = {
        "tiles": args.tiles,
        "candidates": len(rows),
        "status": dict(status.most_common()),
        "failed_checks_any": dict(failed.most_common()),
        "failed_check_alone": dict(only.most_common()),
        "kept": {
            "houses": len(kept),
            "roof_area_m2": round(sum(num(r, "roof_area_m2") for r in kept), 1),
            "facets": int(sum(num(r, "facets") for r in kept)),
            "median_concavity": round(float(np.median([num(r, "concavity") for r in kept])), 3) if kept else None,
        },
        "registration_kept_houses": {
            k: {
                "p10": round(float(np.percentile(v, 10)), 3),
                "median": round(float(np.median(v)), 3),
                "p90": round(float(np.percentile(v, 90)), 3),
            }
            for k, v in {
                "shift_size_m": [np.hypot(num(r, "shift_e_m"), num(r, "shift_n_m")) for r in kept],
                "rms_constant_m": [num(r, "rms_constant_m") for r in kept],
                "rms_pure_relief_m": [num(r, "rms_pure_relief_m") for r in kept],
                "repeat_diff_constant_m": [num(r, "repeat_diff_constant_m") for r in kept],
            }.items()
            if kept
        },
        "model_comparison_houses_where_both_repeat": {
            "houses": len(reliable),
            "pure_relief_better": better,
            "constant_better": len(reliable) - better,
        },
        "residual_by_height_pooled": {
            b: {
                "pieces": v["pieces"],
                "rms_constant_m": round((v["sq_c"] / v["pieces"]) ** 0.5, 3),
                "rms_pure_relief_m": round((v["sq_r"] / v["pieces"]) ** 0.5, 3),
            }
            for b, v in sorted(by_height.items(), key=lambda kv: float(kv[0].split("-")[0]))
        },
    }
    (out / "summary_all.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    plot_shift_map(rows, figs / "shift_map.png")
    plot_distributions(kept, figs / "distributions.png")


def plot_shift_map(rows: list[dict], path: Path) -> None:
    """Each registered house as an arrow: where its roof appears in the photo relative to LiDAR."""
    reg = [r for r in rows if num(r, "shift_e_m") is not None and "x" in r]
    if not reg:
        return
    x = np.array([r["x"] for r in reg]) / 1000
    y = np.array([r["y"] for r in reg]) / 1000
    u = np.array([num(r, "shift_e_m") for r in reg])
    v = np.array([num(r, "shift_n_m") for r in reg])
    kept = np.array([r["status"] == "kept" for r in reg])
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    for m, color, label in ((~kept, "#bbb", "dropped"), (kept, "#2a78d6", "kept")):
        ax.quiver(x[m], y[m], u[m], v[m], color=color, angles="xy", scale_units="xy", scale=4, width=0.003, label=label)
    ax.quiver([x.min()], [y.min() - 0.15], [1.0], [0.0], angles="xy", scale_units="xy", scale=4, width=0.003)
    ax.text(x.min() + 0.3, y.min() - 0.15, "1 m shift", va="center", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("RD x, km")
    ax.set_ylabel("RD y, km")
    ax.set_title("Constant shift per house (photo minus LiDAR), arrows x 250", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_distributions(kept: list[dict], path: Path) -> None:
    if not kept:
        return
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    shift = [np.hypot(num(r, "shift_e_m"), num(r, "shift_n_m")) for r in kept]
    axs[0].hist(shift, bins=30, color="#2a78d6")
    axs[0].set_xlabel("shift size, m")
    bins = np.linspace(0, 0.3, 31)
    axs[1].hist([num(r, "rms_constant_m") for r in kept], bins=bins, alpha=0.6, label="constant", color="#555")
    axs[1].hist([num(r, "rms_pure_relief_m") for r in kept], bins=bins, alpha=0.6, label="pure relief", color="#d6572a")
    axs[1].set_xlabel("edge residual after correction, m")
    axs[1].legend(fontsize=8)
    axs[2].hist([num(r, "repeat_diff_constant_m") for r in kept], bins=np.linspace(0, 0.15, 31), color="#2a78d6")
    axs[2].set_xlabel("copy-to-copy gap, constant shift, m")
    for ax in axs:
        ax.set_ylabel("houses")
    fig.suptitle(f"Kept houses: {len(kept)}", fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
