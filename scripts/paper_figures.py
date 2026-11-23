"""Figures for docs/paper: every panel drawn from the saved outputs of the earlier steps.

    uv run python scripts/paper_figures.py            # all figures
    uv run python scripts/paper_figures.py --only 6 9 # some of them

Reads outputs/ and data/processed/ (nothing is retrained). Figures that need predictions run the
saved models on MPS: the orientation confusion over all 351 Bilthoven test houses, the example
predictions, and pitch per facet for the two-town models. Writes docs/paper/figures/figN.png.

No house ids or map coordinates appear in the figures: positions on maps are in km from the
south-west corner of the area shown.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from scipy.ndimage import label  # noqa: E402

from roofphoto.data.chips import EDGE_CLASSES  # noqa: E402
from roofphoto.data.lookup import footprints  # noqa: E402
from roofphoto.edges import edge_raster, facets_from_orientation  # noqa: E402
from roofphoto.io.lidar import labels_from_report  # noqa: E402
from roofphoto.models import unet  # noqa: E402
from roofphoto.models.dataset import IGNORE, IMAGENET_MEAN, IMAGENET_STD, N_CLASSES, kept_chips, load_targets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/paper/figures"
S4 = ROOT / "outputs/stage4"
FOLDS = {
    "west": ["32CN1_03", "32CN1_08", "32CN1_13"],
    "centre": ["32CN1_04", "32CN1_09"],
    "east": ["32CN1_05", "32CN1_10", "32CN1_15", "32CN1_20"],
}
HEEM_FOLDS = {"west": ["25CN1_01", "25CN1_06"], "centre": ["25CN1_02", "25CN1_07"], "east": ["25CN1_13"]}
BILT_TILES = [t for ts in FOLDS.values() for t in ts]
HEEM_TILES = [t for ts in HEEM_FOLDS.values() for t in ts]

MODEL_C, BASE_C, ALT_C, GREY = "#2563b0", "#d97a1e", "#2a9d6f", "#8a8a8a"
ORIENT_NAMES = ["flat", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]
CREDIT = "Photo: Beeldmateriaal Nederland / PDOK (CC-BY 4.0)"

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def orient_cmap():
    """Flat grey, then the eight downhill directions around a cyclic hue wheel."""
    hues = plt.get_cmap("hsv")(np.linspace(0, 1, 9)[:8])
    cols = np.vstack([[0.62, 0.62, 0.62, 1.0], hues])
    return matplotlib.colors.ListedColormap(cols)


def edge_cmap():
    return plt.get_cmap("tab10", len(EDGE_CLASSES) + 1)


def rgb_of(image: np.ndarray) -> np.ndarray:
    return np.clip(np.moveaxis(image * IMAGENET_STD + IMAGENET_MEAN, 0, -1), 0, 1)


def save(fig, n: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"fig{n}.png")
    plt.close(fig)
    print(f"fig{n}.png", flush=True)


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(run: str, dev, edges: bool = False):
    m = unet.build(pretrained=False, edges=edges)
    m.load_state_dict(torch.load(S4 / run / "model.pt", map_location="cpu"))
    return m.to(dev).eval()


def per_house(run: str, name: str = "per_house.csv") -> list[dict]:
    return list(csv.DictReader(open(S4 / run / name)))


def manifest_rows(tiles: list[str]) -> list[dict]:
    rows = []
    for t in tiles:
        rows += list(csv.DictReader(open(ROOT / "outputs/stage2" / t / "manifest.csv")))
    return rows


# --------------------------------------------------------------------------------------- data


def bilthoven_area() -> dict[str, np.ndarray]:
    """Per house, the Stage 4 numbers: mean of the fold's three seeds, x the cross-fold factor."""
    cal = json.loads((S4 / "cv_summary.json").read_text())["calibration"]
    truth, model, b1, flat, unc, fold_of = [], [], [], [], [], []
    for fold in FOLDS:
        runs = [f"cv_{fold}", f"cv_{fold}_s1", f"cv_{fold}_s2"]
        tables = [{r["house_id"]: r for r in per_house(run)} for run in runs]
        for hid, r in tables[0].items():
            truth.append(float(r["truth_m2"]))
            model.append(
                np.mean([float(t[hid]["model_pitch_m2"]) for t in tables]) * cal[f"cv_{fold}"]["cross_fold_factor"]
            )
            b1.append(float(r["b1_m2"]))
            flat.append(float(r["flat_share"]))
            unc.append(float(r["uncertain_share"]))
            fold_of.append(fold)
    return {k: np.array(v) for k, v in dict(truth=truth, model=model, b1=b1, flat=flat, unc=unc, fold=fold_of).items()}


def two_town_area(town: str) -> dict[str, np.ndarray]:
    """Two-town (bt_) and Bilthoven-only seed-0 (cv_) models on one town, calibrated as in
    scripts/two_town_compare.py (factor from the other folds' houses)."""
    tag_bt, tag_one = ("bilt", "") if town == "bilthoven" else ("heem", "heem")

    def name(tag):
        return "per_house.csv" if not tag else f"per_house_{tag}.csv"

    out = {}
    for name_, prefix, tag in (("two_town", "bt_", tag_bt), ("bilthoven_only", "cv_", tag_one)):
        per_fold = {f: per_house(f"{prefix}{f}", name(tag)) for f in FOLDS}
        cal = []
        for f, rs in per_fold.items():
            others = [r for g, rr in per_fold.items() if g != f for r in rr if float(r["model_pitch_m2"]) > 0]
            c = np.median([float(r["truth_m2"]) / float(r["model_pitch_m2"]) for r in others])
            cal += [float(r["model_pitch_m2"]) * c for r in rs]
        rs = [r for f in FOLDS for r in per_fold[f]]
        out[name_] = np.array(cal)
        out[f"{name_}_b1"] = np.array([float(r["b1_m2"]) for r in rs])
        out["truth"] = np.array([float(r["truth_m2"]) for r in rs])
    return out


def abs_err(pred, truth):
    return 100 * np.abs(pred / truth - 1)


def boot_median(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = np.median(x[idx], axis=1)
    return np.percentile(m, [2.5, 97.5])


# ------------------------------------------------------------------------------------ figures


def fig1_overview() -> None:
    """Method overview: the LiDAR teacher, registration, label rasters, the student, evaluation."""
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec="#333", lw=0.8, alpha=0.95
            )
        )
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.5, weight="bold")
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=7.2, linespacing=1.35)

    def arrow(x0, y0, x1, y1, text=None, dy=1.2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=0.9, color="#333"))
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontsize=6.8, style="italic")

    lid, pho, lab, stu = "#e8eef8", "#f6ecdf", "#eaf4ec", "#efe8f6"
    box(1, 24, 17, 14, "AHN5 LiDAR", "~15 pts/m², 2023\nbuilding + ground\nclasses", lid)
    box(1, 3, 17, 14, "BAG outlines", "candidate houses:\nresidential, 70–300 m²,\n≥ 3 m from others", lid)
    box(
        23,
        24,
        19,
        14,
        "Teacher: roofrecon",
        "RANSAC planes, topology,\nalpha shapes; radii scaled\nto point spacing (§4.2)",
        lid,
    )
    box(23, 3, 19, 14, "PDOK orthophoto", "8 cm RGB, 2023,\nJPEG via WMS;\ntwo renderings / house", pho)
    box(47, 13.5, 19, 14, "Registration", "edge pieces vs image\ngradient; relief model\nshift = h · v (§4.3–4.5)", pho)
    box(71, 24, 13, 14, "Labels", "facets, edges,\nroof extent;\nignore = 255", lab)
    box(71, 3, 13, 14, "Quality gates", "coverage, residual,\nrepeatability ...\n351 of 1,010 kept", lab)
    box(87.5, 13.5, 12, 14, "Student", "U-Net, ResNet-34\nroof · orientation\n· pitch", stu)

    arrow(18.5, 31, 22.5, 31)
    arrow(18.5, 10, 22.5, 10)
    arrow(9.5, 17.5, 22.5, 25.5)
    arrow(42.5, 31, 46.5, 24)
    arrow(42.5, 10, 46.5, 17)
    ax.text(49, 31.5, "3-D facets + edges", fontsize=6.8, style="italic")
    ax.text(49, 8.2, "photo chip", fontsize=6.8, style="italic")
    arrow(66.5, 22, 70.5, 29)
    arrow(66.5, 19, 70.5, 12)
    arrow(84.5, 31, 87, 24)
    arrow(84.5, 10, 87, 17)
    ax.text(
        50,
        0.2,
        "Evaluation: spatial 3-fold CV (columns of 1 × 1.25 km sub-tiles), 3 seeds, "
        "paired bootstrap against footprint × k; transfer to a second town.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#444",
    )
    save(fig, 1)


def fig2_relief() -> None:
    """(a) relief displacement geometry, (b) shift field over Bilthoven, (c) ground check."""
    fig = plt.figure(figsize=(10, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.9], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1.2, 11)
    ax.axis("off")
    ax.plot([-1, 11], [0, 0], color="#555", lw=1)
    ax.plot([1, 1], [0, 10], color="#999", lw=0.7, ls=":")
    ax.plot(1, 10, "v", color="#333", ms=8)
    ax.text(1.3, 10.2, "camera", fontsize=7.5)
    ax.text(1.0, -0.8, "nadir", ha="center", fontsize=7.5)
    ax.add_patch(plt.Rectangle((6, 0), 1.2, 3.2, fc="#c9b28f", ec="#555", lw=0.8))
    ax.plot([1, 7.2], [10, 3.2], color=MODEL_C, lw=0.9)
    ax.plot([7.2, 7.2 + (7.2 - 1) * 3.2 / (10 - 3.2)], [3.2, 0], color=MODEL_C, lw=0.9, ls="--")
    xr = 7.2 + (7.2 - 1) * 3.2 / (10 - 3.2)
    ax.annotate("", (xr, -0.35), (7.2, -0.35), arrowprops=dict(arrowstyle="<->", lw=0.8, color="#b03030"))
    ax.text((7.2 + xr) / 2, -1.15, "d", color="#b03030", ha="center", fontsize=8, style="italic")
    ax.text(7.5, 1.4, "h", fontsize=8, style="italic")
    ax.text(-0.6, 5, "H", fontsize=8, style="italic")
    ax.text(3.4, 2.0, "r", fontsize=8, style="italic")
    ax.annotate("", (1, 1.6), (7.2, 1.6), arrowprops=dict(arrowstyle="<->", lw=0.6, color="#555"))
    ax.text(5.2, 8.0, r"$d = h\,r\,/\,H$", fontsize=9)
    ax.set_title("(a) relief displacement", loc="left")

    ax = fig.add_subplot(gs[1])
    xs, ys, us, vs, kept = [], [], [], [], []
    for t in BILT_TILES:
        fps = footprints(ROOT, t)
        for r in csv.DictReader(open(ROOT / "outputs/stage2" / t / "manifest.csv")):
            if not r.get("shift_e_m") or r["house_id"] not in fps:
                continue
            c = fps[r["house_id"]].centroid
            xs.append(c.x)
            ys.append(c.y)
            us.append(float(r["shift_e_m"]))
            vs.append(float(r["shift_n_m"]))
            kept.append(r["status"] == "kept")
    xs, ys = (np.array(xs) - min(xs)) / 1000, (np.array(ys) - min(ys)) / 1000
    us, vs, kept = np.array(us), np.array(vs), np.array(kept)
    for sel, col, lab in ((~kept, "#bbbbbb", "dropped"), (kept, MODEL_C, "kept")):
        ax.quiver(
            xs[sel],
            ys[sel],
            us[sel],
            vs[sel],
            color=col,
            scale=2.5,
            scale_units="xy",
            angles="xy",
            width=0.0035,
            label=lab,
        )
    ax.quiver([0.15], [0.1], [1.0], [0], scale=2.5, scale_units="xy", angles="xy", width=0.0035, color="k")
    ax.text(0.15, 0.22, "1 m", fontsize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("km east")
    ax.set_ylabel("km north")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(f"(b) photo − LiDAR shift, {len(xs)} houses (×400)", loc="left")

    ax = fig.add_subplot(gs[2])
    g = json.loads((ROOT / "outputs/stage1/ground_check_summary.json").read_text())
    names = sorted(k for k in g if k.startswith("delft"))
    yy = np.arange(len(names))
    ax.barh(yy + 0.18, [g[n]["roof_shift_size_m"] for n in names], 0.36, color=BASE_C, label="roof edges")
    ax.barh(yy - 0.18, [g[n]["ground_shift_size_m"] for n in names], 0.36, color=MODEL_C, label="ground features")
    for i, n in enumerate(names):
        w = g[n]["zero_vs_roof_shift"]["no_shift_wins_fraction"]
        ax.text(g[n]["roof_shift_size_m"] + 0.03, i + 0.18, f"no-shift wins {w:.0%}", va="center", fontsize=6.8)
    ax.set_yticks(yy, [f"house {i + 1}" for i in range(len(names))])
    ax.set_xlabel("best-fit shift, m")
    ax.set_xlim(0, 1.75)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("(c) ground check (Delft)", loc="left")
    save(fig, 2)


def fig3_registration() -> None:
    """(a, b) LiDAR edges on the photo before and after registration, (c) residual by height."""
    m = {r["house_id"]: r for r in manifest_rows(["32CN1_08", "32CN1_09"])}
    good = [
        c
        for c in kept_chips(ROOT, ["32CN1_08", "32CN1_09"])
        if c.labels == "relief"
        and float(m[c.house_id]["constant_shift_m"]) > 0.7
        and float(m[c.house_id]["rms_pure_relief_m"]) < 0.8 * float(m[c.house_id]["rms_constant_m"])
    ]
    chip = max(good, key=lambda c: float(m[c.house_id]["constant_shift_m"]))
    reg = json.loads((chip.folder / "registration.json").read_text())
    rep = json.loads((chip.folder / "report.json").read_text())
    fps = footprints(ROOT, chip.tile)
    lab = labels_from_report(rep, fps[chip.house_id])
    with rasterio.open(chip.folder / "photo.tif") as src:
        rgb = np.moveaxis(src.read(), 0, -1)
        inv = ~src.transform
    v = np.array(reg["label_shift"]["relief_v"])
    gz = reg["ground_z_nap_m"]

    pts = np.array([e.p0[:2] for e in lab.edges] + [e.p1[:2] for e in lab.edges])
    cx, cy = inv * tuple(pts.mean(0))
    half = 150

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, shifted, title in (
        (axs[0], False, "(a) LiDAR edges, no shift"),
        (axs[1], True, "(b) after registration (relief)"),
    ):
        ax.imshow(rgb)
        for e in lab.edges:
            p = []
            for q in (e.p0, e.p1):
                s = v * (q[2] - gz) if shifted else np.zeros(2)
                p.append(inv * (q[0] + s[0], q[1] + s[1]))
            p = np.array(p)
            ax.plot(p[:, 0], p[:, 1], color="#ff3b3b" if not shifted else "#22e0ff", lw=1.3)
            if shifted:
                q = np.array([inv * tuple(e.p0[:2]), inv * tuple(e.p1[:2])])
                ax.plot(q[:, 0], q[:, 1], color="#ff3b3b", lw=0.6, alpha=0.6)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy + half, cy - half)
        ax.set_axis_off()
        ax.set_title(title, loc="left")
    axs[0].plot([cx - half + 15, cx - half + 15 + 62.5], [cy + half - 15] * 2, color="w", lw=2)
    axs[0].text(cx - half + 15, cy + half - 22, "5 m", color="w", fontsize=7)
    a = reg["rendering_a"]
    axs[1].text(
        0.02,
        0.02,
        f"rms: constant {a['constant_model']['rms_m']:.2f} m, relief {a['pure_relief_model']['rms_m']:.2f} m",
        transform=axs[1].transAxes,
        color="w",
        fontsize=7,
        bbox=dict(fc="k", alpha=0.45, lw=0),
    )

    ax = axs[2]
    s = json.loads((ROOT / "outputs/stage2/summary_all.json").read_text())["residual_by_height_pooled"]
    keys = list(s)
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [s[k]["rms_constant_m"] for k in keys], 0.4, color=GREY, label="constant shift")
    ax.bar(x + 0.2, [s[k]["rms_pure_relief_m"] for k in keys], 0.4, color=MODEL_C, label="relief h·v")
    ax.set_xticks(x, [k.replace("8-99 m", "> 8 m") for k in keys])
    for i, k in enumerate(keys):
        ax.text(i, 0.005, f"{s[k]['pieces'] // 1000}k", ha="center", fontsize=6.5, color="w")
    ax.set_xlabel("height of edge piece above ground")
    ax.set_ylabel("rms residual, m")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("(c) residual by height, all registered houses", loc="left")
    fig.tight_layout()
    fig.text(0.01, -0.02, CREDIT, fontsize=6.5, color="#666")
    save(fig, 3)


def fig4_labels() -> None:
    """One training chip and its three label bands."""
    chip = [c for c in kept_chips(ROOT, ["32CN1_09"])][7]
    image, orient, pitch, roof = load_targets(chip)
    with rasterio.open(chip.folder / f"labels_{chip.labels}.tif") as src:
        edges_b, roof_b = src.read(2), src.read(3)
    rgb = rgb_of(image)
    fig, axs = plt.subplots(1, 4, figsize=(10, 2.85))
    axs[0].imshow(rgb)
    axs[0].set_title("(a) photo chip, 512 px = 41 m", loc="left")

    axs[1].imshow(rgb * 0.55)
    o = np.ma.masked_where((orient == 0) | (orient == IGNORE), orient.astype(float) - 1)
    axs[1].imshow(o, cmap=orient_cmap(), vmin=-0.5, vmax=8.5, interpolation="nearest")
    ign = np.ma.masked_where(orient != IGNORE, np.ones_like(orient, float))
    axs[1].imshow(ign, cmap=matplotlib.colors.ListedColormap(["#ffffff"]), alpha=0.35, interpolation="nearest")
    axs[1].set_title("(b) facets: flat + 8 directions", loc="left")
    handles = [Patch(color=orient_cmap()(i), label=n) for i, n in enumerate(ORIENT_NAMES)]
    axs[1].legend(
        handles=handles, ncol=3, fontsize=5.6, loc="lower left", frameon=True, handlelength=1, columnspacing=0.6
    )

    axs[2].imshow(rgb * 0.55)
    e = np.ma.masked_where((edges_b == 0) | (edges_b == IGNORE), edges_b)
    axs[2].imshow(e, cmap=edge_cmap(), vmin=0, vmax=len(EDGE_CLASSES), interpolation="nearest")
    hs = [
        Patch(color=edge_cmap()(v), label=k)
        for k, v in EDGE_CLASSES.items()
        if k != "unclassified" and (edges_b == v).any()
    ]
    axs[2].legend(handles=hs, fontsize=5.6, loc="lower left", handlelength=1)
    axs[2].set_title("(c) typed edges, 3 px wide", loc="left")

    axs[3].imshow(rgb * 0.55)
    axs[3].imshow(np.ma.masked_where(roof_b != 1, roof_b), cmap=matplotlib.colors.ListedColormap([MODEL_C]), alpha=0.75)
    axs[3].imshow(
        np.ma.masked_where(roof_b != IGNORE, roof_b), cmap=matplotlib.colors.ListedColormap(["#ffffff"]), alpha=0.45
    )
    axs[3].legend(
        handles=[Patch(color=MODEL_C, label="roof"), Patch(color="#dddddd", label="ignore (255)")],
        fontsize=6,
        loc="lower left",
    )
    axs[3].set_title("(d) roof extent", loc="left")
    for ax in axs:
        ax.set_axis_off()
    fig.tight_layout()
    fig.text(0.01, 0.0, CREDIT, fontsize=6.5, color="#666")
    save(fig, 4)


def fig5_dataset() -> None:
    """(a) why candidates were dropped, (b) roof/footprint ratio, (c) pitch distribution by town."""
    s = json.loads((ROOT / "outputs/stage2/summary_all.json").read_text())
    names = {
        "unlabelled_interior_share": "unlabelled points inside walls > 5%",
        "footprint_check_abs": "footprint consistency > 10%",
        "coverage": "plane coverage < 0.90",
        "uncertain_area_share": "small/traced facets > 50%",
        "best_model_rms_m": "edge residual > 0.25 m",
        "repeat_diff_constant_m": "shift differs across renderings",
        "chip_size": "house larger than chip",
        "pieces_along_axis": "too few edge pieces along axis",
    }
    fa, fo = s["failed_checks_any"], s["failed_check_alone"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axs[0]
    keys = list(names)[::-1]
    y = np.arange(len(keys))
    ax.barh(y, [fa.get(k, 0) for k in keys], color="#c9d6ea", label="fails this check")
    ax.barh(y, [fo.get(k, 0) for k in keys], color=MODEL_C, label="fails only this check")
    ax.set_yticks(y, [names[k] for k in keys], fontsize=7)
    ax.set_xlabel("houses (of 1,010 Bilthoven candidates)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"(a) quality gates: {s['kept']['houses']} kept", loc="left")

    ax = axs[1]
    rows = [r for r in manifest_rows(BILT_TILES) if r["status"] == "kept"]
    ratio = np.array([float(r["roof_area_m2"]) / float(r["footprint_m2"]) for r in rows])
    ax.hist(ratio, bins=np.arange(0.9, 2.4, 0.05), color=GREY)
    ax.axvline(np.median(ratio), color=BASE_C, lw=1.2)
    ax.text(np.median(ratio) + 0.03, ax.get_ylim()[1] * 0.9, f"median {np.median(ratio):.2f}", fontsize=7, color=BASE_C)
    ax.set_xlabel("LiDAR roof area / BAG footprint")
    ax.set_ylabel("houses")
    ax.set_title("(b) what one factor must cover", loc="left")

    ax = axs[2]
    f = list(csv.DictReader(open(ROOT / "outputs/pitch/facets.csv")))
    for town, col in (("Bilthoven", MODEL_C), ("Heemstede", BASE_C)):
        t = np.array([float(r["true_pitch"]) for r in f if r["town"] == town])
        w = np.array([float(r["area_m2"]) for r in f if r["town"] == town])
        ax.hist(t, bins=np.arange(0, 66, 3), weights=w / w.sum(), histtype="step", lw=1.4, color=col, label=town)
    ax.set_xlabel("facet pitch (LiDAR), degrees")
    ax.set_ylabel("share of roof area")
    ax.legend(frameon=False)
    ax.set_title("(c) pitch, reliable facets", loc="left")
    fig.tight_layout()
    save(fig, 5)


def fig6_area() -> None:
    """(a) predicted vs LiDAR area, (b) error CDF, (c) error by roof type and size."""
    d = bilthoven_area()
    em, eb = abs_err(d["model"], d["truth"]), abs_err(d["b1"], d["truth"])
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    ax = axs[0]
    ax.scatter(d["truth"], d["b1"], s=7, color=BASE_C, alpha=0.5, lw=0, label=f"footprint × k ({np.median(eb):.1f}%)")
    ax.scatter(d["truth"], d["model"], s=7, color=MODEL_C, alpha=0.7, lw=0, label=f"photo model ({np.median(em):.1f}%)")
    lim = [60, 460]
    ax.plot(lim, lim, color="#333", lw=0.7)
    ax.fill_between(lim, np.array(lim) * 0.9, np.array(lim) * 1.1, color="#000", alpha=0.06, lw=0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("LiDAR roof area, m²")
    ax.set_ylabel("predicted, m²")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    ax.set_title("(a) 351 held-out houses (band ±10%)", loc="left")

    ax = axs[1]
    for e, col, lab in ((em, MODEL_C, "photo model"), (eb, BASE_C, "footprint × k")):
        xs = np.sort(e)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=col, lw=1.4, label=lab)
    ax.axvline(10, color="#999", lw=0.6, ls=":")
    ax.set_xlim(0, 45)
    ax.set_xlabel("|area error|, %")
    ax.set_ylabel("share of houses")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("(b) cumulative error", loc="left")

    ax = axs[2]
    small = d["truth"] < 150
    groups = [
        ("all", np.ones_like(small)),
        ("no flat\npart", d["flat"] < 0.05),
        ("mixed", (d["flat"] >= 0.05) & (d["flat"] < 0.5)),
        ("mostly\nflat", d["flat"] >= 0.5),
        ("< 150\nm²", small),
        ("≥ 150\nm²", ~small),
    ]
    x = np.arange(len(groups))
    for off, e, col in ((-0.2, em, MODEL_C), (0.2, eb, BASE_C)):
        meds = [np.median(e[g]) for _, g in groups]
        ci = np.array([boot_median(e[g]) for _, g in groups]).T
        ax.bar(x + off, meds, 0.4, color=col, label="photo model" if col == MODEL_C else "footprint × k")
        ax.errorbar(
            x + off,
            meds,
            yerr=[np.array(meds) - ci[0], ci[1] - np.array(meds)],
            fmt="none",
            color="#333",
            lw=0.7,
            capsize=1.5,
        )
    ax.set_xticks(x, [f"{n}\n({g.sum()})" for n, g in groups], fontsize=6.8)
    ax.set_ylabel("median |area error|, %")
    ax.set_title("(c) by roof type and size (95% CI)", loc="left")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, 6)


def predict(models, x):
    with torch.no_grad():
        return [unet.split(m(x)) for m in models]


def fig7_examples_and_fig8_confusion() -> None:
    """Fig 7: example held-out predictions. Fig 8: orientation confusion over all test houses."""
    dev = device()
    conf = np.zeros((N_CLASSES - 1, N_CLASSES - 1), dtype=np.int64)
    examples = []
    want = {"west": 2, "centre": 1, "east": 1}
    for fold, tiles in FOLDS.items():
        roof_models = [load_model(r, dev) for r in (f"cv_{fold}", f"cv_{fold}_s1", f"cv_{fold}_s2")]
        orient_models = [load_model(f"s5w_{fold}_geom_s{s}", dev) for s in range(3)]
        fps = {}
        for t in tiles:
            fps.update(footprints(ROOT, t))
        for i, chip in enumerate(kept_chips(ROOT, tiles)):
            image, orient_t, pitch_t, roof_t = load_targets(chip)
            x = torch.from_numpy(image)[None].to(dev)
            with torch.no_grad():
                probs = np.mean([torch.softmax(unet.split(m(x))[0][0, 1:], 0).cpu().numpy() for m in orient_models], 0)
            orient_p = probs.argmax(0)  # 0 = flat, 1..8 = N..NW
            sel = (orient_t >= 1) & (orient_t != IGNORE)
            np.add.at(conf, (orient_t[sel].astype(int) - 1, orient_p[sel]), 1)
            if want[fold] and i % 9 == 4 and (roof_t == 1).sum() > 15000:
                with torch.no_grad():
                    outs = [unet.split(m(x)) for m in roof_models]
                roof_p = np.mean([torch.sigmoid(o[2][0]).cpu().numpy() for o in outs], 0)
                pitch_p = np.mean([o[1][0].cpu().numpy() for o in outs], 0)
                with rasterio.open(chip.folder / f"labels_{chip.labels}.tif") as src:
                    t = src.transform
                seed = rasterize([(fps[chip.house_id].buffer(1.5), 1)], out_shape=roof_t.shape, transform=t) == 1
                comp, _ = label(roof_p > 0.5, structure=np.ones((3, 3)))
                house = np.isin(comp, np.unique(comp[seed & (comp > 0)])) & (comp > 0)
                facets, cls = facets_from_orientation(house, orient_p + 1, 150)
                examples.append(
                    dict(
                        rgb=rgb_of(image),
                        orient_t=orient_t,
                        pitch_t=pitch_t,
                        house=house,
                        orient_p=orient_p,
                        pitch_p=pitch_p,
                        edges=edge_raster(facets, cls),
                    )
                )
                want[fold] -= 1
        print(f"  {fold}: confusion pixels so far {conf.sum():,}", flush=True)

    # Fig 7
    n = len(examples)
    fig, axs = plt.subplots(n, 5, figsize=(10, 2.0 * n + 0.6))
    cm, pc = orient_cmap(), plt.get_cmap("viridis")
    for r, e in enumerate(examples):
        a = axs[r]
        a[0].imshow(e["rgb"])
        a[1].imshow(e["rgb"] * 0.5)
        ot = np.ma.masked_where((e["orient_t"] == 0) | (e["orient_t"] == IGNORE), e["orient_t"].astype(float) - 1)
        a[1].imshow(ot, cmap=cm, vmin=-0.5, vmax=8.5, interpolation="nearest")
        a[2].imshow(e["rgb"] * 0.5)
        a[2].imshow(
            np.ma.masked_where(~e["house"], e["orient_p"].astype(float)),
            cmap=cm,
            vmin=-0.5,
            vmax=8.5,
            interpolation="nearest",
        )
        a[3].imshow(e["rgb"] * 0.5)
        a[3].imshow(np.ma.masked_where(np.isnan(e["pitch_t"]), e["pitch_t"]), cmap=pc, vmin=0, vmax=60)
        a[4].imshow(e["rgb"] * 0.5)
        im = a[4].imshow(np.ma.masked_where(~e["house"], np.clip(e["pitch_p"], 0, 70)), cmap=pc, vmin=0, vmax=60)
        ed = np.ma.masked_where(e["edges"] == 0, e["edges"])
        a[2].imshow(ed, cmap=edge_cmap(), vmin=0, vmax=len(EDGE_CLASSES), interpolation="nearest", alpha=0.9)
        ys, xs = np.nonzero(e["house"] | ((e["orient_t"] > 0) & (e["orient_t"] != IGNORE)))
        pad = 30
        cxp, cyp = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + pad
        for ax in a:
            ax.set_xlim(cxp - half, cxp + half)
            ax.set_ylim(cyp + half, cyp - half)
            ax.set_axis_off()
    for ax, t in zip(
        axs[0],
        ["photo", "label: orientation", "predicted orientation\n+ typed edges", "label: pitch", "predicted pitch"],
    ):
        ax.set_title(t)
    fig.subplots_adjust(left=0.01, right=0.9, top=0.93, bottom=0.07, wspace=0.04, hspace=0.05)
    cb = fig.colorbar(im, cax=fig.add_axes([0.915, 0.3, 0.012, 0.4]))
    cb.set_label("pitch, °")
    handles = [Patch(color=cm(i), label=nm) for i, nm in enumerate(ORIENT_NAMES)]
    fig.legend(handles=handles, ncol=9, loc="lower center", bbox_to_anchor=(0.45, 0.0), fontsize=7, frameon=False)
    fig.text(0.01, -0.01, CREDIT, fontsize=6.5, color="#666")
    save(fig, 7)

    # Fig 8
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    norm = conf / conf.sum(1, keepdims=True)
    ax = axs[0]
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(9):
        for j in range(9):
            if norm[i, j] >= 0.03:
                ax.text(
                    j,
                    i,
                    f"{norm[i, j] * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.3,
                    color="w" if norm[i, j] > 0.5 else "#222",
                )
    ax.set_xticks(range(9), ORIENT_NAMES)
    ax.set_yticks(range(9), ORIENT_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("LiDAR label")
    acc = np.trace(conf) / conf.sum()
    ax.set_title(f"(a) orientation, {conf.sum() / 1e6:.1f} M roof px, accuracy {acc:.1%} (row %)", loc="left")
    ax = axs[1]
    sl = conf[1:, 1:]
    steps = np.zeros(5)
    for i in range(8):
        for j in range(8):
            k = abs(i - j)
            steps[min(k, 8 - k)] += sl[i, j]
    steps /= steps.sum()
    ax.bar(range(5), steps * 100, color=[MODEL_C, "#7fa3d1", BASE_C, "#e0a768", "#b03030"])
    for k, v in enumerate(steps):
        ax.text(k, v * 100 + 1, f"{v:.0%}", ha="center", fontsize=7.5)
    ax.set_xticks(range(5), ["exact", "45°", "90°", "135°", "180°"])
    ax.set_ylabel("% of sloped pixels (true and predicted sloped)")
    ax.set_title("(b) angular error of downhill direction", loc="left")
    fig.tight_layout()
    save(fig, 8)
    (ROOT / "outputs/paper").mkdir(parents=True, exist_ok=True)
    np.save(ROOT / "outputs/paper/orientation_confusion.npy", conf)


def fig9_edges() -> None:
    """Precision and recall per edge type: predictions against the rules' oracle."""
    e = json.loads((ROOT / "outputs/stage5/edges.json").read_text())["pooled"]
    types = ["any", "ridge", "eave", "rake", "hip", "valley", "break"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 2.9), sharey=True)
    for ax, metric in zip(axs, ("precision", "recall")):
        x = np.arange(len(types))
        for off, key, col, lab in (
            (-0.2, "model", MODEL_C, "rules on predictions"),
            (0.2, "oracle", GREY, "rules on LiDAR labels (oracle)"),
        ):
            vals = [e[key][t][metric] or 0 for t in types]
            ax.bar(x + off, vals, 0.4, color=col, label=lab)
        ax.set_xticks(x, types)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"({'a' if metric == 'precision' else 'b'}) {metric}, tolerance 3 px (24 cm)", loc="left")
    axs[0].set_ylabel("share of edge pixels")
    axs[0].legend(frameon=False, loc="upper right", fontsize=7)
    fig.tight_layout()
    save(fig, 9)


def fig10_pitch() -> None:
    """Pitch per facet: Bilthoven-only models in both towns, and two-town models in Heemstede."""
    dev = device()
    f = list(csv.DictReader(open(ROOT / "outputs/pitch/facets.csv")))
    two = []
    for fold, tiles in HEEM_FOLDS.items():
        m = load_model(f"bt_{fold}", dev)
        for chip in kept_chips(ROOT, tiles):
            image, _, _, _ = load_targets(chip)
            with rasterio.open(chip.folder / f"labels_{chip.labels}.tif") as src:
                band = src.read(1)
            with torch.no_grad():
                p = unet.split(m(torch.from_numpy(image)[None].to(dev)))[1][0].cpu().numpy()
            for fc in json.loads((chip.folder / "facets.json").read_text()):
                sel = band == fc["class"]
                if fc["ignored_in_labels"] or sel.sum() < 20:
                    continue
                two.append((fc["pitch_deg"], float(np.clip(p[sel], 0, 70).mean()), fc["area_m2"]))
    two = np.array(two)
    sets = [
        ("(a) Bilthoven, Bilthoven models\n(out of fold)", [r for r in f if r["town"] == "Bilthoven"], MODEL_C),
        ("(b) Heemstede, Bilthoven models\n(unseen town)", [r for r in f if r["town"] == "Heemstede"], BASE_C),
        ("(c) Heemstede, two-town models\n(out of fold)", None, ALT_C),
    ]
    bands = [0, 5, 20, 30, 40, 50, 90]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, (title, rows, col) in zip(axs, sets):
        if rows is None:
            t, p, w = two[:, 0], two[:, 1], two[:, 2]
        else:
            t = np.array([float(r["true_pitch"]) for r in rows])
            p = np.array([float(r["pred_pitch"]) for r in rows])
            w = np.array([float(r["area_m2"]) for r in rows])
        ax.scatter(t, p, s=4 + w / 6, color=col, alpha=0.25, lw=0)
        bt, bp = [], []
        for lo, hi in zip(bands[:-1], bands[1:]):
            s = (t >= lo) & (t < hi)
            if s.sum() >= 5:
                bt.append(np.average(t[s], weights=w[s]))
                bp.append(np.average(p[s], weights=w[s]))
        ax.plot(bt, bp, "o-", color="#222", ms=3.5, lw=1, label="area-weighted mean per band")
        ax.plot([0, 70], [0, 70], color="#888", lw=0.7, ls="--", label="y = x")
        mae = np.average(np.abs(p - t), weights=w)
        bias = np.average(p - t, weights=w)
        ax.text(2, 64, f"{len(t)} facets\nMAE {mae:.1f}°, bias {bias:+.1f}°", fontsize=7.5, va="top")
        ax.set_xlim(0, 70)
        ax.set_ylim(0, 70)
        ax.set_aspect("equal")
        ax.set_xlabel("LiDAR pitch, °")
        ax.set_title(title, loc="left")
    axs[0].set_ylabel("predicted pitch (facet mean), °")
    axs[0].legend(frameon=False, loc="lower right", fontsize=6.8)
    fig.tight_layout()
    save(fig, 10)


def fig11_transfer() -> None:
    """Median area error with bootstrap CIs, by town and training set."""
    nt = list(csv.DictReader(open(ROOT / "outputs/new_town/heemstede/per_house.csv")))
    heem_truth = np.array([float(r["truth_m2"]) for r in nt])
    heem_ens = abs_err(np.array([float(r["model_cal_m2"]) for r in nt]), heem_truth)
    heem_b1 = abs_err(np.array([float(r["b1_m2"]) for r in nt]), heem_truth)
    b = bilthoven_area()
    tb, th = two_town_area("bilthoven"), two_town_area("heemstede")
    groups = {
        "Bilthoven": [
            ("footprint × k", abs_err(b["b1"], b["truth"]), BASE_C),
            ("Bilthoven-only, 3 seeds", abs_err(b["model"], b["truth"]), MODEL_C),
            ("Bilthoven-only, seed 0", abs_err(tb["bilthoven_only"], tb["truth"]), "#7fa3d1"),
            ("two-town, seed 0", abs_err(tb["two_town"], tb["truth"]), ALT_C),
        ],
        "Heemstede": [
            ("footprint × k", heem_b1, BASE_C),
            ("Bilthoven-only, 9-model ensemble", heem_ens, MODEL_C),
            ("Bilthoven-only, seed 0", abs_err(th["bilthoven_only"], th["truth"]), "#7fa3d1"),
            ("two-town, seed 0", abs_err(th["two_town"], th["truth"]), ALT_C),
        ],
    }
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.0), sharex=True)
    for ax, (town, items) in zip(axs, groups.items()):
        y = np.arange(len(items))[::-1]
        for yy, (_, e, col) in zip(y, items):
            m = np.median(e)
            lo, hi = boot_median(e)
            ax.barh(yy, m, color=col, height=0.6)
            ax.errorbar(m, yy, xerr=[[m - lo], [hi - m]], fmt="none", color="#222", lw=0.8, capsize=2)
            ax.text(hi + 0.3, yy, f"{m:.1f}%", va="center", fontsize=7.5)
        ax.set_yticks(y, [i[0] for i in items])
        ax.set_xlabel("median |area error|, % (95% bootstrap CI)")
        ax.set_title(f"({'a' if town == 'Bilthoven' else 'b'}) {town}, {len(items[0][1])} held-out houses", loc="left")
        ax.set_xlim(0, 19)
    fig.tight_layout()
    save(fig, 11)


def fig12_curves() -> None:
    """Training curves of the area models (three seeds per fold)."""
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
    for ax, fold in zip(axs, FOLDS):
        for s, run in enumerate((f"cv_{fold}", f"cv_{fold}_s1", f"cv_{fold}_s2")):
            h = json.loads((S4 / run / "history.json").read_text())
            ep = [r["epoch"] for r in h]
            ax.plot(
                ep,
                [r["train_roof"] for r in h],
                color=MODEL_C,
                lw=0.9,
                alpha=0.5 + 0.2 * s,
                label="train" if s == 0 else None,
            )
            ax.plot(
                ep,
                [r["val_roof"] for r in h],
                color=BASE_C,
                lw=0.9,
                alpha=0.5 + 0.2 * s,
                label="validation" if s == 0 else None,
            )
            best = min(h, key=lambda r: r["val_total"])
            ax.plot(best["epoch"], best["val_roof"], "k|", ms=8)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(f"{fold} fold: roof loss, 3 seeds", loc="left")
    axs[0].set_ylabel("BCE (log)")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, 12)


FIGS = {
    1: fig1_overview,
    2: fig2_relief,
    3: fig3_registration,
    4: fig4_labels,
    5: fig5_dataset,
    6: fig6_area,
    7: fig7_examples_and_fig8_confusion,
    9: fig9_edges,
    10: fig10_pitch,
    11: fig11_transfer,
    12: fig12_curves,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="+", type=int, default=list(FIGS))
    args = ap.parse_args()
    for n in args.only:
        FIGS[8 - 1 if n == 8 else n]()


if __name__ == "__main__":
    main()
