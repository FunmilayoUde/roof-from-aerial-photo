"""Stage 5: facets and edge types from the trained models, scored against the Stage 2 edge labels.

    uv run python scripts/stage5_edges.py

Per fold, on its test houses:
  roof         mean roof probability of the 3 Stage 4 models (cv_<fold>, _s1, _s2) > 0.5, keeping
               the regions that touch the house's BAG outline + 1.5 m (as in the area evaluation)
  orientation  mean of the 3 orientation models' probabilities (s5w_<fold>_geom_s0..2), over the
               9 roof classes, inside that roof
  facets/edges roofphoto.edges: connected regions of one orientation; boundaries typed by geometry
Scored per edge type: a predicted edge pixel is correct if a labelled edge of the same type lies
within 3 px (24 cm), and a labelled pixel is found if a prediction of its type does. Pixels within
3 px of label-ignore regions (neighbours, uncertain facets) are left out.
An "oracle" applies the same rules to the true roof and orientation labels: the ceiling of the
rules themselves, separating rule errors from prediction errors.
Writes outputs/stage5/edges.json and previews in outputs/stage5/edge_previews/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from scipy.ndimage import binary_dilation, distance_transform_edt, label  # noqa: E402

from roofphoto.data.chips import EDGE_CLASSES  # noqa: E402
from roofphoto.data.lookup import footprints  # noqa: E402
from roofphoto.edges import edge_raster, facets_from_orientation, score_edges  # noqa: E402
from roofphoto.regularise import axis_angle_px, regularise_facets  # noqa: E402
from roofphoto.models import unet  # noqa: E402
from roofphoto.models.dataset import IGNORE, IMAGENET_MEAN, IMAGENET_STD, kept_chips, load_targets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FOLDS = {
    "west": (["cv_west", "cv_west_s1", "cv_west_s2"], [f"s5w_west_geom_s{s}" for s in range(3)]),
    "centre": (["cv_centre", "cv_centre_s1", "cv_centre_s2"], [f"s5w_centre_geom_s{s}" for s in range(3)]),
    "east": (["cv_east", "cv_east_s1", "cv_east_s2"], [f"s5w_east_geom_s{s}" for s in range(3)]),
}
# Learned edge output (Stage 5): scored as "model_edge" for folds that have these runs.
EDGE_RUNS = {fold: [f"s5e_{fold}_s{s}" for s in range(3)] for fold in FOLDS}
# Learned edge locations, typed by the rules (Stage 5): scored as "model_edge_bin".
EDGE_BIN_RUNS = {fold: [f"s5b_{fold}_s{s}" for s in range(3)] for fold in FOLDS}
TOL_PX = 3.0
MIN_FACET_PX = 150  # ~1 m² at 8 cm


def load(run: str, device, edges: bool = False) -> torch.nn.Module:
    m = unet.build(pretrained=False, edges=edges)
    m.load_state_dict(torch.load(ROOT / "outputs/stage4" / run / "model.pt", map_location="cpu"))
    return m.to(device).eval()


def add(total: dict, part: dict) -> None:
    for k, v in part.items():
        total[k] = [a + b for a, b in zip(total.get(k, [0, 0, 0, 0]), v)]


def rates(counts: dict) -> dict:
    return {
        k: {
            "precision": round(tp_p / n_p, 3) if n_p else None,
            "recall": round(tp_t / n_t, 3) if n_t else None,
            "predicted_px": n_p,
            "labelled_px": n_t,
        }
        for k, (tp_p, n_p, tp_t, n_t) in counts.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--previews", type=int, default=4, help="per fold")
    ap.add_argument("--folds", nargs="+", default=list(FOLDS))
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = ROOT / "outputs/stage5"
    (out_dir / "edge_previews").mkdir(parents=True, exist_ok=True)
    result, pooled = {}, {"model": {}, "model_reg": {}, "model_edge": {}, "model_edge_bin": {}, "oracle": {}}

    for fold, (roof_runs, orient_runs) in ((f, FOLDS[f]) for f in args.folds):
        split = json.loads((ROOT / "outputs/stage4" / roof_runs[0] / "split.json").read_text())
        roof_models = [load(r, device) for r in roof_runs]
        orient_models = [load(r, device) for r in orient_runs]
        edge_runs = [r for r in EDGE_RUNS[fold] if (ROOT / "outputs/stage4" / r / "model.pt").exists()]
        edge_models = [load(r, device, edges=True) for r in edge_runs]
        bin_runs = [r for r in EDGE_BIN_RUNS[fold] if (ROOT / "outputs/stage4" / r / "model.pt").exists()]
        bin_models = [load(r, device, edges=True) for r in bin_runs]
        print(f"{fold}: edge models {edge_runs or 'none'}; edge-location models {bin_runs or 'none'}")
        counts = {"model": {}, "model_reg": {}, "model_edge": {}, "model_edge_bin": {}, "oracle": {}}
        shown = 0
        for tile in split["test"]:
            fps = footprints(ROOT, tile)
            for chip in kept_chips(ROOT, [tile]):
                image, orient_t, _, roof_t = load_targets(chip)
                with rasterio.open(chip.folder / f"labels_{chip.labels}.tif") as src:
                    edges_t, t = src.read(2), src.transform
                x = torch.from_numpy(image)[None].to(device)
                with torch.no_grad():
                    roof_p = np.mean([torch.sigmoid(unet.split(m(x))[2][0]).cpu().numpy() for m in roof_models], 0)
                    probs = np.mean(
                        [torch.softmax(unet.split(m(x))[0][0, 1:], 0).cpu().numpy() for m in orient_models], 0
                    )
                orient_p = probs.argmax(0) + 1
                seed = rasterize([(fps[chip.house_id].buffer(1.5), 1)], out_shape=orient_t.shape, transform=t) == 1
                comp, _ = label(roof_p > 0.5, structure=np.ones((3, 3)))
                house = np.isin(comp, np.unique(comp[seed & (comp > 0)])) & (comp > 0)

                facets, cls = facets_from_orientation(house, orient_p, MIN_FACET_PX)
                edges_p = edge_raster(facets, cls)
                # Straightened: facets as polygons squared to the house's axes (from its outline).
                facets_r = regularise_facets(facets, axis_angle_px(fps[chip.house_id], t))
                edges_r = edge_raster(facets_r, {i: c for i, c in cls.items() if (facets_r == i).any()})
                known = (orient_t >= 1) & (orient_t != IGNORE)
                facets_o, cls_o = facets_from_orientation(known, orient_t.astype(int), MIN_FACET_PX)
                edges_o = edge_raster(facets_o, cls_o)

                ignore = binary_dilation(edges_t == IGNORE, iterations=int(TOL_PX))
                add(counts["model"], score_edges(edges_p, edges_t, ignore, TOL_PX))
                add(counts["model_reg"], score_edges(edges_r, edges_t, ignore, TOL_PX))
                if edge_models:
                    # Learned edges: mean probability over the edge models, kept within ~30 cm of
                    # the predicted house roof (eaves sit on its outline).
                    with torch.no_grad():
                        ep = np.mean(
                            [torch.softmax(unet.edge_logits(m(x))[0], 0).cpu().numpy() for m in edge_models], 0
                        )
                    edges_e = np.where(binary_dilation(house, iterations=4), ep.argmax(0), 0).astype(np.uint8)
                    add(counts["model_edge"], score_edges(edges_e, edges_t, ignore, TOL_PX))
                if bin_models and edges_p.any():
                    # Where: the models' mean edge probability > 0.5 near the predicted roof.
                    # Which type: that of the nearest rule-based edge on the same house.
                    with torch.no_grad():
                        pe = np.mean(
                            [torch.softmax(unet.edge_logits(m(x))[0, :2], 0)[1].cpu().numpy() for m in bin_models], 0
                        )
                    where = (pe > 0.5) & binary_dilation(house, iterations=4)
                    _, (rr, cc) = distance_transform_edt(edges_p == 0, return_indices=True)
                    edges_b = np.where(where, edges_p[rr, cc], 0).astype(np.uint8)
                    add(counts["model_edge_bin"], score_edges(edges_b, edges_t, ignore, TOL_PX))
                add(counts["oracle"], score_edges(edges_o, edges_t, ignore, TOL_PX))
                if shown < args.previews:
                    preview(
                        image,
                        edges_t,
                        edges_o,
                        edges_p,
                        edges_r,
                        chip.house_id,
                        out_dir / "edge_previews" / f"{chip.house_id}.png",
                    )
                    shown += 1
        result[fold] = {k: rates(v) for k, v in counts.items()}
        for k in counts:
            add(pooled[k], counts[k])
        print(
            fold,
            json.dumps(
                {
                    k: {
                        t: v
                        for t, v in rates(counts[k]).items()
                        if t in ("any", "ridge", "eave", "rake", "hip", "valley")
                    }
                    for k in counts
                }
            ),
        )
    result["pooled"] = {k: rates(v) for k, v in pooled.items()}
    (out_dir / "edges.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["pooled"], indent=2))


def preview(image, truth, oracle, pred, pred_reg, hid, path) -> None:
    rgb = np.clip(np.moveaxis(image * IMAGENET_STD + IMAGENET_MEAN, 0, -1), 0, 1)
    cmap = plt.get_cmap("tab10", len(EDGE_CLASSES) + 1)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.4), constrained_layout=True)
    for ax, e, title in (
        (axs[0], truth, "labels (from LiDAR)"),
        (axs[1], oracle, "rules on true orientation"),
        (axs[2], pred, "rules on predictions"),
        (axs[3], pred_reg, "predictions, straightened"),
    ):
        ax.imshow(rgb)
        shown = np.ma.masked_where((e == 0) | (e == IGNORE), e)
        ax.imshow(shown, cmap=cmap, vmin=0, vmax=len(EDGE_CLASSES), interpolation="nearest")
        ax.set_title(f"{hid}: {title}", fontsize=9)
        ax.set_axis_off()
    handles = [plt.Line2D([], [], color=cmap(v), lw=3, label=k) for k, v in EDGE_CLASSES.items() if k != "unclassified"]
    axs[3].legend(handles=handles, fontsize=7, loc="lower right")
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
