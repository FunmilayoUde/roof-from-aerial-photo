"""Stage 1: line up one house's aerial photo with its LiDAR roof, and measure the misalignment.

    uv run python scripts/stage1_align.py --house-id delft_03

Writes
    data/raw/<layer>/<house>.tif         photo around the house (with margin for the search)
    data/processed/<house>/photo.tif     photo cropped to footprint + 1.0 m, as the pipeline crops
    data/processed/<house>/ground.json    local ground height from AHN (cached; reads the whole tile)
    outputs/stage1/<house>/overlay.png   photo + roof, as is and with the fitted shift removed
    outputs/stage1/<house>/relief.png    measured offset against height above ground
    outputs/stage1/<house>/alignment.json
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
from shapely.geometry import JOIN_STYLE  # noqa: E402

from roofphoto.geo.align import bootstrap, edge_pieces, fit_shift, gradients, search_offsets  # noqa: E402
from roofphoto.geo.raster import crop_window  # noqa: E402
from roofphoto.io.lidar import ground_height, load_house  # noqa: E402
from roofphoto.io.pdok import ATTRIBUTION, fetch_ortho, write_geotiff  # noqa: E402
from roofphoto.viz.overlay import draw_overlay, legend_handles  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CROP_BUFFER_M = 1.0  # the LiDAR pipeline's crop: footprint + 1.0 m, mitred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--house-id", required=True)
    ap.add_argument("--layer", default="2023_orthoHR", help="PDOK luchtfotorgb WMS layer")
    ap.add_argument("--res", type=float, default=0.08, help="pixel size in metres (native for orthoHR)")
    ap.add_argument("--margin", type=float, default=6.0, help="metres of photo around the walls to download")
    ap.add_argument("--search", type=float, default=1.2, help="max offset searched per edge piece, m")
    ap.add_argument("--narrow", type=float, default=0.4, help="second-pass search around the first fit, m")
    ap.add_argument("--refresh", action="store_true", help="download the photo again")
    args = ap.parse_args()

    house = load_house(args.house_id)
    out_dir = ROOT / "outputs" / "stage1" / args.house_id
    proc_dir = ROOT / "data" / "processed" / args.house_id
    out_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    # 1. Photo around the house, with room for the offset search.
    raw_path = ROOT / "data" / "raw" / args.layer / f"{args.house_id}.tif"
    if args.refresh or not raw_path.exists():
        fetch_ortho(args.layer, house.footprint.buffer(args.margin).bounds, args.res, raw_path)
        print(f"downloaded {raw_path.relative_to(ROOT)}")
    with rasterio.open(raw_path) as src:
        rgb, transform = src.read(), src.transform
        assert src.crs.to_epsg() == 28992

    # 2. The pipeline's crop: footprint + 1.0 m with mitred corners; the raster takes its bounds.
    crop_poly = house.footprint.buffer(CROP_BUFFER_M, join_style=JOIN_STYLE.mitre)
    rows, cols, crop_t = crop_window(transform, rgb.shape[1:], crop_poly.bounds)
    write_geotiff(proc_dir / "photo.tif", rgb[:, rows, cols], crop_t, tags={"layer": args.layer})

    # 3. Ground height, so roof heights can be expressed above the ground the photo was corrected to.
    ground_path = proc_dir / "ground.json"
    if not ground_path.exists():
        print("reading ground points from the AHN tile (one pass over the tile)...")
        ground_path.write_text(json.dumps(ground_height(args.house_id, house.footprint), indent=2))
    ground = json.loads(ground_path.read_text())
    gz = ground["ground_z_nap_m"]

    # 4. Per-piece offsets, then the two models.
    #    'unclassified' edges come from the pipeline's traced fallback outlines, which it marks
    #    approximate; they would measure the pipeline's error, not the photo's offset.
    edges = [e for e in house.edges if e.type != "unclassified"]
    pieces = edge_pieces(edges)
    grad = gradients(rgb)
    # Pass 1: wide search, so a large offset can be found at all.
    first = [r for r in search_offsets(grad, transform, pieces, search_m=args.search) if not r.at_limit]
    n1 = np.array([r.piece.normal for r in first])
    h1 = np.array([r.piece.z - gz for r in first])
    fit1 = fit_shift(n1, np.array([r.offset_m for r in first]), h1)
    # Pass 2: narrow search around where pass 1 predicts each edge, which keeps pieces from
    # locking onto a stronger line (gutter, chimney, trim) further away.
    h_all = np.array([p.z - gz for p in pieces])
    centres = np.einsum("ij,ij->i", np.array([p.normal for p in pieces]), fit1.s0 + h_all[:, None] * fit1.v)
    found = search_offsets(grad, transform, pieces, search_m=args.narrow, centres_m=centres)
    found = [r for r in found if not r.at_limit]
    normals = np.array([r.piece.normal for r in found])
    d = np.array([r.offset_m for r in found])
    h = np.array([r.piece.z - gz for r in found])
    types = np.array([r.piece.edge_type for r in found])
    const = fit_shift(normals, d)
    relief = fit_shift(normals, d, h)
    draws = bootstrap(normals, d, h)
    lo, hi = np.percentile(draws, [5, 95], axis=0)

    def residual(fit) -> np.ndarray:
        shift = np.broadcast_to(fit.s0 + (0 if fit.v is None else h[:, None] * fit.v), normals.shape)
        return d - np.sum(normals * shift, axis=1)

    r_const, r_relief = residual(const), residual(relief)
    by_type = {}
    for t in sorted(set(types)):
        m = (types == t) & relief.inliers
        if m.any():
            by_type[t] = {
                "pieces": int(m.sum()),
                "median_height_m": round(float(np.median(h[m])), 2),
                "rms_residual_constant_m": round(float(np.sqrt(np.mean(r_const[m] ** 2))), 3),
                "rms_residual_relief_m": round(float(np.sqrt(np.mean(r_relief[m] ** 2))), 3),
            }

    # The house's long axis (from the BAG outline's minimum rotated rectangle). On a rectangular
    # house nearly every edge is parallel to one of the two axes, so each axis is constrained only
    # by the edges across it; reporting per axis shows when one direction rests on few pieces.
    rect = np.asarray(house.footprint.minimum_rotated_rectangle.exterior.coords)[:3]
    sides = np.diff(rect, axis=0)
    long_side = sides[np.argmax(np.hypot(sides[:, 0], sides[:, 1]))]
    along_dir = long_side / np.linalg.norm(long_side)
    if along_dir[0] < 0:
        along_dir = -along_dir
    across_dir = np.array([-along_dir[1], along_dir[0]])

    def per_axis(vec: np.ndarray) -> dict:
        return {"along_m": round(float(vec @ along_dir), 3), "across_m": round(float(vec @ across_dir), 3)}

    axis_support = {
        name: int(((np.abs(normals @ vec) > 0.9) & relief.inliers).sum())
        for name, vec in (("along", along_dir), ("across", across_dir))
    }

    h_eave = float(np.percentile(h, 10))
    h_top = float(np.percentile(h, 90))
    result = {
        "house_id": args.house_id,
        "photo": {
            "layer": args.layer,
            "res_m": args.res,
            "attribution": ATTRIBUTION,
            "file": str(raw_path.relative_to(ROOT)),
        },
        "ground": ground,
        "pieces": {
            "first_pass": len(first),
            "searched": len(d),
            "inliers_constant": const.n,
            "inliers_relief": relief.n,
        },
        "constant_model": {
            "shift_m": const.s0.round(3).tolist(),
            "magnitude_m": round(float(np.hypot(*const.s0)), 3),
            "rms_m": round(const.rms_m, 3),
        },
        "relief_model": {
            "s0_m": relief.s0.round(3).tolist(),
            "s0_90ci_m": [lo[:2].round(3).tolist(), hi[:2].round(3).tolist()],
            "v_m_per_m": relief.v.round(4).tolist(),
            "v_90ci": [lo[2:].round(4).tolist(), hi[2:].round(4).tolist()],
            "v_direction_deg_from_east": round(float(np.degrees(np.arctan2(relief.v[1], relief.v[0]))), 1),
            "rms_m": round(relief.rms_m, 3),
            "shift_at_heights_m": {f"{hh:.1f}": relief.shift_at(hh).round(3).tolist() for hh in (0.0, h_eave, h_top)},
        },
        "house_axes": {
            "along_bearing_deg": round(float((90 - np.degrees(np.arctan2(along_dir[1], along_dir[0]))) % 360), 1),
            "inlier_pieces_constraining": axis_support,
            "constant_shift": per_axis(const.s0),
            "relief_s0": per_axis(relief.s0),
            "relief_v_per_m": per_axis(relief.v),
        },
        "by_edge_type": by_type,
        "heights_above_ground_m": {"p10": round(h_eave, 2), "p90": round(h_top, 2)},
    }
    (out_dir / "alignment.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    # 5. Overlays: as delivered, with the constant shift, and with the height-dependent shift.
    def const_shift(p: np.ndarray) -> np.ndarray:
        return p[:, :2] + const.s0

    def relief_shift(p: np.ndarray) -> np.ndarray:
        return p[:, :2] + relief.s0 + (p[:, 2:3] - gz) * relief.v

    crop_rgb = rgb[:, rows, cols]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), constrained_layout=True)
    draw_overlay(axes[0], crop_rgb, crop_t, house, title=f"{args.house_id}: LiDAR roof as is ({args.layer})")
    draw_overlay(axes[1], crop_rgb, crop_t, house, shift=const_shift, title="constant shift s0 applied")
    draw_overlay(axes[2], crop_rgb, crop_t, house, shift=relief_shift, title="height-dependent shift s0 + h·v applied")
    axes[2].legend(handles=legend_handles(house), loc="lower right", fontsize=7, framealpha=0.85)
    fig.text(
        0.99, 0.005, f"Photo: {ATTRIBUTION}. Roof: AHN5 (CC0) via roofrecon.", ha="right", fontsize=7, color="#555"
    )
    fig.savefig(out_dir / "overlay.png", dpi=150)
    plt.close(fig)

    # 6. Offset against height, per house axis. Each dot is one edge piece's measured offset
    #    (no model). Relief displacement: dots rise with height along a line through ~0 at h = 0.
    #    Constant coordinate error: dots flat, away from zero.
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True, sharey=True)
    hh = np.linspace(0, max(h) * 1.05, 50)
    for ax, (name, ax_vec) in zip(axs, (("across", across_dir), ("along", along_dir))):
        cos = normals @ ax_vec
        use = np.abs(cos) > 0.9
        dd = d[use] * np.sign(cos[use])  # offset in the axis direction
        inl = relief.inliers[use]
        ax.scatter(h[use][inl], dd[inl], s=16, color="#2a78d6", label="edge pieces")
        ax.scatter(h[use][~inl], dd[~inl], s=16, color="#bbb", label="outliers")
        ax.plot(hh, np.full_like(hh, const.s0 @ ax_vec), color="#555", ls="--", label="constant model")
        ax.plot(hh, relief.s0 @ ax_vec + hh * (relief.v @ ax_vec), color="#d6572a", label="s0 + h·v model")
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xlim(0, None)
        ax.set_xlabel("height above ground, m")
        bearing = (90 - np.degrees(np.arctan2(ax_vec[1], ax_vec[0]))) % 360
        ax.set_title(f"{name} the long axis (towards bearing {bearing:.0f}°): {int(use.sum())} pieces", fontsize=9)
    axs[0].set_ylabel("photo edge minus LiDAR edge, m")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"{args.house_id}: where the photo shows each roof edge, relative to LiDAR", fontsize=10)
    fig.savefig(out_dir / "relief.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out_dir.relative_to(ROOT)}/overlay.png, relief.png, alignment.json")


if __name__ == "__main__":
    main()
