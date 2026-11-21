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

from roofphoto.geo.raster import crop_window  # noqa: E402
from roofphoto.geo.register import register_house  # noqa: E402
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

    # 4. Per-piece offsets and the three fits (see roofphoto.geo.register).
    reg = register_house(house, rgb, transform, gz, search_m=args.search, narrow_m=args.narrow, with_bootstrap=True)
    if reg is None:
        raise SystemExit(f"{args.house_id}: too few edge pieces matched to fit a shift")
    normals, d, h = reg.normals, reg.d, reg.h
    const, pure, relief = reg.const, reg.pure, reg.free
    common = reg.common
    along_dir, across_dir = reg.along_dir, reg.across_dir
    result = {
        "house_id": args.house_id,
        "photo": {
            "layer": args.layer,
            "res_m": args.res,
            "attribution": ATTRIBUTION,
            "file": str(raw_path.relative_to(ROOT)),
        },
        "ground": ground,
    } | reg.summary()
    (out_dir / "alignment.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    # 5. Overlays: as delivered, with the constant shift, and with the height-dependent shift.
    def const_shift(p: np.ndarray) -> np.ndarray:
        return p[:, :2] + const.s0

    def relief_shift(p: np.ndarray) -> np.ndarray:
        return p[:, :2] + (p[:, 2:3] - gz) * pure.v

    crop_rgb = rgb[:, rows, cols]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), constrained_layout=True)
    draw_overlay(axes[0], crop_rgb, crop_t, house, title=f"{args.house_id}: LiDAR roof as is ({args.layer})")
    draw_overlay(axes[1], crop_rgb, crop_t, house, shift=const_shift, title="constant shift s0 applied")
    draw_overlay(
        axes[2],
        crop_rgb,
        crop_t,
        house,
        shift=relief_shift,
        title="relief shift h·v applied (each vertex by its height)",
    )
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
        inl = common[use]
        ax.scatter(h[use][inl], dd[inl], s=16, color="#2a78d6", label="edge pieces")
        ax.scatter(h[use][~inl], dd[~inl], s=16, color="#bbb", label="outliers")
        ax.plot(hh, np.full_like(hh, const.s0 @ ax_vec), color="#555", ls="--", label="constant model")
        ax.plot(
            hh,
            relief.s0 @ ax_vec + hh * (relief.v @ ax_vec),
            color="#d6572a",
            lw=0.8,
            ls=":",
            label="s0 + h·v (free s0)",
        )
        ax.plot(hh, hh * (pure.v @ ax_vec), color="#d6572a", label="h·v (relief, s0 = 0)")
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
