"""Stage 1 follow-up: is the photo offset at *ground* level about zero (relief displacement) or
about the same as on the roofs (coordinate error)?

    uv run python scripts/stage1_ground_check.py                       # the four Stage 1 houses
    uv run python scripts/stage1_ground_check.py --house-id delft_03

Needs scripts/stage1_align.py to have run for each house (it compares with the roof shift).

Writes
    data/raw/<layer>/<house>_ground.tif             photo, footprint + 25 m
    data/processed/<house>/ground_points.npz          AHN ground points with intensity (cached)
    outputs/stage1/<house>/ground_check.png, ground_check.json
    outputs/stage1/ground_check_summary.json
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

from roofphoto.geo.align import gradients  # noqa: E402
from roofphoto.geo.ground import compare_shifts, filled_edges, rasterize_mean, search_shift  # noqa: E402
from roofphoto.geo.raster import pixel_to_world  # noqa: E402
from roofphoto.io.lidar import load_house, read_ground_intensity, tile_path  # noqa: E402
from roofphoto.io.pdok import ATTRIBUTION, fetch_ortho  # noqa: E402
from roofphoto.viz.overlay import raster_extent  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HOUSES = ["delft_01", "delft_02", "delft_03", "delft_05"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--house-id", action="append", help="repeatable; default: the four Stage 1 houses")
    ap.add_argument("--layer", default="2023_orthoHR")
    ap.add_argument("--res", type=float, default=0.08, help="photo pixel size, m")
    ap.add_argument("--cell", type=float, default=0.25, help="intensity raster cell, m (~2 points per cell)")
    ap.add_argument("--radius", type=float, default=25.0, help="metres of ground around the walls to use")
    ap.add_argument("--search", type=float, default=1.6, help="max shift searched, m")
    args = ap.parse_args()
    houses = {h: load_house(h) for h in (args.house_id or HOUSES)}

    # Ground points for every house that isn't cached yet, in one pass per tile.
    todo: dict[Path, dict[str, tuple]] = {}
    for hid, house in houses.items():
        if not (ROOT / "data" / "processed" / hid / "ground_points.npz").exists():
            todo.setdefault(tile_path(hid), {})[hid] = house.footprint.buffer(args.radius).bounds
    for tile, bboxes in todo.items():
        print(f"reading ground points for {', '.join(bboxes)} from {tile.name} (one pass)...")
        for hid, pts in read_ground_intensity(tile, bboxes).items():
            out = ROOT / "data" / "processed" / hid / "ground_points.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, xyzi=pts)

    summary = {}
    for hid, house in houses.items():
        summary[hid] = check_house(hid, house, args)
    (ROOT / "outputs" / "stage1" / "ground_check_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def check_house(hid: str, house, args) -> dict:
    out_dir = ROOT / "outputs" / "stage1" / hid
    roof = json.loads((out_dir / "alignment.json").read_text())
    roof_shift = np.array(roof["constant_model"]["shift_m"])

    photo_path = ROOT / "data" / "raw" / args.layer / f"{hid}_ground.tif"
    if not photo_path.exists():
        fetch_ortho(args.layer, house.footprint.buffer(args.radius).bounds, args.res, photo_path)
    with rasterio.open(photo_path) as src:
        rgb, photo_t = src.read(), src.transform
    # Smooth the photo to the intensity raster's scale before taking edges, so both see the same detail.
    photo_edges = np.hypot(*gradients(rgb, sigma_px=args.cell / args.res))

    xyzi = np.load(ROOT / "data" / "processed" / hid / "ground_points.npz")["xyzi"]
    lo, hi = np.percentile(xyzi[:, 3], [1, 99])
    intensity = np.clip(xyzi[:, 3], lo, hi)
    bounds = house.footprint.buffer(args.radius).bounds
    mean, count, ref_t = rasterize_mean(xyzi[:, :2], intensity, bounds, args.cell)
    ref, valid = filled_edges(mean)

    # Whole ring, then four sectors around the house separately: if the photo is offset at
    # ground level, every sector should agree on it.
    whole = search_shift(ref, valid, ref_t, photo_edges, photo_t, search_m=args.search)
    rows, cols = np.nonzero(valid)
    x, y = pixel_to_world(ref_t, cols + 0.5, rows + 0.5)
    c = house.footprint.centroid
    bearing = (90 - np.degrees(np.arctan2(y - c.y, x - c.x))) % 360
    sectors = {}
    for name, centre in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
        in_sector = np.abs((bearing - centre + 180) % 360 - 180) < 45
        mask = np.zeros_like(valid)
        mask[rows[in_sector], cols[in_sector]] = True
        if mask.sum() < 500:
            continue
        s = search_shift(ref, mask, ref_t, photo_edges, photo_t, search_m=args.search)
        sectors[name] = {"shift_m": s.shift.round(2).tolist(), "ncc": round(s.peak, 3), "cells": s.cells}

    # The direct test: does "no shift" (relief displacement) match the ground better than the
    # roof's shift (coordinate error)? On a site whose ground features run mostly one way, the
    # two can tie, and this says so instead of picking a winner from noise.
    test = compare_shifts(ref, valid, ref_t, photo_edges, photo_t, np.zeros(2), roof_shift)
    test = {
        "ncc_no_shift": test["ncc_a"],
        "ncc_roof_shift": test["ncc_b"],
        "no_shift_wins_fraction": test["a_beats_b_fraction"],
        "blocks_5m": test["blocks"],
    }

    spread = np.array([v["shift_m"] for v in sectors.values()])
    result = {
        "ground_shift_m": whole.shift.round(3).tolist(),
        "ground_shift_size_m": round(float(np.hypot(*whole.shift)), 3),
        "ncc_peak": round(whole.peak, 3),
        "ncc_at_zero": round(whole.at_zero, 3),
        "cells": whole.cells,
        "ground_points": int(len(xyzi)),
        "zero_vs_roof_shift": test,
        "sectors": sectors,
        "sector_spread_m": spread.std(axis=0).round(3).tolist() if len(spread) > 1 else None,
        "roof_shift_m": roof_shift.round(3).tolist(),
        "roof_shift_size_m": round(float(np.hypot(*roof_shift)), 3),
    }
    (out_dir / "ground_check.json").write_text(json.dumps(result, indent=2))
    plot(hid, house, rgb, photo_t, mean, ref_t, whole, roof_shift, out_dir / "ground_check.png")
    return result


def plot(hid, house, rgb, photo_t, mean, ref_t, whole, roof_shift, path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
    axs[0].imshow(np.moveaxis(rgb, 0, -1), extent=raster_extent(photo_t, rgb.shape[1:]))
    axs[0].set_title(f"{hid}: photo ({ATTRIBUTION})", fontsize=9)
    axs[1].imshow(mean, extent=raster_extent(ref_t, mean.shape), cmap="gray", interpolation="nearest")
    axs[1].set_title("AHN ground-point intensity (blank: no ground points)", fontsize=9)
    for ax in axs[:2]:
        wx, wy = house.footprint.exterior.xy
        ax.plot(wx, wy, color="#d6572a", lw=1)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(useOffset=False, style="plain")
    o = whole.offsets
    step = o[1] - o[0]
    ext = (o[0] - step / 2, o[-1] + step / 2, o[0] - step / 2, o[-1] + step / 2)
    im = axs[2].imshow(whole.surface, origin="lower", extent=ext, cmap="viridis")
    fig.colorbar(im, ax=axs[2], shrink=0.8, label="NCC of edges")
    axs[2].plot(*whole.shift, "x", color="white", ms=12, mew=2.5, label=f"ground best {whole.shift.round(2)}")
    axs[2].plot(0, 0, "+", color="#ff4d4d", ms=14, mew=2, label="no shift (relief displacement)")
    axs[2].plot(*roof_shift, "o", mfc="none", mec="#ffb000", ms=12, mew=2, label=f"roof shift {roof_shift.round(2)}")
    axs[2].set_xlabel("dx (east), m")
    axs[2].set_ylabel("dy (north), m")
    axs[2].set_title("ground-level match: photo minus LiDAR", fontsize=9)
    axs[2].legend(fontsize=7, loc="lower left")
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
