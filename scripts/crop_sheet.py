"""Cut sub-tiles out of a full AHN sheet, GeoTiles-style: 1 x 1.25 km, numbered 01-25 row by row
from the north-west, each with an overlap so houses near a border keep their surroundings.

    uv run python scripts/crop_sheet.py --sheet data/raw/ahn5/2023_C_32CN1.LAZ --subtiles 03 04 05 08 09 10 13 15 20

Writes data/raw/ahn5/<sheet>_<nn>.LAZ and <sheet>_<nn>.json (core bounds, overlap, point counts).
One pass over the sheet. The sheet itself is never modified or deleted by this script.

Each sub-tile's "core" is its 1 x 1.25 km cell; the LAZ also holds `overlap` metres around it.
Stage 2 assigns a house to the sub-tile whose core holds its centroid, so no house is counted twice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import laspy
import numpy as np

SHEET_W, SHEET_H, SUB_W, SUB_H = 5000, 6250, 1000, 1250


def subtile_core(sheet_x0: float, sheet_y0: float, k: int) -> tuple[float, float, float, float]:
    col, row = (k - 1) % 5, (k - 1) // 5
    x0 = sheet_x0 + col * SUB_W
    y1 = sheet_y0 + SHEET_H - row * SUB_H
    return (x0, y1 - SUB_H, x0 + SUB_W, y1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheet", type=Path, required=True)
    ap.add_argument("--subtiles", nargs="+", type=int, required=True)
    ap.add_argument("--overlap", type=float, default=25.0)
    ap.add_argument("--chunk", type=int, default=5_000_000)
    ap.add_argument("--out-dir", type=Path, help="default: next to the sheet")
    args = ap.parse_args()

    stem = args.sheet.stem.split("_")[-1]  # 2023_C_32CN1 -> 32CN1
    with laspy.open(args.sheet) as src:
        h = src.header
        # Sheets are aligned to multiples of 5000 m (x) and 6250 m (y) in RD New. Round to the
        # nearest grid line, not down: a file with an overlap starts a little before its sheet.
        x0 = SHEET_W * round(h.mins[0] / SHEET_W)
        y0 = SHEET_H * round(h.mins[1] / SHEET_H)
        print(f"sheet origin (RD New, south-west): {x0:.0f}, {y0:.0f}")
        tiles = {}
        for k in args.subtiles:
            core = subtile_core(x0, y0, k)
            full = (core[0] - args.overlap, core[1] - args.overlap, core[2] + args.overlap, core[3] + args.overlap)
            out = (args.out_dir or args.sheet.parent) / f"{stem}_{k:02d}.LAZ"
            # The header must be complete before the file opens: adding records later moves the
            # point data and laspy refuses. Same format, scales, offsets and VLRs (CRS) as the sheet.
            hdr = laspy.LasHeader(point_format=h.point_format, version=h.version)
            hdr.scales, hdr.offsets = h.scales, h.offsets
            hdr.vlrs = list(h.vlrs)
            tiles[k] = {
                "core": core,
                "full": full,
                "path": out,
                "writer": laspy.open(out, mode="w", header=hdr),
                "written": 0,
                "core_points": 0,
            }
        total = 0
        for chunk in src.chunk_iterator(args.chunk):
            x, y = np.asarray(chunk.x), np.asarray(chunk.y)
            total += len(x)
            for t in tiles.values():
                xmin, ymin, xmax, ymax = t["full"]
                m = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
                if m.any():
                    t["writer"].write_points(chunk[m])
                    t["written"] += int(m.sum())
                    cx0, cy0, cx1, cy1 = t["core"]
                    t["core_points"] += int(((x >= cx0) & (x < cx1) & (y >= cy0) & (y < cy1)).sum())
            print(f"\r  {total / h.point_count:6.1%} of {h.point_count:,} points", end="", flush=True)
        print()
        for t in tiles.values():
            t["writer"].close()

    for k, t in tiles.items():
        with laspy.open(t["path"]) as check:
            n = check.header.point_count
        if n != t["written"]:
            raise SystemExit(f"{t['path'].name}: header says {n} points, {t['written']} were written")
        meta = {
            "sheet": args.sheet.name,
            "subtile": k,
            "core_bounds": t["core"],
            "overlap_m": args.overlap,
            "points": n,
            "core_points": t["core_points"],
            "sheet_points": int(h.point_count),
        }
        t["path"].with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(
            f"{t['path'].name}: {n:,} points ({t['core_points']:,} in the core), "
            f"{t['path'].stat().st_size / 1e6:.0f} MB"
        )


if __name__ == "__main__":
    main()
