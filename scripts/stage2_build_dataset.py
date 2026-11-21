"""Stage 2: build the training set for one AHN tile: photo chip + LiDAR-derived labels per house.

    uv run python scripts/stage2_build_dataset.py                              # the local Delft tile
    uv run python scripts/stage2_build_dataset.py --tile path/to/<sheet>_<nn>.LAZ

For every detached-house candidate in the tile:
  1. teacher: the LiDAR pipeline, run in memory (roofphoto.data.teacher), one pass over the tile
  2. photo: the PDOK 2023 8 cm orthophoto, a square chip around the house
  3. registration: the Stage 1 edge matching, constant and pure-relief fits
  4. quality: provisional checks on the teacher and the registration (roofphoto.data.quality)
  5. labels: facets + edge types rasterised on the chip's grid, once per correction model

Writes (all gitignored)
  data/raw/bag/<tile>.json                              BAG outlines in the tile (cached)
  data/raw/<layer>/stage2/<house>.tif                   photo chip
  data/processed/stage2/<tile>/houses_private.csv       house id -> BAG id (never commit)
  data/processed/stage2/<tile>/<house>/                 report.json, teacher.json, registration.json,
                                                        quality.json, photo.tif, labels_constant.tif,
                                                        labels_relief.tif, facets.json
  outputs/stage2/<tile>/manifest.csv, summary.json, previews/<house>.png
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
from roofrecon.las_io import header_summary  # noqa: E402

from roofphoto.data.candidates import bag_buildings, find_candidates  # noqa: E402
from roofphoto.data.chips import (  # noqa: E402
    IGNORE,
    chip_grid,
    constant_shift,
    neighbour_ignore,
    rasterize_labels,
    relief_shift,
)
from roofphoto.data.quality import registration_checks, summarise, teacher_checks  # noqa: E402
from roofphoto.data.teacher import READ_MARGIN_M, run_teacher  # noqa: E402
from roofphoto.geo.register import register_house  # noqa: E402
from roofphoto.io.lidar import labels_from_report, lidar_root, read_points_multi  # noqa: E402
from roofphoto.io.pdok import ATTRIBUTION, fetch_ortho, write_geotiff  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HEIGHT_BINS = [0, 4, 6, 8, 99]  # m above ground, for pooling residuals by height


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tile", type=Path, default=lidar_root() / "data" / "raw" / "AHN5_(withheld).LAZ")
    ap.add_argument("--layer", default="2023_orthoHR")
    ap.add_argument("--res", type=float, default=0.08)
    ap.add_argument("--chip-m", type=float, default=40.96, help="chip side, m (512 px at 8 cm)")
    ap.add_argument("--min-gap", type=float, default=3.0)
    ap.add_argument("--min-area", type=float, default=70.0)
    ap.add_argument("--max-area", type=float, default=300.0)
    ap.add_argument("--limit", type=int, help="process only the first N candidates (for trial runs)")
    args = ap.parse_args()

    tile_name = args.tile.stem
    proc = ROOT / "data" / "processed" / "stage2" / tile_name
    out = ROOT / "outputs" / "stage2" / tile_name
    (out / "previews").mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)

    h = header_summary(args.tile)
    bounds = (h["mins"][0], h["mins"][1], h["maxs"][0], h["maxs"][1])
    feats = bag_buildings(bounds, ROOT / "data" / "raw" / "bag" / f"{tile_name}.json")
    cands = find_candidates(feats, bounds, args.min_area, args.max_area, args.min_gap)[: args.limit]
    print(f"{tile_name}: {len(feats)} BAG buildings, {len(cands)} candidates")
    ids = {c.bag_id: f"{tile_name}_{i:04d}" for i, c in enumerate(cands)}
    with (proc / "houses_private.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["house_id", "bag_id"])
        w.writerows((ids[c.bag_id], c.bag_id) for c in cands)

    all_polys = [(f["properties"]["identificatie"], f["geometry"]) for f in feats]
    print("reading points for all candidates (one pass over the tile)...")
    pts = read_points_multi(args.tile, {ids[c.bag_id]: c.footprint.buffer(READ_MARGIN_M).bounds for c in cands})

    rows, pooled = [], []
    for c in cands:
        hid = ids[c.bag_id]
        row, pieces = process_house(hid, c, pts.pop(hid), all_polys, tile_name, proc / hid, out, args)
        rows.append(row)
        if pieces is not None:
            pooled.append(pieces)
        print(f"  {hid}: {row['status']}" + (f" ({', '.join(row['failed'])})" if row["failed"] else ""))

    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r | {"failed": ";".join(r["failed"])})
    summary = summarise_run(rows, pooled, tile_name, args)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def process_house(hid, cand, pts, all_polys, tile_name, house_dir: Path, out: Path, args) -> tuple[dict, dict | None]:
    from shapely.geometry import box, shape

    house_dir.mkdir(parents=True, exist_ok=True)
    row = {"house_id": hid, "status": "", "failed": [], "footprint_m2": round(cand.footprint.area, 1)}
    row |= {"concavity": round(cand.concavity, 3), "built": cand.built}

    # 1. Teacher.
    t = run_teacher(hid, pts, cand.footprint, cand.bag_id, tile_name)
    (house_dir / "teacher.json").write_text(json.dumps(t.stats | {"ok": t.ok, "reason": t.reason}, indent=2))
    if not t.ok:
        row |= {"status": "teacher failed", "failed": [t.reason]}
        return row, None
    (house_dir / "report.json").write_text(json.dumps(t.report, indent=2, ensure_ascii=False))
    tot = t.report["totals"]
    row |= {"roof_area_m2": tot["roof_area_m2"], "facets": tot["facets"]}
    gz = t.stats["ground"]["ground_z_nap_m"]
    if gz is None:
        row |= {"status": "no ground points", "failed": ["ground"]}
        return row, None

    # 2. Photo chip: the house must fit with 6 m around it (room for the shift search).
    house = labels_from_report(t.report, cand.footprint)
    c = cand.footprint.centroid
    chip_t, n = chip_grid((c.x, c.y), args.chip_m, args.res)
    chip_box = box(chip_t.c, chip_t.f - n * args.res, chip_t.c + n * args.res, chip_t.f)
    if not chip_box.contains(cand.footprint.buffer(6.0)):
        row |= {"status": "too large for chip", "failed": ["chip_size"]}
        return row, None
    photo_path = ROOT / "data" / "raw" / args.layer / "stage2" / f"{hid}.tif"
    if not photo_path.exists():
        fetch_ortho(args.layer, chip_box.bounds, args.res, photo_path)
    with rasterio.open(photo_path) as src:
        rgb, photo_t = src.read(), src.transform
    if not photo_t.almost_equals(chip_t) or rgb.shape[1:] != (n, n):
        raise RuntimeError(f"{hid}: photo grid does not match the chip grid")

    # 3. Registration and 4. quality.
    reg = register_house(house, rgb, photo_t, gz)
    checks = teacher_checks(t.report, t.stats) + registration_checks(reg)
    q = summarise(checks)
    (house_dir / "quality.json").write_text(json.dumps(q, indent=2))
    row |= {"status": "kept" if q["kept"] else "dropped", "failed": q["failed"]}
    row |= {k: v["value"] for k, v in q["checks"].items()}
    if reg is None:
        return row, None
    s = reg.summary()
    (house_dir / "registration.json").write_text(json.dumps({"ground_z_nap_m": gz} | s, indent=2))
    cmp = s["model_comparison_same_pieces"]
    row |= {
        "shift_e_m": s["constant_model"]["shift_m"][0],
        "shift_n_m": s["constant_model"]["shift_m"][1],
        "rms_constant_m": cmp["rms_constant_m"],
        "rms_pure_relief_m": cmp["rms_pure_relief_m"],
        "v_e": s["pure_relief_model"]["v_m_per_m"][0],
        "v_n": s["pure_relief_model"]["v_m_per_m"][1],
    }

    # 5. Labels, once per correction model, plus the photo chip.
    neighbours = [shape(g) for bid, g in all_polys if bid != cand.bag_id]
    near = [p for p in neighbours if p.intersects(chip_box)]
    ignore = neighbour_ignore(near) if near else None
    labels = {
        "constant": rasterize_labels(house, constant_shift(reg.const.s0), photo_t, n, ignore),
        "relief": rasterize_labels(house, relief_shift(reg.pure.v, gz), photo_t, n, ignore),
    }
    write_geotiff(house_dir / "photo.tif", rgb, photo_t, tags={"layer": args.layer, "attribution": ATTRIBUTION})
    for name, lab in labels.items():
        write_geotiff(house_dir / f"labels_{name}.tif", lab, photo_t, tags={"bands": "facets,edges", "ignore": IGNORE})
    facets = {f["id"]: f for f in t.report["facets"]}
    (house_dir / "facets.json").write_text(
        json.dumps(
            [
                {"class": i + 1, "facet_id": f.id}
                | {k: facets[f.id][k] for k in ("pitch_deg", "azimuth_deg", "area_m2", "confidence", "outline")}
                for i, f in enumerate(house.facets)
            ],
            indent=2,
        )
    )
    preview(hid, rgb, labels, row, out / "previews" / f"{hid}.png")

    common = reg.common
    pieces = {
        "h": reg.h[common],
        "r_constant": reg.residual(reg.const)[common],
        "r_relief": reg.residual(reg.pure)[common],
        "type": reg.types[common],
        "kept": q["kept"],
    }
    return row, pieces


def preview(hid: str, rgb: np.ndarray, labels: dict, row: dict, path: Path) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
    img = np.moveaxis(rgb, 0, -1)
    axs[0].imshow(img)
    axs[0].set_title(
        f"{hid}: {row['status']}" + (f" ({', '.join(row['failed'])})" if row["failed"] else ""), fontsize=9
    )
    for ax, name in zip(axs[1:], ("constant", "relief")):
        ax.imshow(img)
        fac, edg = labels[name]
        fac_rgba = np.zeros((*fac.shape, 4))
        cmap = plt.get_cmap("tab10")
        for k in np.unique(fac):
            if k == 0:
                continue
            fac_rgba[fac == k] = (0.5, 0.5, 0.5, 0.45) if k == IGNORE else (*cmap((k - 1) % 10)[:3], 0.35)
        edg_rgba = np.zeros((*edg.shape, 4))
        edg_rgba[(edg > 0) & (edg != IGNORE)] = (1, 1, 1, 0.9)
        ax.imshow(fac_rgba)
        ax.imshow(edg_rgba)
        rms = row.get(f"rms_{'constant' if name == 'constant' else 'pure_relief'}_m")
        ax.set_title(f"labels, {name} shift (edge RMS {rms} m); grey = ignore", fontsize=9)
    for ax in axs:
        ax.set_axis_off()
    fig.text(0.99, 0.01, f"Photo: {ATTRIBUTION}", ha="right", fontsize=7, color="#555")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def summarise_run(rows: list[dict], pooled: list[dict], tile_name: str, args) -> dict:
    from collections import Counter

    status = Counter(r["status"] for r in rows)
    failures = Counter(f for r in rows for f in r["failed"])
    kept = [r for r in rows if r["status"] == "kept"]
    out = {
        "tile": tile_name,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "candidates": len(rows),
        "status": dict(status),
        "failed_checks": dict(failures.most_common()),
        "kept_roof_area_m2": round(sum(r["roof_area_m2"] for r in kept), 1),
        "kept_facets": sum(r["facets"] for r in kept),
    }
    reg_rows = [r for r in rows if r.get("rms_constant_m") is not None and r.get("rms_pure_relief_m") is not None]
    if reg_rows:
        better = sum(r["rms_pure_relief_m"] < r["rms_constant_m"] for r in reg_rows)
        out["model_comparison"] = {
            "houses_registered": len(reg_rows),
            "pure_relief_better": better,
            "constant_better": len(reg_rows) - better,
            "median_rms_constant_m": round(float(np.median([r["rms_constant_m"] for r in reg_rows])), 3),
            "median_rms_pure_relief_m": round(float(np.median([r["rms_pure_relief_m"] for r in reg_rows])), 3),
        }
    if pooled:
        h = np.concatenate([p["h"] for p in pooled])
        rc = np.concatenate([p["r_constant"] for p in pooled])
        rr = np.concatenate([p["r_relief"] for p in pooled])
        bins = {}
        for lo, hi in zip(HEIGHT_BINS[:-1], HEIGHT_BINS[1:]):
            m = (h >= lo) & (h < hi)
            if m.any():
                bins[f"{lo}-{hi} m"] = {
                    "pieces": int(m.sum()),
                    "rms_constant_m": round(float(np.sqrt(np.mean(rc[m] ** 2))), 3),
                    "rms_pure_relief_m": round(float(np.sqrt(np.mean(rr[m] ** 2))), 3),
                }
        out["residual_by_height_all_registered_pieces"] = bins
    return out


if __name__ == "__main__":
    main()
