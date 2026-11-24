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
from roofphoto.data.quality import (  # noqa: E402
    registration_checks,
    relief_reliable,
    repeat_checks,
    repeat_diffs,
    summarise,
    teacher_checks,
)
from roofphoto.data.teacher import READ_MARGIN_M, run_teacher  # noqa: E402
from roofphoto.geo.register import register_house  # noqa: E402
from roofphoto.io.lidar import labels_from_report, read_points_multi  # noqa: E402
from roofphoto.io.pdok import ATTRIBUTION, fetch_ortho, write_geotiff  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HEIGHT_BINS = [0, 4, 6, 8, 99]  # m above ground, for pooling residuals by height


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tile", type=Path, required=True, help="AHN sub-tile .LAZ to process")
    ap.add_argument("--layer", default="2023_orthoHR")
    ap.add_argument("--res", type=float, default=0.08)
    ap.add_argument("--chip-m", type=float, default=40.96, help="chip side, m (512 px at 8 cm)")
    ap.add_argument("--min-gap", type=float, default=3.0)
    ap.add_argument("--min-area", type=float, default=70.0)
    ap.add_argument("--max-area", type=float, default=300.0)
    ap.add_argument("--limit", type=int, help="process only the first N candidates (for trial runs)")
    ap.add_argument(
        "--photos-only",
        nargs=4,
        type=float,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="only download the photo chips for houses in this sub-tile core (no LAZ needed yet)",
    )
    ap.add_argument("--overlap", type=float, default=25.0, help="with --photos-only: the sub-tile overlap, m")
    args = ap.parse_args()
    if args.photos_only:
        prefetch(args)
        return

    tile_name = args.tile.stem
    proc = ROOT / "data" / "processed" / "stage2" / tile_name
    out = ROOT / "outputs" / "stage2" / tile_name
    (out / "previews").mkdir(parents=True, exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)

    h = header_summary(args.tile)
    bounds = (h["mins"][0], h["mins"][1], h["maxs"][0], h["maxs"][1])
    # A sub-tile cut by crop_sheet.py carries its core bounds; houses are assigned by centroid.
    sidecar = args.tile.with_suffix(".json")
    core = tuple(json.loads(sidecar.read_text())["core_bounds"]) if sidecar.exists() else None
    feats = bag_buildings(bounds, ROOT / "data" / "raw" / "bag" / f"{tile_name}.json")
    cands = find_candidates(feats, bounds, args.min_area, args.max_area, args.min_gap, core)[: args.limit]
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
        try:
            row, pieces = process_house(hid, c, pts.pop(hid), all_polys, tile_name, proc / hid, out, args)
        except Exception as exc:  # one bad house must not stop the tile; record it and go on
            row = {"house_id": hid, "status": "error", "failed": [f"{type(exc).__name__}: {exc}"]}
            pieces = None
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


def prefetch(args) -> None:
    from concurrent.futures import ThreadPoolExecutor

    core = tuple(args.photos_only)
    o = args.overlap
    bounds = (core[0] - o, core[1] - o, core[2] + o, core[3] + o)
    feats = bag_buildings(bounds, ROOT / "data" / "raw" / "bag" / f"{args.tile.stem}.json")
    cands = find_candidates(feats, bounds, args.min_area, args.max_area, args.min_gap, core)[: args.limit]
    print(f"{args.tile.stem}: {len(cands)} candidates; fetching 2 photo renderings each")
    with ThreadPoolExecutor(4) as pool:  # a few parallel requests; PDOK is a shared public service
        list(pool.map(lambda c: fetch_chip_photos(c, args), cands))
    print(f"{args.tile.stem}: done")


def chip_for(cand, args):
    """Chip grid, its box, and the two photo paths. Photos are keyed by the chip's grid position,
    not by house number, so a cached photo always belongs to the place it shows."""
    from shapely.geometry import box

    c = cand.footprint.centroid
    chip_t, n = chip_grid((c.x, c.y), args.chip_m, args.res)
    chip_box = box(chip_t.c, chip_t.f - n * args.res, chip_t.c + n * args.res, chip_t.f)
    key = f"{round(chip_t.c / args.res)}_{round(chip_t.f / args.res)}_{n}"
    folder = ROOT / "data" / "raw" / args.layer / "chips"
    return chip_t, n, chip_box, folder / f"{key}.tif", folder / f"{key}_b.tif"


def fetch_chip_photos(cand, args) -> None:
    """Both renderings of a house's chip (the second offset by 7 x 5 px; see process_house)."""
    from shapely.affinity import translate

    _, _, chip_box, path_a, path_b = chip_for(cand, args)
    if not path_a.exists():
        fetch_ortho(args.layer, chip_box.bounds, args.res, path_a)
    if not path_b.exists():
        fetch_ortho(args.layer, translate(chip_box, 7 * args.res, 5 * args.res).bounds, args.res, path_b)


def process_house(hid, cand, pts, all_polys, tile_name, house_dir: Path, out: Path, args) -> tuple[dict, dict | None]:
    import shutil

    from shapely.geometry import shape

    # Start from an empty folder: a house that fails earlier than in a previous run must not
    # keep that run's labels. (Photos are cached elsewhere, under data/raw.)
    if house_dir.exists():
        shutil.rmtree(house_dir)
    house_dir.mkdir(parents=True)
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
    chip_t, n, chip_box, photo_path, path_b = chip_for(cand, args)
    if not chip_box.contains(cand.footprint.buffer(6.0)):
        row |= {"status": "too large for chip", "failed": ["chip_size"]}
        return row, None
    fetch_chip_photos(cand, args)
    with rasterio.open(photo_path) as src:
        rgb, photo_t = src.read(), src.transform
    if not photo_t.almost_equals(chip_t) or rgb.shape[1:] != (n, n):
        raise RuntimeError(f"{hid}: photo grid does not match the chip grid")

    # 3. Registration, on two JPEG renderings. PDOK encodes each request separately, and the
    #    differences sit on edges, where registration looks (docs/stage2_report.md). A second
    #    request offset by 7 x 5 px (not a multiple of JPEG's 8 px blocks) gives an independent
    #    rendering of the same pixels; the two fits' disagreement measures the registration's
    #    uncertainty, and their mean halves that noise for the labels.
    reg = register_house(house, rgb, photo_t, gz)
    with rasterio.open(path_b) as src:
        reg_b = register_house(house, src.read(), src.transform, gz)

    # 4. Quality.
    checks = teacher_checks(t.report, t.stats) + registration_checks(reg) + repeat_checks(reg, reg_b)
    q = summarise(checks)
    (house_dir / "quality.json").write_text(json.dumps(q, indent=2))
    row |= {"status": "kept" if q["kept"] else "dropped", "failed": q["failed"]}
    row |= {k: v["value"] for k, v in q["checks"].items()}
    if reg is None or reg_b is None:
        return row, None
    s, s_b = reg.summary(), reg_b.summary()
    s0 = (reg.const.s0 + reg_b.const.s0) / 2
    v = (reg.pure.v + reg_b.pure.v) / 2
    (house_dir / "registration.json").write_text(
        json.dumps(
            {
                "ground_z_nap_m": gz,
                "label_shift": {"constant_s0_m": s0.round(3).tolist(), "relief_v": v.round(4).tolist()},
            }
            | {"rendering_a": s, "rendering_b": s_b},
            indent=2,
        )
    )

    def mean_rms(key: str) -> float | None:
        vals = [x["model_comparison_same_pieces"][key] for x in (s, s_b)]
        return None if None in vals else round(sum(vals) / 2, 3)

    rd = repeat_diffs(reg, reg_b)
    row |= {
        "repeat_diff_relief_m": round(rd["relief_m"], 4),
        "relief_labels_reliable": relief_reliable(reg, reg_b),
        "shift_e_m": round(float(s0[0]), 3),
        "shift_n_m": round(float(s0[1]), 3),
        "rms_constant_m": mean_rms("rms_constant_m"),
        "rms_pure_relief_m": mean_rms("rms_pure_relief_m"),
        "v_e": round(float(v[0]), 4),
        "v_n": round(float(v[1]), 4),
    }

    # 5. Labels, once per correction model, plus the photo chip.
    neighbours = [shape(g) for bid, g in all_polys if bid != cand.bag_id]
    near = [p for p in neighbours if p.intersects(chip_box)]
    ignore = neighbour_ignore(near) if near else None
    labels = {
        "constant": rasterize_labels(house, constant_shift(s0), photo_t, n, ignore),
        "relief": rasterize_labels(house, relief_shift(v, gz), photo_t, n, ignore),
    }
    write_geotiff(house_dir / "photo.tif", rgb, photo_t, tags={"layer": args.layer, "attribution": ATTRIBUTION})
    for name, lab in labels.items():
        write_geotiff(
            house_dir / f"labels_{name}.tif", lab, photo_t, tags={"bands": "facets,edges,roof", "ignore": IGNORE}
        )
    facets = {f["id"]: f for f in t.report["facets"]}
    (house_dir / "facets.json").write_text(
        json.dumps(
            [
                {"class": i + 1, "facet_id": f.id, "ignored_in_labels": f.confidence == "low"}
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
        fac, edg = labels[name][:2]
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
    # Only houses whose registrations repeat on both renderings, for both models: elsewhere the
    # comparison is decided by JPEG noise (docs/stage2_report.md).
    reg_rows = [
        r
        for r in rows
        if r.get("rms_constant_m") is not None
        and r.get("rms_pure_relief_m") is not None
        and r.get("relief_labels_reliable")
        and "repeat_diff_constant_m" not in r["failed"]
    ]
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
