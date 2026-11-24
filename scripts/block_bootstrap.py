"""Robustness of the paper's model-vs-reference comparisons to spatial dependence between houses.

    uv run python scripts/block_bootstrap.py

For every comparison in the paper's Tables 1, 5 and 6, three 95% intervals for the difference in
median |area error| (model minus reference, percentage points):
  house     resampling houses (as in the paper so far; reproduces its intervals)
  block250  resampling 250 m x 250 m grid cells (by house centroid), a neighbourhood scale
  subtile   resampling whole 1 x 1.25 km sub-tiles (9 in Bilthoven, 5 in Heemstede)
plus the house-level sign test, and the number of sub-tiles on which the model's median error is
lower with an exact sign test over sub-tiles. With 5-9 sub-tiles the sub-tile percentile interval
is unreliable (too narrow); the sub-tile sign test is the conservative check at that scale.
Medians are given to two decimals. Writes outputs/block_bootstrap/summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from roofphoto.data.lookup import footprints
from roofphoto.results import abs_err, bilthoven_area, heemstede_untrained, two_town_area
from roofphoto.stats import compare, compare_blocked

ROOT = Path(__file__).resolve().parents[1]
CELL_M = 250.0


def cells(house_ids: np.ndarray, tiles: np.ndarray) -> np.ndarray:
    fps = {}
    for t in np.unique(tiles):
        fps.update(footprints(ROOT, t))
    out = []
    for h in house_ids:
        c = fps[h].centroid
        out.append(f"{int(c.x // CELL_M)}_{int(c.y // CELL_M)}")
    return np.array(out)


def evaluate(model_err, base_err, tiles, cell_ids, model_name: str, ref_name: str) -> dict:
    h = compare(model_err, base_err)
    c = compare_blocked(model_err, base_err, cell_ids)
    s = compare_blocked(model_err, base_err, tiles)
    return {
        "houses": len(model_err),
        f"{model_name}_median_pct": round(100 * float(np.median(model_err)), 2),
        f"{ref_name}_median_pct": round(100 * float(np.median(base_err)), 2),
        "diff_pts": s["median_diff_pts"],
        "ci95_house": h["ci95_pts"],
        "ci95_block250": c["ci95_pts"],
        "cells_250m": c["blocks"],
        "ci95_subtile": s["ci95_pts"],
        "subtiles": s["blocks"],
        "subtile_sizes": s["block_sizes"],
        "model_better_subtiles": s["model_better_blocks"],
        "subtile_sign_test_p_two_sided": s["block_sign_test_p_two_sided"],
        "model_better_houses": h["model_better_houses"],
        "sign_test_p_two_sided": h["sign_test_p_two_sided"],
    }


def main() -> None:
    out_dir = ROOT / "outputs/block_bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}

    b = bilthoven_area(ROOT)
    cb = cells(b["house_id"], b["tile"])
    em, eb = abs_err(b["model"], b["truth"]), abs_err(b["b1"], b["truth"])
    clean = b["unc"] <= 0.10
    res["T1 Bilthoven, model vs footprint x k"] = evaluate(em, eb, b["tile"], cb, "model", "reference")
    res["T1 Bilthoven clean, model vs footprint x k"] = evaluate(
        em[clean], eb[clean], b["tile"][clean], cb[clean], "model", "reference"
    )

    h = heemstede_untrained(ROOT)
    ch = cells(h["house_id"], h["tile"])
    em, eb = abs_err(h["model"], h["truth"]), abs_err(h["b1"], h["truth"])
    clean = h["unc"] <= 0.10
    res["T5 Heemstede untrained, model vs footprint x k"] = evaluate(em, eb, h["tile"], ch, "model", "reference")
    res["T5 Heemstede untrained clean, model vs footprint x k"] = evaluate(
        em[clean], eb[clean], h["tile"][clean], ch[clean], "model", "reference"
    )

    for town, label in (("heemstede", "Heemstede"), ("bilthoven", "Bilthoven")):
        t = two_town_area(ROOT, town)
        ct = cells(t["house_id"], t["tile"])
        e2 = abs_err(t["two_town"], t["truth"])
        res[f"T6 {label}, two-town vs footprint x k"] = evaluate(
            e2, abs_err(t["two_town_b1"], t["truth"]), t["tile"], ct, "two_town", "reference"
        )
        res[f"T6 {label}, two-town vs Bilthoven-only"] = evaluate(
            e2, abs_err(t["bilthoven_only"], t["truth"]), t["tile"], ct, "two_town", "bilthoven_only"
        )

    (out_dir / "summary.json").write_text(json.dumps(res, indent=2))
    hdr = (
        f"{'comparison':52} {'diff':>6}  {'house CI':>15}  {'250 m CI':>15}  {'sub-tile CI':>15}"
        "  tiles  p(tiles)  p(houses)"
    )
    print(hdr)
    for k, r in res.items():

        def f(ci):
            return f"{ci[0]:+.2f} to {ci[1]:+.2f}"

        print(
            f"{k:52} {r['diff_pts']:+6.2f}  {f(r['ci95_house']):>15}  {f(r['ci95_block250']):>15}  "
            f"{f(r['ci95_subtile']):>15}  {r['model_better_subtiles']:>5}  "
            f"{r['subtile_sign_test_p_two_sided']:>8}  {r['sign_test_p_two_sided']}"
        )


if __name__ == "__main__":
    main()
