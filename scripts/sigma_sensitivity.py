"""How much do the labels depend on the step in the teacher's spacing rule?

    uv run python scripts/sigma_sensitivity.py

The teacher keeps the LiDAR project's settings up to a 4th-neighbour spacing of 0.22 m (so the
Delft reports reproduce exactly) and uses a connectivity radius of 1.8 x spacing above it, a step
from 1.36 x to 1.8 x spacing at 0.22 m. The continuous alternative is radius = max(0.3 m, 1.8 x
spacing), which differs only for spacing in (0.167, 0.22] m. This re-runs the teacher with that
rule on every Stage 2 house in that range and compares its roof area and plane coverage with the
labels actually used. Then, with the models unchanged, re-scores the paper's area comparisons with
those houses' truth replaced by the continuous rule's roof area (--eval-only skips the re-run).
Writes outputs/sigma_sensitivity/houses.csv and summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import roofphoto.data.teacher as teacher
from roofphoto.data.lookup import footprints, manifest
from roofphoto.io.lidar import read_points_multi
from roofphoto.results import abs_err, bilthoven_area, heemstede_untrained, two_town_area
from roofphoto.stats import compare

ROOT = Path(__file__).resolve().parents[1]
TILES = ["32CN1_03", "32CN1_04", "32CN1_05", "32CN1_08", "32CN1_09", "32CN1_10", "32CN1_13", "32CN1_15", "32CN1_20",
         "25CN1_01", "25CN1_02", "25CN1_06", "25CN1_07", "25CN1_13"]  # fmt: skip
LO, HI = 0.3 / teacher.CONNECT_PER_SPACING, teacher.SPACING_REF_M


def continuous(spacing_m: float) -> float:
    return max(1.0, teacher.CONNECT_PER_SPACING * spacing_m / 0.3)


def coverage(report: dict) -> float | None:
    q = report.get("quality", {})
    for k in ("coverage", "roof_points_in_facets_share", "share_of_roof_points_in_planes"):
        if k in q:
            return q[k]
    return None


def rescore(new_area: dict[str, float]) -> dict:
    """The paper's comparisons, with truth as used and with the continuous rule's roof area."""

    def both(name, d, model_key, ref_key):
        out = {}
        for label, truth in (
            ("as_used", d["truth"]),
            ("continuous", np.array([new_area.get(h, t) for h, t in zip(d["house_id"], d["truth"])])),
        ):
            em, eb = abs_err(d[model_key], truth), abs_err(d[ref_key], truth)
            c = compare(em, eb)
            out[label] = {
                "model_median_pct": round(100 * float(np.median(em)), 2),
                "reference_median_pct": round(100 * float(np.median(eb)), 2),
                "diff_pts": c["median_diff_pts"],
                "ci95_house": c["ci95_pts"],
                "model_better": c["model_better_houses"],
                "p_two_sided": c["sign_test_p_two_sided"],
            }
        out["houses_with_changed_truth"] = int(sum(h in new_area for h in d["house_id"]))
        return {name: out}

    res = {}
    res |= both("T1 Bilthoven, model vs footprint x k", bilthoven_area(ROOT), "model", "b1")
    res |= both("T5 Heemstede untrained, model vs footprint x k", heemstede_untrained(ROOT), "model", "b1")
    for town in ("heemstede", "bilthoven"):
        t = two_town_area(ROOT, town)
        res |= both(f"T6 {town}, two-town vs footprint x k", t, "two_town", "two_town_b1")
        res |= both(f"T6 {town}, two-town vs Bilthoven-only", t, "two_town", "bilthoven_only")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-only", action="store_true", help="reuse houses.csv, only re-score")
    args = ap.parse_args()
    out_dir = ROOT / "outputs/sigma_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        rows = list(csv.DictReader(open(out_dir / "houses.csv")))
        new_area = {r["house_id"]: float(r["area_new_m2"]) for r in rows if r["kept"] == "True"}
        summary = json.loads((out_dir / "summary.json").read_text())
        summary["rescored"] = rescore(new_area)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary["rescored"], indent=1))
        return
    rows = []
    for tile in TILES:
        proc = ROOT / "data/processed/stage2" / tile
        man = manifest(ROOT, tile)
        todo = {}
        for hid in man:
            tj = proc / hid / "teacher.json"
            if not tj.exists():
                continue
            s = json.loads(tj.read_text()).get("spacing_4nn_m")
            if s is not None and LO < s <= HI and (proc / hid / "report.json").exists():
                todo[hid] = s
        if not todo:
            continue
        fps = footprints(ROOT, tile)
        with open(proc / "houses_private.csv") as fh:
            bag = {r["house_id"]: r["bag_id"] for r in csv.DictReader(fh)}
        pts = read_points_multi(
            ROOT / "data/raw/ahn5" / f"{tile}.LAZ",
            {h: fps[h].buffer(teacher.READ_MARGIN_M).bounds for h in todo},
        )
        original = teacher.spacing_scale
        teacher.spacing_scale = continuous
        try:
            for hid, s in todo.items():
                old = json.loads((proc / hid / "report.json").read_text())
                new = teacher.run_teacher(hid, pts[hid], fps[hid], bag[hid], tile)
                a0 = old["totals"]["roof_area_m2"]
                a1 = new.report["totals"]["roof_area_m2"] if new.ok else float("nan")
                rows.append(
                    {
                        "tile": tile,
                        "house_id": hid,
                        "kept": man[hid]["status"] == "kept",
                        "spacing_m": s,
                        "scale_new": round(continuous(s), 3),
                        "area_old_m2": a0,
                        "area_new_m2": a1,
                        "area_change_pct": round(100 * (a1 / a0 - 1), 2) if a0 else float("nan"),
                        "facets_old": old["totals"]["facets"],
                        "facets_new": new.report["totals"]["facets"] if new.ok else None,
                        "coverage_old": coverage(old),
                        "coverage_new": coverage(new.report) if new.ok else None,
                    }
                )
        finally:
            teacher.spacing_scale = original
        print(f"{tile}: {len(todo)} houses", flush=True)

    with (out_dir / "houses.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def summ(sel):
        ch = np.array([r["area_change_pct"] for r in sel], dtype=float)
        ch = ch[np.isfinite(ch)]
        return {
            "houses": len(sel),
            "teacher_failed_with_new_rule": int(sum(not np.isfinite(r["area_new_m2"]) for r in sel)),
            "median_abs_area_change_pct": round(float(np.median(np.abs(ch))), 2),
            "p90_abs_area_change_pct": round(float(np.percentile(np.abs(ch), 90)), 2),
            "share_within_1pct": round(float(np.mean(np.abs(ch) <= 1)), 3),
            "share_within_5pct": round(float(np.mean(np.abs(ch) <= 5)), 3),
            "facet_count_changed": int(sum(r["facets_old"] != r["facets_new"] for r in sel)),
        }

    summary = {
        "spacing_range_m": [round(LO, 4), HI],
        "all": summ(rows),
        "kept": summ([r for r in rows if r["kept"]]),
        "rescored": rescore({r["house_id"]: r["area_new_m2"] for r in rows if r["kept"]}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
