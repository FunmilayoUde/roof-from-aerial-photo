"""Run the LiDAR pipeline (the "teacher") on one house, in memory, from this repo.

The LiDAR project's stage scripts write into its own data/interim/, which this project must not
touch. The scripts are thin glue around roofrecon functions, so this module calls the same
functions with the same settings and returns the results instead of writing files there:

    isolate   building class, footprint + 1.0 m mitred crop, recentre   (scripts/stage1_isolate.py)
    planes    build_roof(): RANSAC 0.12 m, 1 m² minimum, clean-up       (roofrecon.pipeline)
    topology  facet polygons and typed edges                            (roofrecon.pipeline)
    report    measurement_report()                                      (scripts/stage4_report.py)
    check     overhang + footprint estimate D                           (scripts/stage5_validate.py)

Run on the four Delft houses, it must reproduce the LiDAR project's own report.json.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from roofrecon.isolate import crop_mask, density_stats, in_polygon, non_building_inside_walls, recenter
from roofrecon.las_io import PointData
from roofrecon.measure import measurement_report
from roofrecon.pipeline import build_roof
from roofrecon.validate import footprint_estimates, overhang
from shapely.geometry import Polygon

BUILDING = 6
GROUND = 2
CROP_BUFFER_M = 1.0  # the LiDAR pipeline's default, on the plateau of its buffer sweep
READ_MARGIN_M = 9.0  # points read around the walls: crop buffer, ground ring (to 8 m), slack


@dataclass
class TeacherResult:
    house_id: str
    ok: bool
    reason: str | None  # why there is no report, if ok is False
    report: dict | None
    stats: dict


def ground_ring_height(pts: PointData, footprint: Polygon, inner_m: float = 2.0, outer_m: float = 8.0) -> dict:
    """Median NAP height of ground points 2-8 m from the walls (relief displacement is relative to it)."""
    ring = footprint.buffer(outer_m).difference(footprint.buffer(inner_m))
    g = pts.xyz[pts.classification == GROUND]
    g = g[in_polygon(g[:, :2], ring)]
    if len(g) == 0:
        return {"ground_z_nap_m": None, "ground_points": 0}
    return {
        "ground_z_nap_m": float(np.median(g[:, 2])),
        "ground_z_iqr_m": float(np.subtract(*np.percentile(g[:, 2], [75, 25]))),
        "ground_points": int(len(g)),
        "ring_m": [inner_m, outer_m],
    }


def run_teacher(house_id: str, pts: PointData, footprint: Polygon, bag_id: str, tile_name: str) -> TeacherResult:
    """pts: every point within READ_MARGIN_M of the walls, all classes, in EPSG:28992."""
    xy = pts.xyz[:, :2]
    stats: dict = {"house_id": house_id, "footprint_area_m2": float(footprint.area)}
    stats["ground"] = ground_ring_height(pts, footprint)

    nonb = non_building_inside_walls(xy, pts.classification, footprint)
    stats["non_building_inside_walls"] = {"total": nonb["total"], "interior": nonb["interior"]}

    building = pts.subset(pts.classification == BUILDING)
    roof = building.subset(crop_mask(building.xyz[:, :2], footprint, CROP_BUFFER_M))
    stats["roof_points"] = len(roof)
    if len(roof) < 100:
        return TeacherResult(house_id, False, f"only {len(roof)} building points in the crop", None, stats)

    local, offset = recenter(roof.xyz)
    in_walls = local[crop_mask(roof.xyz[:, :2], footprint, 0.0)]
    density = density_stats(in_walls[:, :2], footprint.area)["nominal_pts_per_m2"]
    stats["offset_xyz"] = offset.tolist()
    stats["density_pts_per_m2"] = density

    meta = {
        "house_id": house_id,
        "bag_id": bag_id,
        "tile": tile_name,
        "crs": "EPSG:28992",
        "offset_xyz": offset.tolist(),
        "footprint_area_m2": round(footprint.area, 1),
    }
    try:
        report = measurement_report(build_roof(local, density), meta)
    except Exception as exc:  # the pipeline was built on four houses; record what breaks on the rest
        return TeacherResult(house_id, False, f"pipeline error: {type(exc).__name__}: {exc}", None, stats)
    if not report["facets"]:
        return TeacherResult(house_id, False, "no facets found", report, stats)

    # Footprint consistency check (estimate D of the LiDAR project's stage 5): roof area from the
    # BAG outline + measured overhang and the facets' pitches. Tests consistency, not truth.
    counted = [f for f in report["facets"] if f["area_m2"] > 0]
    oh = overhang(local[:, :2], Polygon(np.asarray(footprint.exterior.coords) - offset[:2]))
    est = footprint_estimates(footprint, counted, oh["overhang_m"], oh["points_outside"] / density)
    area = report["totals"]["roof_area_m2"]
    stats["overhang_m"] = round(oh["overhang_m"], 3)
    stats["footprint_check_D_rel"] = round(est["D_footprint_plus_overhang_mean_secant"] / area - 1, 4) if area else None
    return TeacherResult(house_id, True, None, report, stats)
