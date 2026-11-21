"""Run the LiDAR pipeline (the "teacher") on one house, in memory, from this repo.

The LiDAR project's stage scripts write into its own data/interim/, which this project must not
touch. The scripts are thin glue around roofrecon functions, so this module calls the same
functions with the same settings and returns the results instead of writing files there:

    isolate   building class, footprint + 1.0 m mitred crop, recentre   (scripts/stage1_isolate.py)
    planes    build_roof(): RANSAC 0.12 m, 1 m² minimum, clean-up       (roofrecon.pipeline)
    topology  facet polygons and typed edges                            (roofrecon.pipeline)
    report    measurement_report()                                      (scripts/stage4_report.py)
    check     overhang + footprint estimate D                           (scripts/stage5_validate.py)

One deliberate change, for sparse point clouds: the pipeline's distances that depend on point
spacing (0.3 m connectivity and adjacency, 0.15 m line band, 0.75 m alpha-shape radius, 0.4 m
touch radius) were tuned on Delft data at 25-50 points/m². At Bilthoven's ~15 points/m² the
typical gap to a point's 4th neighbour reaches 0.3 m, the connectivity check cuts roof planes
into fragments, and the median coverage falls to 0.53. So, per house, with spacing = the median
4th-nearest-neighbour distance in plan:

  spacing <= 0.22 m  (the density the LiDAR project validated; the sparsest Delft house is
                      0.216 m): its settings unchanged, so the Delft reports reproduce exactly;
  spacing  > 0.22 m: connectivity radius = 1.8 x spacing, the other spacing-dependent distances
                      scaled by the same factor.

1.8 is the middle of the range that worked (1.6-2.1) in a sweep on 60 Bilthoven houses, checked
on 60 others. Noise-related settings (0.12 m plane threshold) and geometric ones (1 m snap
distance) are not scaled. See docs/stage2_report.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from roofrecon.isolate import crop_mask, density_stats, in_polygon, non_building_inside_walls, recenter
from roofrecon.las_io import PointData
from roofrecon.measure import measurement_report
from roofrecon.pipeline import MIN_AREA_M2, THRESHOLD_M, RoofModel, build_roof
from roofrecon.polygons import roof_polygons
from roofrecon.segment import segment_ransac
from roofrecon.topology import internal_edges
from scipy.spatial import cKDTree
from roofrecon.validate import footprint_estimates, overhang
from shapely.geometry import Polygon

BUILDING = 6
GROUND = 2
CROP_BUFFER_M = 1.0  # the LiDAR pipeline's default, on the plateau of its buffer sweep
READ_MARGIN_M = 9.0  # points read around the walls: crop buffer, ground ring (to 8 m), slack
SPACING_REF_M = 0.22  # 4th-neighbour spacing up to which the pipeline's distances are used as tuned
CONNECT_PER_SPACING = 1.8  # connectivity radius in units of spacing, beyond SPACING_REF_M (swept)


def point_spacing(xy: np.ndarray) -> float:
    """Median distance (plan) to each point's 4th-nearest neighbour: the gap a connectivity
    radius has to bridge in the sparse direction of a scan pattern, not just to the nearest point."""
    return float(np.median(cKDTree(xy).query(xy, k=5)[0][:, 4]))


def spacing_scale(spacing_m: float) -> float:
    """Factor on the pipeline's spacing-dependent distances (1.0 = its validated settings)."""
    if spacing_m <= SPACING_REF_M:
        return 1.0
    return max(1.0, CONNECT_PER_SPACING * spacing_m / 0.3)


def build_roof_scaled(xyz: np.ndarray, density: float, scale: float) -> RoofModel:
    """roofrecon's build_roof, with its spacing-dependent distances multiplied by scale."""
    if scale == 1.0:
        return build_roof(xyz, density)  # exactly the LiDAR project's settings
    min_points = max(3, int(round(MIN_AREA_M2 * density)))
    seg = segment_ransac(xyz, threshold=THRESHOLD_M, min_points=min_points, connect_radius=0.3 * scale, seed=0)
    edges = internal_edges(xyz, seg, radius=0.3 * scale, line_band=0.15 * scale)
    polys = roof_polygons(xyz, seg, edges, alpha_radius=0.75 * scale, touch_radius=0.4 * scale)
    settings = {
        "method": "ransac",
        "threshold_m": THRESHOLD_M,
        "min_area_m2": MIN_AREA_M2,
        "min_points": min_points,
        "density_pts_per_m2": round(density, 2),
        "seed": 0,
        "spacing_scale": round(scale, 3),
    }
    return RoofModel(xyz, seg, edges, polys, settings)


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
    spacing = point_spacing(local[:, :2])
    scale = spacing_scale(spacing)
    stats["offset_xyz"] = offset.tolist()
    stats["density_pts_per_m2"] = density
    stats["spacing_4nn_m"] = round(spacing, 3)
    stats["spacing_scale"] = round(scale, 3)

    meta = {
        "house_id": house_id,
        "bag_id": bag_id,
        "tile": tile_name,
        "crs": "EPSG:28992",
        "offset_xyz": offset.tolist(),
        "footprint_area_m2": round(footprint.area, 1),
    }
    try:
        report = measurement_report(build_roof_scaled(local, density, scale), meta)
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
