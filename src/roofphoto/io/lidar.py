"""Read the LiDAR pipeline's per-house outputs (read-only) and put them in world coordinates.

The pipeline stores roof geometry in *local* coordinates: it subtracts a whole-metre offset
(report.json -> source.offset_xyz) so values stay small enough for float32. Adding it back gives
RD New (EPSG:28992) x, y and NAP height z, the same system as the aerial photo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import roofrecon
from shapely.geometry import Polygon, shape

CRS = "EPSG:28992"


def lidar_root() -> Path:
    """The LiDAR project folder, found from the installed (editable) roofrecon package."""
    return Path(roofrecon.__file__).resolve().parents[2]


def house_dir(house_id: str) -> Path:
    return lidar_root() / "data" / "interim" / house_id


def local_to_world(xyz_local: np.ndarray, offset_xyz: np.ndarray) -> np.ndarray:
    """Undo the pipeline's recentering: world = local + offset."""
    return np.asarray(xyz_local, dtype=float) + np.asarray(offset_xyz, dtype=float)


@dataclass
class Facet:
    id: int
    polygon_xyz: np.ndarray  # (N, 3) world coordinates, ring not closed
    pitch_deg: float
    area_m2: float
    confidence: str
    fallback: bool  # traced outline rather than plane and line intersections


@dataclass
class Edge:
    id: int
    type: str
    p0: np.ndarray  # (3,) world
    p1: np.ndarray  # (3,) world


@dataclass
class HouseLabels:
    house_id: str
    offset_xyz: np.ndarray
    footprint: Polygon  # BAG outline (walls), world xy
    facets: list[Facet]
    edges: list[Edge]


def load_house(house_id: str) -> HouseLabels:
    d = house_dir(house_id)
    report_path = d / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"{report_path} missing: the LiDAR pipeline has no report for {house_id}")
    report = json.loads(report_path.read_text())
    if report["source"]["crs"] != CRS:
        raise ValueError(f"{house_id}: expected {CRS}, report says {report['source']['crs']}")
    offset = np.asarray(report["source"]["offset_xyz"], dtype=float)

    feature = json.loads((d / "footprint.geojson").read_text())
    footprint = shape(feature["geometry"])

    facets = [
        Facet(
            id=f["id"],
            polygon_xyz=local_to_world(f["vertices_local_m"], offset),
            pitch_deg=f["pitch_deg"],
            area_m2=f["area_m2"],
            confidence=f["confidence"],
            fallback=f["fallback_reason"] is not None,
        )
        for f in report["facets"]
        if f["vertices_local_m"]
    ]
    edges = [
        Edge(
            id=e["id"],
            type=e["type"],
            p0=local_to_world(e["p0_local_m"], offset),
            p1=local_to_world(e["p1_local_m"], offset),
        )
        for e in report["edges"]
    ]
    return HouseLabels(house_id, offset, footprint, facets, edges)


def ground_height(house_id: str, footprint: Polygon, inner_m: float = 2.0, outer_m: float = 8.0) -> dict:
    """Median NAP height of ground-class points in a ring around the house.

    Relief displacement grows with height above the ground surface the orthophoto was
    corrected to, so roof z must be compared with the local ground, not with NAP zero.
    Reads the AHN tile named in the house's stats.json (a chunked scan of the whole tile).
    """
    from roofrecon.las_io import read_points

    stats = json.loads((house_dir(house_id) / "stats.json").read_text())
    tile = lidar_root() / stats["tile"]["path"]
    ring = footprint.buffer(outer_m).difference(footprint.buffer(inner_m))
    pts = read_points(tile, bbox=ring.bounds)
    ground = pts.xyz[pts.classification == 2]
    from roofrecon.isolate import in_polygon

    ground = ground[in_polygon(ground[:, :2], ring)]
    if len(ground) == 0:
        raise ValueError(f"{house_id}: no ground points within {outer_m} m of the walls")
    return {
        "ground_z_nap_m": float(np.median(ground[:, 2])),
        "ground_z_iqr_m": float(np.subtract(*np.percentile(ground[:, 2], [75, 25]))),
        "ground_points": int(len(ground)),
        "ring_m": [inner_m, outer_m],
    }
