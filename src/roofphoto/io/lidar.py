"""Read the LiDAR pipeline's per-house outputs (read-only) and put them in world coordinates.

The pipeline stores roof geometry in *local* coordinates: it subtracts a whole-metre offset
(report.json -> source.offset_xyz) so values stay small enough for float32. Adding it back gives
RD New (EPSG:28992) x, y and NAP height z, the same system as the aerial photo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    holes_xyz: list[np.ndarray] = field(default_factory=list)  # world, e.g. curved parts left out


@dataclass
class Edge:
    id: int
    type: str
    p0: np.ndarray  # (3,) world
    p1: np.ndarray  # (3,) world
    facets: tuple[int, ...] = ()  # report facet ids on either side (one for an outer edge)


@dataclass
class HouseLabels:
    house_id: str
    offset_xyz: np.ndarray
    footprint: Polygon  # BAG outline (walls), world xy
    facets: list[Facet]
    edges: list[Edge]


def load_house(house_id: str) -> HouseLabels:
    """A house the LiDAR project processed (its data/interim/<house_id>/)."""
    d = house_dir(house_id)
    report_path = d / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"{report_path} missing: the LiDAR pipeline has no report for {house_id}")
    feature = json.loads((d / "footprint.geojson").read_text())
    return labels_from_report(json.loads(report_path.read_text()), shape(feature["geometry"]))


def labels_from_report(report: dict, footprint: Polygon) -> HouseLabels:
    """World-coordinate labels from a roofrecon report.json (as a dict) and the BAG outline."""
    if report["source"]["crs"] != CRS:
        raise ValueError(f"{report['house_id']}: expected {CRS}, report says {report['source']['crs']}")
    offset = np.asarray(report["source"]["offset_xyz"], dtype=float)
    facets = [
        Facet(
            id=f["id"],
            polygon_xyz=local_to_world(f["vertices_local_m"], offset),
            pitch_deg=f["pitch_deg"],
            area_m2=f["area_m2"],
            confidence=f["confidence"],
            fallback=f["fallback_reason"] is not None,
            holes_xyz=[local_to_world(h, offset) for h in f.get("holes_local_m", [])],
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
            facets=tuple(e["facets"]),
        )
        for e in report["edges"]
    ]
    return HouseLabels(report["house_id"], offset, footprint, facets, edges)


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


def tile_path(house_id: str) -> Path:
    stats = json.loads((house_dir(house_id) / "stats.json").read_text())
    return lidar_root() / stats["tile"]["path"]


def read_ground_intensity(
    tile: Path, bboxes: dict[str, tuple[float, float, float, float]], chunk_size: int = 2_000_000
) -> dict[str, np.ndarray]:
    """Ground-class points (x, y, z, intensity) inside each bbox, from one pass over the tile.

    roofrecon's reader drops intensity, so this reads the tile with laspy directly. One pass
    serves every house: a tile is tens of millions of points and reading it dominates the cost.
    """
    import laspy

    parts: dict[str, list[np.ndarray]] = {k: [] for k in bboxes}
    with laspy.open(tile) as f:
        for chunk in f.chunk_iterator(chunk_size):
            ground = np.asarray(chunk.classification) == 2
            if not ground.any():
                continue
            x = np.asarray(chunk.x, dtype=np.float64)[ground]
            y = np.asarray(chunk.y, dtype=np.float64)[ground]
            for key, (xmin, ymin, xmax, ymax) in bboxes.items():
                m = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
                if m.any():
                    z = np.asarray(chunk.z, dtype=np.float64)[ground][m]
                    inten = np.asarray(chunk.intensity, dtype=np.float64)[ground][m]
                    parts[key].append(np.column_stack([x[m], y[m], z, inten]))
    return {k: (np.concatenate(v) if v else np.empty((0, 4))) for k, v in parts.items()}


def read_points_multi(
    tile: Path, bboxes: dict[str, tuple[float, float, float, float]], chunk_size: int = 2_000_000
) -> dict:
    """All points (every class) inside each bbox, from one pass over the tile, as roofrecon PointData.

    roofrecon's read_points scans the whole tile for one bbox; for hundreds of houses that is
    hundreds of scans. This reads the tile once and hands each house its points.
    """
    import laspy
    from roofrecon.las_io import PointData, header_summary

    keys = list(bboxes)
    boxes = np.array([bboxes[k] for k in keys])
    parts: dict[str, list[tuple]] = {k: [] for k in keys}
    with laspy.open(tile) as f:
        for chunk in f.chunk_iterator(chunk_size):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            # Skip boxes that miss this chunk's extent entirely (chunks are spatially coherent in AHN).
            near = np.nonzero(
                (boxes[:, 0] <= x.max())
                & (boxes[:, 2] >= x.min())
                & (boxes[:, 1] <= y.max())
                & (boxes[:, 3] >= y.min())
            )[0]
            if not len(near):
                continue
            z = cls = rn = nr = None
            for i in near:
                xmin, ymin, xmax, ymax = boxes[i]
                m = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
                if not m.any():
                    continue
                if z is None:
                    z = np.asarray(chunk.z, dtype=np.float64)
                    cls = np.asarray(chunk.classification, dtype=np.uint8)
                    rn = np.asarray(chunk.return_number, dtype=np.uint8)
                    nr = np.asarray(chunk.number_of_returns, dtype=np.uint8)
                parts[keys[i]].append((np.column_stack([x[m], y[m], z[m]]), cls[m], rn[m], nr[m]))
    header = header_summary(tile)
    out = {}
    for k, p in parts.items():
        if p:
            out[k] = PointData(*(np.concatenate([q[j] for q in p]) for j in range(4)), header)
        else:
            e = np.empty(0, dtype=np.uint8)
            out[k] = PointData(np.empty((0, 3)), e, e, e, header)
    return out
