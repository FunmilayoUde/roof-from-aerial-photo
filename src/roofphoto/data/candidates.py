"""Detached-house candidates in a tile, from BAG building outlines.

The BAG query is the LiDAR project's (scripts/find_candidates.py: bag_buildings_in_bbox), loaded
from its scripts folder rather than copied. The selection criteria match that script's defaults:
residential, one dwelling, 70-300 m² footprint, at least 3 m from any other building, and fully
inside the tile with 5 m to spare. Row houses are excluded because their roofs run on across party
walls that the point cloud doesn't show (see the LiDAR README).
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Polygon, box, shape
from shapely.strtree import STRtree

from roofphoto.io.lidar import lidar_root


def _lidar_script(name: str):
    path = lidar_root() / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"lidar_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bag_buildings(bounds: tuple[float, float, float, float], cache: Path) -> list[dict]:
    """All BAG buildings in bounds (EPSG:28992), cached as GeoJSON features."""
    if cache.exists():
        return json.loads(cache.read_text())
    feats = _lidar_script("find_candidates").bag_buildings_in_bbox(bounds)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(feats))
    return feats


@dataclass
class Candidate:
    bag_id: str
    footprint: Polygon
    gap_m: float
    concavity: float  # 1 - area / convex hull area: 0 for a rectangle, higher for L/T plans
    built: int | None


def find_candidates(
    feats: list[dict],
    tile_bounds: tuple[float, float, float, float],
    min_area: float = 70.0,
    max_area: float = 300.0,
    min_gap: float = 3.0,
) -> list[Candidate]:
    tile = box(*tile_bounds)
    polys = [shape(f["geometry"]) for f in feats]
    tree = STRtree(polys)
    out = []
    for i, (f, p) in enumerate(zip(feats, polys)):
        pr = f["properties"]
        if not (
            "woonfunctie" in (pr.get("gebruiksdoel") or "")
            and pr.get("aantal_verblijfsobjecten") == 1
            and min_area <= p.area <= max_area
            and tile.contains(p.buffer(5))
        ):
            continue
        gaps = [polys[j].distance(p) for j in tree.query(p.buffer(30)) if j != i]
        gap = min(gaps) if gaps else 30.0
        if gap < min_gap:
            continue
        out.append(Candidate(pr["identificatie"], p, gap, 1 - p.area / p.convex_hull.area, pr.get("bouwjaar")))
    return sorted(out, key=lambda c: c.bag_id)
