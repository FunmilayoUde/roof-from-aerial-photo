"""Look up Stage 2 house records: BAG outlines (via the private id map) and manifest rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from shapely.geometry import Polygon, shape


def footprints(root: Path, tile: str) -> dict[str, Polygon]:
    """house id -> BAG outline, for every candidate in a tile."""
    with open(root / "data/processed/stage2" / tile / "houses_private.csv") as fh:
        bag = {r["house_id"]: r["bag_id"] for r in csv.DictReader(fh)}
    feats = {
        f["properties"]["identificatie"]: f for f in json.loads((root / "data/raw/bag" / f"{tile}.json").read_text())
    }
    return {h: shape(feats[b]["geometry"]) for h, b in bag.items()}


def manifest(root: Path, tile: str) -> dict[str, dict]:
    with open(root / "outputs/stage2" / tile / "manifest.csv") as fh:
        return {r["house_id"]: r for r in csv.DictReader(fh)}
