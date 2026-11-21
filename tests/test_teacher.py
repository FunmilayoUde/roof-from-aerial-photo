"""The in-memory teacher must reproduce the LiDAR project's own report for a house it processed.

Reads the AHN tile (about 15 s), so it is skipped when the LiDAR data isn't present.
"""

import json

import pytest
from shapely.geometry import shape

from roofphoto.data.teacher import READ_MARGIN_M, run_teacher
from roofphoto.io.lidar import house_dir, lidar_root, read_points_multi, tile_path

HOUSE = "delft_03"
have_data = (house_dir(HOUSE) / "report.json").exists() and (lidar_root() / "data" / "raw").exists()


@pytest.mark.skipif(not have_data, reason="LiDAR outputs or tile not present")
def test_reproduces_lidar_project_report():
    ref = json.loads((house_dir(HOUSE) / "report.json").read_text())
    fp = shape(json.loads((house_dir(HOUSE) / "footprint.geojson").read_text())["geometry"])
    pts = read_points_multi(tile_path(HOUSE), {HOUSE: fp.buffer(READ_MARGIN_M).bounds})[HOUSE]
    res = run_teacher(HOUSE, pts, fp, ref["source"]["bag_id"], ref["source"]["tile"])
    assert res.ok
    for key in ("source", "settings", "totals", "facets", "edges"):
        assert res.report[key] == ref[key], key
