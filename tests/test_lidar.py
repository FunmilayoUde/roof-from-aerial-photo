import numpy as np
import pytest

from roofphoto.io.lidar import house_dir, load_house, local_to_world


def test_local_to_world_adds_offset():
    local = np.array([[-5.749, -0.374, 8.478], [0.0, 0.0, 0.0]])
    world = local_to_world(local, [50159.0, 499759.0, 0.0])
    np.testing.assert_allclose(world, [[50153.251, 499758.626, 8.478], [50159.0, 499759.0, 0.0]])


@pytest.mark.skipif(not (house_dir("delft_03") / "report.json").exists(), reason="LiDAR outputs not present")
def test_delft_03_labels_land_on_the_footprint():
    """Roof labels in world coordinates must overlap the BAG outline they were cropped from."""
    from shapely.geometry import Polygon

    house = load_house("delft_03")
    roof = Polygon(house.facets[0].polygon_xyz[:, :2])
    for f in house.facets[1:]:
        roof = roof.union(Polygon(f.polygon_xyz[:, :2]).buffer(0))
    overlap = roof.intersection(house.footprint).area / house.footprint.area
    assert overlap > 0.9
    # The roof overhangs the walls, but by well under the 1.0 m crop buffer.
    assert house.footprint.buffer(1.0).contains(roof)
