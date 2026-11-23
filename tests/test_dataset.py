"""Augmentation must move orientation labels exactly as it moves the roof they describe."""

import itertools

import numpy as np
import pytest

from roofphoto.models.dataset import IGNORE, augment, facet_class, octant


def downhill_octant(height: np.ndarray) -> int:
    """Octant a height field slopes down towards, from its gradient (rows run south, columns east)."""
    d_row, d_col = np.gradient(height)
    east, north = -d_col.mean(), d_row.mean()  # downhill = minus the gradient; north is up (-rows)
    return octant(float(np.degrees(np.arctan2(east, north)) % 360))


def test_octants_and_classes():
    assert [octant(a) for a in (0, 22, 23, 90, 180, 270, 337, 338)] == [0, 0, 1, 2, 4, 6, 7, 0]
    assert facet_class(3.0, 90.0) == 1  # under 5°: flat, whatever the azimuth
    assert facet_class(40.0, 0.0) == 2  # north
    assert facet_class(40.0, 315.0) == 9  # north-west


@pytest.mark.parametrize("az", [0, 45, 90, 135, 180, 225, 270, 315])
def test_every_rotation_and_flip_remaps_orientation_like_the_roof(az):
    n = 16
    rows, cols = np.mgrid[0:n, 0:n].astype(float)
    east, north = np.sin(np.radians(az)), np.cos(np.radians(az))
    height = -(cols * east - rows * north)  # a plane sloping down towards az
    assert downhill_octant(height) == octant(az)  # the synthetic roof faces the way we think
    orient = np.full((n, n), facet_class(40.0, az), dtype=np.uint8)
    orient[0, 0] = IGNORE
    for k, lr, ud in itertools.product(range(4), (False, True), (False, True)):
        img, o, (h,) = augment(height[None], orient, [height.astype(np.float32)], k, lr, ud)
        expected = 2 + downhill_octant(img[0])
        assert set(np.unique(o)) == {expected, IGNORE}, (az, k, lr, ud)
        np.testing.assert_allclose(h, img[0], atol=1e-5)  # pitch moves with the image


def test_colour_jitter_keeps_shading_order():
    """Colour jitter must not reverse which slope is brighter (the sun-shading cue)."""
    from roofphoto.models.dataset import colour_jitter

    rng = np.random.default_rng(0)
    img = np.zeros((3, 4, 4), dtype=np.float32)
    img[:, :, :2] = -1.0  # a dark (north-facing) slope
    img[:, :, 2:] = 1.0  # a bright (south-facing) slope
    for _ in range(50):
        out = colour_jitter(img, rng)
        assert (out[:, :, 2:].mean(axis=(1, 2)) > out[:, :, :2].mean(axis=(1, 2))).all()


def test_rotated_samples_keep_roof_and_background_but_not_orientation():
    from roofphoto.models.dataset import rotate_without_orientation

    rng = np.random.default_rng(1)
    orient = np.zeros((8, 8), dtype=np.uint8)
    orient[2:6, 2:6] = 4  # a north-east-facing roof
    orient[0, 0] = IGNORE  # a neighbour
    roof = (orient == 4).astype(np.uint8)
    pitch = np.where(orient == 4, 30.0, np.nan).astype(np.float32)
    for _ in range(20):
        _, o, p, r = rotate_without_orientation(np.zeros((3, 8, 8), np.float32), orient, pitch, roof, rng)
        assert set(np.unique(o)) <= {0, IGNORE}  # no orientation class survives a rotation
        assert (o == 0).sum() == (orient == 0).sum()  # background stays labelled (no all-ignore batches)
        assert r.sum() == roof.sum() and np.nansum(p) == np.nansum(pitch)  # roof and pitch kept


def test_orientation_loss_ignores_background_and_counts_roof_pixels():
    import torch

    from roofphoto.models import unet

    orient = torch.zeros((1, 4, 4), dtype=torch.long)
    orient[0, :2, :2] = 3  # four roof pixels facing NE
    out = torch.zeros((1, 12, 4, 4))
    out[0, 3, :2, :2] = 10.0  # confident and right on the roof
    out[0, 5, 2:, :] = 10.0  # confident nonsense on background: must not count
    roof = (orient > 0).long()
    pitch = torch.full((1, 4, 4), float("nan"))
    loss = unet.loss(out, orient, pitch, roof)
    assert float(loss["orientation"]) < 1e-3
    assert unet.roof_orientation_hits(out, orient) == (4, 4)


def test_orient_weight_scales_only_the_orientation_term():
    import torch

    from roofphoto.models import unet

    torch.manual_seed(0)
    out = torch.randn((1, 12, 4, 4))
    orient = torch.zeros((1, 4, 4), dtype=torch.long)
    orient[0, :2, :2] = 5
    roof = (orient > 0).long()
    pitch = torch.where(orient > 0, torch.tensor(30.0), torch.tensor(float("nan")))
    full = unet.loss(out, orient, pitch, roof)
    quarter = unet.loss(out, orient, pitch, roof, orient_weight=0.25)
    expected = full["total"] - 0.75 * full["orientation"]
    assert float(quarter["total"]) == pytest.approx(float(expected), rel=1e-6)


def test_edge_output_and_loss():
    import torch

    from roofphoto.models import unet

    m = unet.build(pretrained=False, edges=True)
    out = m(torch.zeros((1, 3, 64, 64)))
    assert out.shape[1] == 12 + unet.N_EDGE and unet.edge_logits(out).shape[1] == unet.N_EDGE
    assert unet.edge_logits(unet.build(pretrained=False)(torch.zeros((1, 3, 64, 64)))) is None
    orient = torch.zeros((1, 64, 64), dtype=torch.long)
    roof = torch.zeros((1, 64, 64), dtype=torch.long)
    pitch = torch.full((1, 64, 64), float("nan"))
    edges = torch.zeros((1, 64, 64), dtype=torch.long)
    edges[0, 30:33, :] = 1
    edges[0, :3, :] = 255
    with_e = unet.loss(out, orient, pitch, roof, edges=edges, edge_weight=1.0)
    without = unet.loss(out, orient, pitch, roof)
    assert "edges" in with_e and "edges" not in without
    assert float(with_e["total"]) == pytest.approx(float(without["total"] + with_e["edges"]), rel=1e-5)
