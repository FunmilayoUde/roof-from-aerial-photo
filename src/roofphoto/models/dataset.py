"""Stage 4 training data: Stage 2 chips turned into per-pixel targets.

Targets per pixel:
  roof         1 this house's roof (every facet, uncertain ones included), 0 background,
               255 neighbours: roof extent is known even where orientation isn't
  orientation  0 background, 1 flat roof (pitch < 5°), 2-9 sloped roof facing N, NE, E, SE, S,
               SW, W, NW (downhill direction, octants centred on the compass points),
               255 ignore (neighbours, low-confidence facets)
  pitch        degrees, on roof pixels with a reliable facet; NaN elsewhere (no pitch loss there)

Facets are not predicted as separate objects: adjacent facets almost always face different ways,
so a facet is a connected region of one orientation class.

Augmentation rotates and flips the image by multiples of 90°. That also changes which way a roof
faces, so the orientation classes are remapped with it, or the model would learn wrong labels.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

IGNORE = 255
FLAT_MAX_PITCH_DEG = 5.0  # reliable facets: 542 under 2°, then a gap until ~10°
N_CLASSES = 10  # background, flat, 8 octants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


def octant(azimuth_deg: float) -> int:
    """0 = N, 1 = NE, ... 7 = NW; each octant spans ±22.5° around its compass point."""
    return int(((azimuth_deg + 22.5) % 360) // 45)


def facet_class(pitch_deg: float, azimuth_deg: float | None) -> int:
    if pitch_deg < FLAT_MAX_PITCH_DEG or azimuth_deg is None:
        return 1
    return 2 + octant(azimuth_deg)


# How each array operation moves a downhill direction, as a map on octants (0 = N, clockwise).
# Rows run south and columns east, as in the photo.
#   rot90 (numpy, k=1): content turns 90° anticlockwise, so north-facing becomes west-facing: o - 2
#   fliplr: east <-> west:   o -> -o
#   flipud: north <-> south: o -> 4 - o
def _rot90_octant(o: np.ndarray) -> np.ndarray:
    return (o - 2) % 8


def _fliplr_octant(o: np.ndarray) -> np.ndarray:
    return (-o) % 8


def _flipud_octant(o: np.ndarray) -> np.ndarray:
    return (4 - o) % 8


def _remap(orient: np.ndarray, fn) -> np.ndarray:
    out = orient.copy()
    sloped = (orient >= 2) & (orient != IGNORE)
    out[sloped] = 2 + fn(orient[sloped].astype(int) - 2)
    return out


def augment(
    image: np.ndarray, orient: np.ndarray, others: list[np.ndarray], k_rot: int, flip_lr: bool, flip_ud: bool
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """image (C, H, W); orient and each of others (H, W). Returns copies; only the orientation
    classes are remapped, the others (pitch, roof) just move with the image."""
    for _ in range(k_rot % 4):
        image, others = np.rot90(image, 1, axes=(1, 2)), [np.rot90(o) for o in others]
        orient = _remap(np.rot90(orient), _rot90_octant)
    if flip_lr:
        image, others = image[:, :, ::-1], [o[:, ::-1] for o in others]
        orient = _remap(orient[:, ::-1], _fliplr_octant)
    if flip_ud:
        image, others = image[:, ::-1, :], [o[::-1, :] for o in others]
        orient = _remap(orient[::-1, :], _flipud_octant)
    return np.ascontiguousarray(image), np.ascontiguousarray(orient), [np.ascontiguousarray(o) for o in others]


@dataclass
class Chip:
    tile: str
    house_id: str
    folder: Path
    labels: str  # "relief" or "constant"


def kept_chips(root: Path, tiles: list[str]) -> list[Chip]:
    """Kept houses in these tiles, with the label file Stage 2 recommends for each."""
    chips = []
    for t in tiles:
        with (root / "outputs" / "stage2" / t / "manifest.csv").open() as fh:
            for r in csv.DictReader(fh):
                if r["status"] != "kept":
                    continue
                which = "relief" if r.get("relief_labels_reliable") == "True" else "constant"
                chips.append(Chip(t, r["house_id"], root / "data" / "processed" / "stage2" / t / r["house_id"], which))
    return chips


def load_targets(chip: Chip) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(image float32 (3, H, W) ImageNet-normalised, orientation uint8, pitch float32, roof uint8)."""
    with rasterio.open(chip.folder / "photo.tif") as src:
        rgb = src.read().astype(np.float32) / 255.0
    with rasterio.open(chip.folder / f"labels_{chip.labels}.tif") as src:
        facet_band, roof = src.read(1), src.read(3)
    facets = json.loads((chip.folder / "facets.json").read_text())
    orient = np.where(facet_band == IGNORE, IGNORE, 0).astype(np.uint8)
    pitch = np.full(facet_band.shape, np.nan, dtype=np.float32)
    for f in facets:
        if f["ignored_in_labels"]:
            continue
        m = facet_band == f["class"]
        orient[m] = facet_class(f["pitch_deg"], f["azimuth_deg"])
        pitch[m] = f["pitch_deg"]
    image = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return image, orient, pitch, roof


def colour_jitter(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Brightness/contrast and a small per-channel shift, on the normalised image. Unlike rotating
    or flipping, it leaves the direction the light comes from, and so the sun-shading of slopes,
    unchanged."""
    gain = rng.uniform(0.85, 1.15)
    shift = rng.uniform(-0.1, 0.1, size=(image.shape[0], 1, 1)).astype(np.float32)
    return (image * gain + shift).astype(np.float32)


def rotate_without_orientation(image, orient, pitch, roof, rng):
    """A random non-identity rotation/flip; roof pixels' orientation becomes ignore, background stays 0."""
    while True:
        k, lr, ud = int(rng.integers(4)), bool(rng.integers(2)), bool(rng.integers(2))
        if (k, lr, ud) != (0, False, False):
            break
    image, orient, (pitch, roof) = augment(image, orient, [pitch, roof], k, lr, ud)
    orient = np.where((orient >= 1) & (orient != IGNORE), IGNORE, orient).astype(orient.dtype)
    return image, orient, pitch, roof


class ChipDataset:
    """A torch-style dataset (len, getitem) over kept chips.

    aug (training only): "geom" rotates and flips by multiples of 90° with the orientation classes
    remapped (Stage 4); "colour" only jitters colour, keeping the sun's direction; "none";
    "mixed": colour jitter always, and on half the images a random rotation or flip with the
    orientation of roof pixels set to ignore. Roof extent and pitch don't depend on where the
    light comes from, so they learn from every image; orientation does (sun-shading is its
    strongest cue), so it learns only from images in their true position (Stage 5).
    """

    def __init__(self, chips: list[Chip], train: bool, seed: int = 0, aug: str = "geom"):
        if aug not in ("geom", "colour", "none", "mixed"):
            raise ValueError(f"unknown augmentation {aug!r}")
        self.chips = chips
        self.train = train
        self.aug = aug
        self.rng = np.random.default_rng(seed)
        self.cache = [load_targets(c) for c in chips]  # 354 chips x ~4 MB: fits in memory

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, i: int):
        import torch

        image, orient, pitch, roof = self.cache[i]
        if self.train and self.aug in ("colour", "mixed"):
            image = colour_jitter(image, self.rng)
        if self.train and self.aug == "mixed" and self.rng.random() < 0.5:
            image, orient, pitch, roof = rotate_without_orientation(image, orient, pitch, roof, self.rng)
        if self.train and self.aug == "geom":
            image, orient, (pitch, roof) = augment(
                image,
                orient,
                [pitch, roof],
                int(self.rng.integers(4)),
                bool(self.rng.integers(2)),
                bool(self.rng.integers(2)),
            )
        return (
            torch.from_numpy(image.copy()),
            torch.from_numpy(orient.astype(np.int64)),
            torch.from_numpy(pitch.copy()),
            torch.from_numpy(roof.astype(np.int64)),
        )
