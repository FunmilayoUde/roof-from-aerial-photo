"""Pixel <-> world conversion for north-up rasters in a projected CRS (here RD New, EPSG:28992).

Convention (the same as rasterio and GDAL): a raster's affine transform maps continuous pixel
coordinates (col, row) to world (x, y), where (0, 0) is the top-left *corner* of the top-left pixel.
The centre of pixel [r, c] is therefore at (c + 0.5, r + 0.5). Array sampling (scipy's
map_coordinates) treats integer indices as pixel centres, so `sample` subtracts that half pixel.
Getting this half pixel wrong shifts every overlay by 4 cm at 8 cm resolution: small, but it is
exactly the kind of error Stage 1 is trying to measure.
"""

from __future__ import annotations

import math

import numpy as np
from affine import Affine
from rasterio.transform import from_origin
from scipy.ndimage import map_coordinates

Bounds = tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax) in world units


def snap_bounds(bounds: Bounds, res: float) -> Bounds:
    """Grow bounds outward to multiples of res, so pixels fall on a fixed grid."""
    xmin, ymin, xmax, ymax = bounds
    # Round first so float noise (50152.64 / 0.08 = 626907.9999...) does not add a pixel.
    return (
        math.floor(round(xmin / res, 6)) * res,
        math.floor(round(ymin / res, 6)) * res,
        math.ceil(round(xmax / res, 6)) * res,
        math.ceil(round(ymax / res, 6)) * res,
    )


def grid_for_bounds(bounds: Bounds, res: float) -> tuple[Affine, int, int]:
    """Transform, width and height of a north-up grid that exactly covers snapped bounds."""
    xmin, ymin, xmax, ymax = snap_bounds(bounds, res)
    width = round((xmax - xmin) / res)
    height = round((ymax - ymin) / res)
    return from_origin(xmin, ymax, res, res), width, height


def world_to_pixel(transform: Affine, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World (x, y) to continuous (col, row); (0, 0) is the top-left corner of the raster."""
    col, row = ~transform @ (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return np.asarray(col), np.asarray(row)


def pixel_to_world(transform: Affine, col: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Continuous (col, row) to world (x, y). Use col + 0.5, row + 0.5 for pixel centres."""
    x, y = transform @ (np.asarray(col, dtype=float), np.asarray(row, dtype=float))
    return np.asarray(x), np.asarray(y)


def sample(image: np.ndarray, transform: Affine, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of a 2D image at world points; NaN outside the raster."""
    col, row = world_to_pixel(transform, x, y)
    ci, ri = col - 0.5, row - 0.5  # continuous array indices (pixel centres at integers)
    out = map_coordinates(image, [ri.ravel(), ci.ravel()], order=1, mode="constant", cval=np.nan)
    inside = (ci >= 0) & (ci <= image.shape[1] - 1) & (ri >= 0) & (ri <= image.shape[0] - 1)
    out = out.reshape(np.shape(ci))
    return np.where(inside, out, np.nan)


def crop_window(transform: Affine, shape: tuple[int, int], bounds: Bounds) -> tuple[slice, slice, Affine]:
    """Row/col slices covering bounds (grown to whole pixels, clipped to the raster) and their transform."""
    xmin, ymin, xmax, ymax = bounds
    c0, r0 = world_to_pixel(transform, xmin, ymax)
    c1, r1 = world_to_pixel(transform, xmax, ymin)
    # Same rounding guard as snap_bounds: a bound on a pixel edge must not pull in an extra pixel.
    c0 = max(math.floor(round(float(c0), 6)), 0)
    r0 = max(math.floor(round(float(r0), 6)), 0)
    c1 = min(math.ceil(round(float(c1), 6)), shape[1])
    r1 = min(math.ceil(round(float(r1), 6)), shape[0])
    if c1 <= c0 or r1 <= r0:
        raise ValueError(f"bounds {bounds} do not overlap the raster")
    return slice(r0, r1), slice(c0, c1), transform @ Affine.translation(c0, r0)
