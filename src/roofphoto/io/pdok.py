"""PDOK aerial photos (Beeldmateriaal Nederland, CC-BY 4.0) via WMS, saved as GeoTIFF.

WMS draws an image for any bounding box and size you ask for, which suits one-house crops. The
GeoTIFF stores the pixels plus the affine transform and CRS, so every later step knows exactly
where each pixel sits on the ground.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile

from roofphoto.geo.raster import Bounds, grid_for_bounds

WMS_URL = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
CRS = "EPSG:28992"
ATTRIBUTION = "Beeldmateriaal Nederland / PDOK, CC-BY 4.0"
MAX_PIXELS = 4000  # per side; larger requests are split by the caller (not needed for one house)


def fetch_ortho(layer: str, bounds: Bounds, res: float, out_path: str | Path, timeout: float = 60) -> Path:
    """Download layer over bounds (snapped to res) and write an RGB GeoTIFF in EPSG:28992."""
    transform, width, height = grid_for_bounds(bounds, res)
    if max(width, height) > MAX_PIXELS:
        raise ValueError(f"{width}x{height} px exceeds {MAX_PIXELS}; request a smaller area")
    xmin, ymax = transform.c, transform.f
    xmax, ymin = xmin + width * res, ymax - height * res
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": layer,
        "styles": "",
        "crs": CRS,
        # EPSG:28992's axis order is easting, northing, so WMS 1.3.0 takes x before y here.
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "width": width,
        "height": height,
        "format": "image/jpeg",
    }
    r = _get_with_retries(WMS_URL, params, timeout)
    if not r.headers.get("content-type", "").startswith("image/"):
        # WMS reports errors as XML with HTTP 200; surface it rather than saving it as an image.
        raise RuntimeError(f"PDOK returned {r.headers.get('content-type')}: {r.text[:500]}")

    with MemoryFile(r.content) as mem, mem.open() as src:
        rgb = src.read()
    if rgb.shape[1:] != (height, width):
        raise RuntimeError(f"asked for {width}x{height}, got {rgb.shape[2]}x{rgb.shape[1]}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_geotiff(out_path, rgb, transform, tags={"layer": layer, "source": WMS_URL, "attribution": ATTRIBUTION})
    return out_path


def _get_with_retries(url: str, params: dict, timeout: float, attempts: int = 6) -> requests.Response:
    """GET with backoff on dropped connections and timeouts; HTTP errors are raised at once."""
    import time

    for attempt in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout):
            if attempt == attempts - 1:
                raise
            time.sleep(5 * 2**attempt)  # 5, 10, 20, 40, 80 s
    raise AssertionError("unreachable")


def write_geotiff(path: str | Path, bands: np.ndarray, transform, tags: dict | None = None) -> None:
    """bands: (count, height, width)."""
    count, height, width = bands.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=bands.dtype,
        crs=CRS,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(bands)
        if tags:
            dst.update_tags(**tags)
