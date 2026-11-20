# Roof Measurement from Aerial Photos

Trains a model to measure roofs from aerial photos alone, using the output of the
[LiDAR roof pipeline](https://roof-measurement-lidar.netlify.app/) (`roofrecon`, in the sibling
folder `../3D reconstruction `) as training labels. See [HANDOFF.md](HANDOFF.md) for the plan.

## Setup

```sh
uv sync
uv run pytest
```

The LiDAR project is installed as an editable dependency and read from, never modified. Its
folder name ends in a space; `pyproject.toml` points at it through `[tool.uv.sources]` because
`uv add` strips the space.

## Stages

| stage | run | report |
|---|---|---|
| 1. Alignment | `uv run python scripts/stage1_align.py --house-id delft_03` | [docs/stage1_report.md](docs/stage1_report.md) |

Stage 1 downloads the PDOK photo around a house, crops it like the LiDAR pipeline does, measures
how far the photo's roof edges sit from the LiDAR edges, and writes overlays to
`outputs/stage1/<house>/`.

## Layout

```
src/roofphoto/
  io/pdok.py       PDOK WMS download to GeoTIFF
  io/lidar.py      the LiDAR pipeline's per-house outputs, in world coordinates; AHN ground height
  geo/raster.py    pixel <-> world transforms, cropping, sampling
  geo/align.py     edge-piece offset search and shift fits
  viz/overlay.py   photo + roof overlays
scripts/           one entry point per stage
tests/             transforms and alignment on synthetic data with known answers
docs/              one report per stage
data/, outputs/    not in git; rebuilt by the scripts
```

## Data sources

- Aerial photos: [PDOK Luchtfoto RGB](https://www.pdok.nl/) (Beeldmateriaal Nederland), CC-BY 4.0.
- LiDAR labels: derived from [AHN](https://www.ahn.nl/), CC0, and BAG building outlines via PDOK, CC0.
