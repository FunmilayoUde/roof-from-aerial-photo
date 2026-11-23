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
| 1. Alignment | `uv run python scripts/stage1_align.py --house-id delft_03`, then `scripts/stage1_ground_check.py` | [docs/stage1_report.md](docs/stage1_report.md) |
| 2. Dataset | `scripts/download_ahn.py`, `scripts/crop_sheet.py`, then `scripts/stage2_build_dataset.py --tile <LAZ>` per sub-tile and `scripts/stage2_summary.py` | [docs/stage2_report.md](docs/stage2_report.md) |
| 3. Baseline | `uv run python scripts/stage3_baseline.py` | [docs/stage3_report.md](docs/stage3_report.md) |
| 4. First model | `scripts/stage4_train.py`, `scripts/stage4_evaluate.py` per fold, then `scripts/stage4_cv_summary.py` | [docs/stage4_report.md](docs/stage4_report.md) |
| Second town | `scripts/download_ahn.py`, `crop_sheet.py`, Stage 2 per sub-tile, then `scripts/new_town_eval.py` | [docs/second_town_report.md](docs/second_town_report.md) |

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
  geo/ground.py    ground-level check: LiDAR intensity vs photo, block bootstrap
  geo/register.py  per-house registration (Stage 1 method as a function)
  data/teacher.py  the LiDAR pipeline run in memory, writing nothing to the LiDAR folder
  data/candidates.py, data/quality.py, data/chips.py   candidate houses, quality checks, label chips
  baseline.py      Stage 3 footprint-based roof-area baselines
  models/          Stage 4: training targets (dataset.py) and the U-Net (unet.py)
  viz/overlay.py   photo + roof overlays
scripts/           one entry point per stage
tests/             transforms and alignment on synthetic data with known answers
docs/              one report per stage
data/, outputs/    not in git; rebuilt by the scripts
```

## Data sources

- Aerial photos: [PDOK Luchtfoto RGB](https://www.pdok.nl/) (Beeldmateriaal Nederland), CC-BY 4.0.
- LiDAR labels: derived from [AHN](https://www.ahn.nl/) (AHN5 via the official [dataroom](https://www.ahn.nl/dataroom) sheet index), CC0, and BAG building outlines via PDOK, CC0.
