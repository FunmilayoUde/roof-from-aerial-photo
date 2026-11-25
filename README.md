# Roof Measurement from Aerial Photos

Trains a model to measure roofs from aerial photos alone, using the output of the
[LiDAR roof pipeline](https://github.com/FunmilayoUde/roof-measurement-from-lidar)
([write-up](https://roof-measurement-lidar.netlify.app/); `roofrecon`, expected in the sibling
folder `../3D reconstruction `) as training labels. See [HANDOFF.md](HANDOFF.md) for the plan and
the paper, **[docs/paper/paper.md](docs/paper/paper.md)** ([PDF](docs/paper/paper.pdf);
online at [roof-measurement-photo.netlify.app](https://roof-measurement-photo.netlify.app/)), for the
method, results, ablations and limitations.

## Results

Median roof-area error against the LiDAR measurement, spatial cross-validation (every house
predicted by models that never saw its area):

| | photo model | footprint × factor |
|---|---|---|
| Bilthoven, 351 houses, Bilthoven-only models | **8.9%** | 11.5% |
| Heemstede, 147 houses, Bilthoven-only models (new town) | 11.8% | 14.0% |
| Heemstede, models trained on both towns | **8.0%** | 13.9% |
| Bilthoven, models trained on both towns | **8.1%** | 11.2% |

Also: facing direction right on 64–65% of roof pixels; pitch per facet within 8.5° (constant
guess 13.4°); typed edges 0.53 precision / 0.43 recall within 24 cm. Pitch is what transfers worst
between towns (a 7° offset in Heemstede, removed by training on both). A third, unseen town was
not tested.

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
| 5. Orientation, edges, pitch | `scripts/stage4_train.py --orient-weight 0.25`, `scripts/stage5_edges.py`, `scripts/pitch_per_facet.py` | [docs/stage5_notes.md](docs/stage5_notes.md) |
| Second town | `scripts/download_ahn.py`, `crop_sheet.py`, Stage 2 per sub-tile, then `scripts/new_town_eval.py` and `scripts/two_town_compare.py` | [docs/second_town_report.md](docs/second_town_report.md) |
| Paper | `scripts/paper_figures.py`, then `uv run --with markdown python scripts/build_paper.py` (`--site` also writes `site/` for Netlify) | [docs/paper/paper.md](docs/paper/paper.md) |

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
  edges.py, regularise.py   Stage 5 edge typing and scoring, facet straightening
  stats.py         paired bootstrap and sign test
  models/          Stage 4: training targets (dataset.py) and the U-Net (unet.py)
  viz/overlay.py   photo + roof overlays
scripts/           one entry point per stage
tests/             69 tests, mostly on synthetic data with known answers
docs/              one report per stage, the second-town report; docs/paper/ the paper and its figures
data/, outputs/    not in git; rebuilt by the scripts
```

## Data sources

- Aerial photos: [PDOK Luchtfoto RGB](https://www.pdok.nl/) (Beeldmateriaal Nederland), CC-BY 4.0.
- LiDAR labels: derived from [AHN](https://www.ahn.nl/) (AHN5 via the official [dataroom](https://www.ahn.nl/dataroom) sheet index), CC0, and BAG building outlines via PDOK, CC0.

## License

MIT, see [LICENSE](LICENSE). Data: AHN and BAG are CC0; PDOK aerial photographs are CC-BY 4.0
(Beeldmateriaal Nederland).
