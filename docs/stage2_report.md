# Stage 2: dataset (in progress)

**Status.** The pipeline is built and runs end to end on the local Delft tile (AHN5_(withheld)),
which holds only 7 detached-house candidates. The few hundred houses Stage 2 needs are planned
from Bilthoven (about 680 candidates in four neighbouring sub-tiles). GeoTiles, the source the
LiDAR project used, returns 404 for every path since 2026-11-21. The official AHN download works
instead: AHN's dataroom (ahn.nl/dataroom) publishes a sheet index (`bladwijzer.gpkg`) with direct
links, served from basisdata.nl. Bilthoven is sheet **32CN1**, flown in 2023 (the photo's year),
one 7.7 GB LAZ for the whole 5 × 6.25 km sheet; the server supports resumed downloads.

## The pipeline

`scripts/stage2_build_dataset.py --tile <LAZ>` does, per detached-house candidate:

1. **Teacher.** Runs the LiDAR pipeline in memory (`roofphoto.data.teacher`), calling
   `roofrecon`'s own functions with its settings. Nothing is written to the LiDAR folder. On the
   four Delft houses it reproduces that project's `report.json` exactly (facets, edges, totals);
   `tests/test_teacher.py` checks delft_03. One pass over the tile serves every house (16 s for
   four houses, against about 40 s per house with `roofrecon`'s single-box reader).
2. **Photo chip.** 512 × 512 px at 8 cm (40.96 m), on the photo's own pixel grid, so it is never
   resampled.
3. **Registration.** The Stage 1 method, now `roofphoto.geo.register`: constant and pure-relief
   fits per house.
4. **Quality.** Provisional checks on the teacher (coverage, share of area in small or traced
   facets, footprint consistency, unlabelled points inside the walls) and on the registration
   (edge pieces constraining each house axis, residual, shift size). Each records value and limit.
5. **Labels.** Facets and edge types rasterised on the chip grid after the fitted shift, once
   per correction model. Neighbouring buildings (outline + 2 m) are marked *ignore* (255), so
   unlabelled roofs aren't taught as background.

Candidate selection follows the LiDAR project's `find_candidates.py` (residential, one
dwelling, 70–300 m² footprint, a gap to the next building); its BAG query is reused, not copied.

## Delft tile (gap ≥ 1 m): 7 candidates

| | houses |
|---|---|
| kept | 3 (delft_01, delft_02, delft_03) |
| dropped: small or traced facets hold > 25% of the area | 2 (one is delft_04, which the LiDAR project also excluded) |
| dropped: > 5% unlabelled points inside the walls | 2 (one has trees over the roof; one is delft_05, probably its solar panels) |

The drops look justified on the teacher side, but the unlabelled-points limit may be too strict
for outline labels: both dropped houses' labels look well placed in their previews. Thresholds
will be set from the Bilthoven distribution.

## Finding: per-house registration depends on which JPEG rendering we get

PDOK serves the orthophoto only as JPEG, encoded per request. Two requests covering the same
house, on the same 8 cm grid with zero offset, differ by about 2 grey levels on average (up to
25), and the differences cluster at edges, which is where registration looks. Run on the two
renderings, the same house gives:

| house | constant shift (m) | residual: constant / pure relief |
|---|---|---|
| delft_03, Stage 1 download | (−0.10, −0.99) | 0.109 / 0.239 m |
| delft_03, Stage 2 download | (−0.01, −0.89) | 0.203 / 0.240 m |
| delft_05, Stage 1 download | (−0.04, −0.77) | 0.154 / 0.076 m |
| delft_05, Stage 2 download | (−0.19, −0.64) | 0.222 / 0.230 m |

Cropping the Stage 2 download to the Stage 1 extent gives the Stage 2 answer, so it is the
pixels, not the extent. Added Gaussian noise of similar size moves the result much less: the
JPEG differences matter because they sit on edges, where they tip pieces between two nearby
lines (a gutter and the roof edge, say).

What it means:

- A single house's registration is uncertain by about 0.1 m in shift, and its residual can
  double. That is about the size of the difference between the two correction models.
- **Comparing the models house by house is not reliable.** It has to be done across many houses.
- **Now in the pipeline:** each house is registered on two renderings, the second requested
  with its box offset by 7 × 5 px (not a multiple of JPEG's 8 px blocks, so the server encodes
  the same ground differently). Labels use the mean of the two fits, which halves this noise. A
  house is kept only if its two constant shifts agree within 0.15 m (about 2 px). The pure-relief
  fit's agreement is recorded per house (`relief_labels_reliable`) rather than used to drop the
  house, and the model comparison only uses houses where both models repeat.

Copy-to-copy gaps on the Delft tile:

| house | constant shift | pure relief (at the 90th-percentile edge height) |
|---|---|---|
| delft_02 | 0.017 m | 0.087 m |
| delft_03 | 0.029 m | 0.023 m |
| delft_01 | 0.055 m | 0.18 m (relief labels flagged unreliable) |
| delft_05 | 0.19 m (dropped) | 0.13 m |
| delft_04 | 0.25 m (dropped) | 0.34 m |

Every house dropped for repeatability also failed another check.

## Next

1. Download sheet 32CN1 (7.7 GB), crop it to the four sub-tiles, and delete the full sheet.
2. Run on Bilthoven; set the quality thresholds from the distribution; compare the correction
   models across houses and by height; write the dataset statistics.
