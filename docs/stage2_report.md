# Stage 2: the dataset

**Result.** 354 houses with photo chips and LiDAR-derived labels: 351 in Bilthoven (AHN5 sheet
32CN1, 9 sub-tiles) and 3 in Delft. That's 83,100 m² of roof and 3,030 facets, valleys included
(the four Delft houses had none). Facets the LiDAR pipeline is unsure of are marked *ignore* in
the labels rather than dropping the whole house: 13% of roof area in the median kept house. 1,017
candidates went in; why the other 663 were dropped is counted below. Along the way:

- the LiDAR pipeline needed one setting adapted to Bilthoven's sparser point cloud;
- labels need registering per house, because roofs in the photo lean by 0.2–0.8 m;
- **pure relief displacement fits better than a constant shift**, by more the taller the roof;
- the lean is **radial around each aerial photo's nadir**, which corrects the flight-strip
  explanation in the Stage 1 report.

## Data

| | |
|---|---|
| LiDAR | AHN5 sheet 32CN1 (Bilthoven/Zeist), flown 2023, 1.27 billion points, CC0. From the official AHN distribution: ahn.nl/dataroom publishes a sheet index (`bladwijzer.gpkg`) with download links on basisdata.nl. |
| Sub-tiles kept | 03, 04, 05, 08, 09, 10, 13, 15, 20: the nine with 40+ candidates (88% of the sheet's 1,110), each 1 × 1.25 km with a 25 m overlap. The full 7.7 GB sheet was deleted after checking the crops against it point by point. The other 16 sub-tiles held 134 candidates. |
| Photo | PDOK `2023_orthoHR`, 8 cm, two JPEG renderings per house (see Registration). |
| Candidates | BAG: residential, one dwelling, 70–300 m² footprint, at least 3 m from other buildings. A house belongs to the sub-tile whose core holds its centroid. |

GeoTiles, which the LiDAR project used, has returned 404 for every path since 2026-11-21.
`scripts/download_ahn.py` and `scripts/crop_sheet.py` replace it with the official source.

## The teacher on sparser data

The LiDAR pipeline runs in memory (`roofphoto.data.teacher`), calling `roofrecon`'s functions,
and writes nothing to the LiDAR folder. On the first Bilthoven run only 63 of 1,010 houses
passed: the median coverage was 0.53 (Delft: 0.95–0.98), and the roof area from the BAG outline
exceeded the pipeline's by a median of 94%.

**Cause.** Several of the pipeline's distances depend on point spacing: 0.3 m connectivity,
0.3 m facet adjacency, a 0.15 m band around edge lines, a 0.75 m alpha-shape radius, and a
0.4 m touch radius. They were tuned on Delft data at 25–50 points/m². Bilthoven's AHN5 has about
15 points/m², and there the typical gap to a point's 4th neighbour is 0.29 m, against 0.22 m in
Delft. The connectivity check then cuts each roof plane into fragments, and the small fragments
are discarded. It isn't flight strips disagreeing in height: the failing houses were each covered
by one strip, with 4–7 cm of scatter around the plane.

**Fix** (chosen with you). Per house, spacing = the median 4th-nearest-neighbour distance:

- spacing ≤ 0.22 m, the density the LiDAR project validated: its settings, unchanged. The Delft
  reports still reproduce exactly (`tests/test_teacher.py`).
- spacing > 0.22 m: connectivity radius = **1.8 × spacing**, and the other distances above
  scaled by the same factor. The plane threshold (sensor noise) and the 1 m snap distance
  (geometry) stay as they are.

1.8 is the middle of the range that worked in a sweep on 60 houses. It was then confirmed on 60
other houses not used to choose it:

| 60 held-out houses | original settings | adapted |
|---|---|---|
| pipeline ran | 51 | 60 |
| median coverage | 0.53 | **0.957** |
| median \|footprint check\| | 94% | **3.9%** |
| 90th percentile \|footprint check\| | 600% | 17% |

| radius, × spacing | 1.4 | 1.6 | **1.8** | 2.1 | 2.5 |
|---|---|---|---|---|---|
| median \|footprint check\| | 4.3% | 3.9% | 3.5% | 3.9% | 5.2% |
| 90th percentile | 26% | 19% | 17% | 14% | 16% |
| pass all teacher checks | 18 | 20 | 21 | 20 | 18 / 60 |

This change belongs in `roofrecon` eventually. It was made here so the LiDAR project stays untouched.

## Registration

Each house is registered on **two JPEG renderings** of its chip, the second requested 7 × 5 px
off (not a multiple of JPEG's 8 px blocks), because PDOK encodes each request separately and the
differences sit on edges, where registration looks. Labels use the mean of the two fits. A house
is kept only if its two constant shifts agree within 0.15 m. Among kept houses the median
copy-to-copy gap is 0.024 m (90th percentile 0.082 m).

| kept houses (351 Bilthoven) | 10th percentile | median | 90th percentile |
|---|---|---|---|
| shift, photo minus LiDAR | 0.21 m | 0.46 m | 0.82 m |
| edge residual, constant shift | 0.10 m | 0.19 m | 0.26 m |
| edge residual, pure relief | 0.07 m | 0.16 m | 0.23 m |

### Which correction model

Over the 773 houses where both models repeat on both renderings, **pure relief (shift =
height × v) fits better on 511 (66%)**. Pooled over all edge pieces:

| edge height above ground | pieces | constant | pure relief |
|---|---|---|---|
| 0–4 m | 17,166 | 0.243 m | 0.245 m |
| 4–6 m | 15,680 | 0.192 m | 0.191 m |
| 6–8 m | 14,986 | 0.219 m | 0.187 m |
| above 8 m | 9,786 | 0.272 m | 0.195 m |

The gain grows with height, which is what relief displacement predicts: a constant shift has to
compromise between low and high edges. Low edges are equally noisy under both, which is also why
delft_03's flat extension seemed off in Stage 1. That house was among the third where the
constant model wins, and four houses couldn't show the overall pattern. **Stage 4 should train on
the relief labels** (`labels_relief.tif`), except where the relief fit didn't repeat on both
renderings (`relief_labels_reliable` false in the manifest); those houses should use
`labels_constant.tif`.

### The lean is radial

`outputs/stage2/figures/shift_map.png` draws every house's shift as an arrow. The arrows point
**outward from a centre near RD (143.1, 461.0) km**, and from other centres to the south-west
and south-east: roofs lean away from the nadir of whichever photo the mosaic took them from. This
corrects the Stage 1 report, which put the north–south shifts of the four Delft houses down to
east–west flight strips. They were radial too; those houses just lay north or south of their nadirs.
A house's lean can therefore be **predicted** from its position relative to the nearest nadir,
rather than fitted from its own few edges.

## What was dropped, and why

| status | houses |
|---|---|
| kept | 354 (351 Bilthoven, 3 Delft) |
| dropped by a quality check | 593 |
| too large for a 512 px chip with 6 m margin | 60 |
| pipeline found no roof (no building points, or a geometry error) | 10 |

The checks a house failed **on its own** (no other check failed), Bilthoven:

| check | houses | what it means |
|---|---|---|
| unlabelled points > 5% inside the walls | 90 | trees over the roof, or roof objects AHN didn't class as building |
| small or traced facets > 50% of roof area | 50 | too little of the roof would be reliably labelled |
| edge residual > 0.25 m | 47 | registration found no good fit |
| coverage < 0.90 | 37 | |
| \|footprint check\| > 10% | 32 | |
| shift doesn't repeat across renderings | 27 | |

Kept per sub-tile: 46, 73, 14, 64, 68, 30, 24, 14, 18 (03 to 20): 23–40% of candidates. Median
8 facets per kept house (90th percentile 15).

### Uncertain facets: ignored, not dropped

The pipeline marks a facet low-confidence when its outline is traced rather than closed from
plane intersections, or when it is under 10 m². These are 4,763 of 8,835 Bilthoven facets and a
quarter of the roof area, not just dormers. Traced outlines are often wrong for segmentation too:
slivers, overlaps, and polygons that spill past the roof. So they are marked *ignore* (255) in
both label bands, as are untyped edges. Edges with a reliable facet on one side, such as a valley
against a dormer, are kept. A house is kept if at least half its roof area is reliably labelled
(the first run dropped any house with more than a quarter uncertain). That added 105 houses, and
the more complex roofs among them. In the median kept house 13% of the roof is ignored (90th
percentile 39%).

## Known label problems

- **Ignored regions hide roof boundaries too.** Where a traced facet sits at the roof's edge, the
  roof/background boundary there is ignored, so the model learns it only from other houses.
- **Even reliable facets can be wrong**: one checked house has a normal-confidence facet reaching
  out over the lawn. The footprint check only catches large errors.
- **Houses whose roofs are mostly uncertain are still dropped** (50 on that check alone), so the
  set still leans towards simpler roofs, though much less than before.
- **Buildings missing from BAG** (carports, some outbuildings) aren't marked ignore, so they
  are taught as background.
- **Per-house registration noise.** The two JPEG renderings disagree by 0.02 m in the median
  but up to 0.15 m (the limit), which is 1–2 px of label jitter.

## Reproduce

```sh
uv run python scripts/download_ahn.py --sheet 32CN1          # 7.7 GB, resumable
uv run python scripts/crop_sheet.py --sheet data/raw/ahn5/2023_C_32CN1.LAZ --subtiles 03 04 05 08 09 10 13 15 20
uv run python scripts/stage2_build_dataset.py --tile data/raw/ahn5/32CN1_04.LAZ     # per sub-tile
uv run python scripts/stage2_build_dataset.py --min-gap 1                              # the Delft tile
uv run python scripts/stage2_summary.py 32CN1_03 32CN1_04 32CN1_05 32CN1_08 32CN1_09 32CN1_10 32CN1_13 32CN1_15 32CN1_20
```

Per house: `data/processed/stage2/<tile>/<house>/` holds `photo.tif` (512 × 512, 8 cm),
`labels_relief.tif` and `labels_constant.tif` (band 1 facets, band 2 edge types, 255 = ignore:
neighbours, low-confidence facets, untyped edges), `facets.json` (pitch, azimuth, area,
confidence and whether it is ignored, per facet class), and the teacher, registration
and quality records. `outputs/stage2/<tile>/manifest.csv` lists every candidate with its status.
BAG ids are only in the untracked `houses_private.csv`.
