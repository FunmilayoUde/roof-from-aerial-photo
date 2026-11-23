# Roof measurement from aerial photographs, trained on LiDAR-derived labels

Final report, roof-from-aerial-photo. November 2026.

## Summary

We train a model to measure house roofs from 8 cm aerial orthophotos alone, using a LiDAR roof
pipeline as an automatic labeller. The labeller (`roofrecon`, a separate project) turns AHN5 point
clouds into facets, typed edges and areas. We register its output onto the photographs, keep the
houses whose labels pass a set of quality checks, and train a U-Net to predict roof extent, pitch
and facing direction per pixel.

On 351 houses in Bilthoven, evaluated by spatial cross-validation so that every house is predicted
by models that never saw its neighbourhood, the photo model measures roof area with a median error
of **8.9%** against the LiDAR measurement. The reference method, the building footprint times a
pitch factor, gets 11.5%. The difference, −2.5 points (95% paired-bootstrap interval −4.1 to
−0.3), holds in each of the three held-out areas, and the model is closer than the reference on
204 of 351 houses (one-sided sign test p = 0.001). Facing direction is right on 64–65% of roof
pixels, pitch per facet is within 8.5° (a constant guess: 13.4°), and about half of predicted roof
edges match a labelled edge of the same type within 24 cm.

In a second town, Heemstede, the Bilthoven model reads roofs as well as it does at home, but
under-reads pitch by about 7°, which makes its area estimates 10.7% low. Adding Heemstede houses to
training removes the offset: 8.0% median area error in Heemstede and 8.1% in Bilthoven, against
footprint baselines of 13.9% and 11.2%. Whether a model trained on two towns transfers to a third
that it has never seen was not tested.

Most of what we learned came from things that did not work the first time. Section 8 lists them,
including three conclusions we had to withdraw.

## 1. Problem

Roof surveys need the area of every roof surface, its pitch, and the type and length of every
edge. Airborne LiDAR gives all of these, but LiDAR is expensive and absent in most of the world.
Aerial and satellite imagery is nearly universal. The question is whether a model can learn roof
geometry from imagery if it is trained on places where LiDAR exists.

The design follows from that question:

- **Labels come from a pipeline, not from people.** `roofrecon` measures a roof from its point
  cloud with RANSAC plane fitting, plane-intersection topology and alpha-shape outlines. On
  synthetic roofs its facet areas are within 0.5% for facets over 10 m² and 23% (median) for
  facets under 10 m². Its known weak spots (small facets, traced outlines, unclassified roof
  objects) become label noise, and much of the work below is about not training on that noise.
- **Evaluation is spatial.** Neighbouring houses share builders, eras and roof styles. A random
  split would put near-duplicates in both training and test. We always hold out whole areas.
- **The reference is non-ML and strong enough to matter.** Footprint × a learned factor already
  captures most of the variation in roof area. A photo model has to beat it to be worth anything.

## 2. Data

| source | what | version | licence |
|---|---|---|---|
| AHN5 point cloud | building and ground points, ~15 pts/m² in Bilthoven, 25–50 in Delft | sheets 32CN1 (Bilthoven, 2023, 1.27 billion points), 25CN1 (Heemstede, 2023) | CC0 |
| BAG | building outlines, use, construction year | via the PDOK WFS | CC0 |
| PDOK Luchtfoto RGB | orthophotos, 8 cm | `2023_orthoHR`, WMS, JPEG only | CC-BY 4.0 |

Everything is in RD New (EPSG:28992), heights in NAP. The LiDAR project's original source
(GeoTiles) went offline during the work; sheets came from the official AHN index (`bladwijzer.gpkg`
on ahn.nl, files on basisdata.nl) and were cut into 1 × 1.25 km sub-tiles with a 25 m overlap.
Each crop was compared point by point with the sheet on a 40 m window straddling its core boundary
(all identical) before the sheet was deleted.

Candidate houses: residential, one dwelling, 70–300 m² footprint, at least 3 m from any other
building, centroid in the sub-tile core. Bilthoven: 9 sub-tiles, 1,010 candidates. Delft: one
sub-tile, 7. Heemstede: 5 sub-tiles, 529.

## 3. From LiDAR measurements to image labels

### 3.1 The photographs do not line up with the LiDAR

LiDAR roof outlines sit 0.45–1.0 m from the roofs in the photograph (6–12 pixels). Two causes were
plausible: a georeferencing error in the photo, or relief displacement (an orthophoto is
rectified against the ground, so anything above it leans away from the camera's nadir by
height × distance to nadir / flying height).

They differ at ground level. We rasterised the intensity of LiDAR ground points (road paint, kerbs
and paving show up), matched its edges to the photo's edges by normalised cross-correlation, and
compared "no shift" with "the roof's shift" in a 5 m block bootstrap. On all four test houses
ground features were in place within 0.2 m, and "no shift" won 98–100% of the resamples. The offset
is relief displacement. Across 1,000 Bilthoven houses the shifts are radial around several nadir
points, as expected for a mosaic of frames.

### 3.2 Registration per house

Each LiDAR edge (minus 0.2 m at the ends) is cut into ~1 m pieces. A piece is slid along its normal
across the photo, scoring the image gradient across the piece rather than its magnitude, which
suppresses tree texture and lines at other angles. A piece only constrains the shift across itself
(the aperture problem), so each gives one equation n · shift = d. Two passes: ±1.2 m, then ±0.4 m
around the first fit, which stops pieces jumping to gutters, chimneys and roof trim (residual on
one test house 0.34 → 0.11 m). Fits are least squares with outliers removed at three robust
standard deviations.

Three shift models were compared:

| model | parameters | outcome |
|---|---|---|
| constant shift s0 | 2 | stable baseline |
| s0 + h·v (h = height above ground) | 4 | unstable on rectangular houses: the long axis is constrained by a few short edges, and extrapolating from 3–8 m to the ground gives s0 of 1.6 m |
| pure relief h·v | 2 | better than constant on 511 of 773 houses where both repeat; gain grows with height (edges above 8 m: 0.195 vs 0.272 m residual) |

Labels use pure relief where it repeats (see 3.3) and the constant shift elsewhere. Median residual
on kept houses: 0.16 m (relief), 0.19 m (constant), about two pixels.

### 3.3 Registration depends on the JPEG rendering

PDOK serves the orthophoto as JPEG, encoded per request. Two requests for the same 8 cm grid differ
by about 2 grey levels on average, concentrated on edges, and that was enough to move one house's
constant shift by 0.1 m and double its residual. Added Gaussian noise of the same size moved it far
less: JPEG differences sit exactly where registration looks.

Each house is therefore registered on two renderings, the second requested with its box offset by
7 × 5 px so that the 8 × 8 JPEG blocks fall differently. Labels use the mean of the two fits, and a
house is kept only if the two constant shifts agree within 0.15 m. Median disagreement on kept
houses: 0.024 m.

### 3.4 The labeller at lower point density

`roofrecon` was tuned on Delft at 25–50 pts/m². In Bilthoven (~15 pts/m²) only 63 of 1,010 houses
passed the quality checks; median coverage (share of roof points assigned to a plane) was 0.53,
and the footprint-based area check was off by 94%. Each failing house was covered by a single
flight strip with 4–7 cm scatter about the plane, which ruled out strip misalignment. The cause
was a fixed 0.3 m connectivity radius: at Bilthoven's density the gap to a point's fourth
neighbour is ~0.29 m, so the connectivity check cut planes into fragments that were then
discarded.

Above Delft's density we scale every spacing-dependent distance (connectivity, adjacency, line
band, alpha-shape and touch radii) so that the connectivity radius is 1.8 × the median
fourth-neighbour spacing. The factor is the centre of the range that worked in a sweep on 60
houses (1.6–2.1); on 60 other houses coverage rose to 0.957 and the footprint check to 3.9%. Below
0.22 m spacing the original settings apply unchanged, and the four Delft reports still reproduce
the LiDAR project's output exactly (a regression test checks this).

### 3.5 Label rasters and quality gates

Per house: a 512 × 512 chip (40.96 m) on the photo's own grid, so the image is never resampled,
and a label file with three bands:

| band | values | ignore (255) |
|---|---|---|
| facets | facet index | neighbouring buildings (BAG outline + 2 m), low-confidence facets |
| edges | ridge, hip, valley, eave, rake, break, perimeter | as above, plus untyped edges |
| roof extent | 1 on every facet of the house | neighbouring buildings only |

A facet is low-confidence if its outline was traced rather than closed from plane intersections,
or if it is under 10 m². These turned out to be a quarter of all roof area, not just dormers, and
their outlines are often wrong (slivers, overlaps, polygons spilling past the roof). Their shape
and orientation are ignored; their existence as roof is not (band 3, see 4.3).

Quality gates (thresholds provisional, set from Delft and the labeller's error table):

| gate | limit | houses dropped by this gate alone (Bilthoven) |
|---|---|---|
| unlabelled points inside the walls | ≤ 5% | 90 |
| small or traced facets | ≤ 50% of roof area | 50 |
| best edge residual | ≤ 0.25 m | 47 |
| coverage | ≥ 0.90 | 37 |
| footprint consistency | within 10% | 32 |
| shift repeats across renderings | ≤ 0.15 m | 27 |

Result: 354 houses (351 Bilthoven, 3 Delft), 83,100 m² of roof, 3,030 facets. In the median kept
house 13% of the roof area is in ignored facets.

## 4. Roof area

### 4.1 Baseline

Footprint × k, with k the median roof/footprint ratio on training houses (it absorbs both slope
and eave overhang). Leave-one-sub-tile-out over Bilthoven: median error 11.4%, 90th percentile
27.9%, 46% of houses within 10%. Footprint alone: 31.2%. A separate factor per construction-year
band: 10.7%, not worth the extra parameters. The ratio ranges from about 1.0 (flat roofs) to 2.3
(steep roofs with deep overhangs), which bounds what any single factor can do.

The three eastern sub-tiles were the three easiest (8.0–9.4%), so a single held-out area can make
any method look better than it is. All later results are over every area.

### 4.2 Model

U-Net with a ResNet-34 encoder pretrained on ImageNet (`segmentation-models-pytorch`), 512 × 512
input, per-pixel outputs:

- roof extent (one logit, binary cross-entropy, ignore 255);
- facing direction (10 classes: background, flat under 5°, eight compass octants of the downhill
  direction; cross-entropy);
- pitch in degrees (smooth L1 on reliable facets).

Rotations by multiples of 90° and flips remap the octant classes with the image; this is tested
against synthetic height fields for every direction and every transform. AdamW, learning rate
3 × 10⁻⁴ with cosine decay, batch 8, 40 epochs, checkpoint at the lowest validation loss. Apple M4
(MPS), 40–90 s per epoch.

Roof area per house: predicted roof regions that touch the house's BAG outline + 1.5 m, each
pixel's plan area divided by cos(predicted pitch).

### 4.3 The first model lost to the baseline

Without the roof-extent band, roof extent came from the orientation classes, which ignore
low-confidence facets. That model underestimated area by 6.5% and lost to the baseline on its
test area (11.1% vs 8.5%), worst on houses with many ignored facets. It had been given no signal
that those regions were roof, and learned to leave dark or uncertain roof out. Adding band 3 and a
separate roof output raised roof IoU from 0.79 to 0.84 and brought the same test area to 7.7%.

### 4.4 Results

Three-fold spatial cross-validation over Bilthoven (folds are columns of sub-tiles: 134, 141 and
76 test houses), three seeds per fold. Each house's area is the mean of its fold's three models,
multiplied by a calibration factor fitted on the other two folds' held-out houses (210–275
houses; factors ×1.05–1.08). The baseline is fitted on the same training houses as the models.

| | photo model | footprint baseline |
|---|---|---|
| median \|error\|, 351 houses | **8.9%** | 11.5% |
| 90th percentile | **21.8%** | 27.7% |
| within 10% | 53% | 47% |
| median bias | +0.4% | +0.3% |
| west / centre / east | 8.6 / 10.5 / 8.6% | 12.3 / 11.7 / 8.9% |
| 159 houses with ≤ 10% uncertain area | 8.2% | 13.5% |

Model minus baseline: −2.5 points (95% paired bootstrap −4.1 to −0.3); the model is closer on
204/351 houses (one-sided sign test p = 0.001).

Calibration matters. Uncalibrated, the ensemble underestimates by 5.5% and the difference is −2.3
points with an interval that reaches +0.1. A factor fitted on one validation sub-tile (24–30
houses) over-corrected to +3.8% bias and was rejected; the cross-fold factor uses ten times as many
houses.

Seed-to-seed variation: 0.5–1.0 points within a fold (single seeds pooled: 9.4%, 9.8%, 10.0%).

Where the model wins and loses (median |error|, model vs baseline):

| | houses | model | baseline |
|---|---|---|---|
| mostly flat roofs | 22 | **10.4%** | 34.9% |
| no flat part | 200 | **8.5%** | 11.5% |
| roofs under 150 m² | 20 | 15.3% | 11.6% |
| 150–250 m² | 199 | **8.0%** | 11.7% |
| uncertain facets 10–30% | 108 | 10.8% | 9.5% |

Flat roofs are where a photograph is most informative: a single footprint factor assumes a pitched
roof and overestimates flat ones by about a third. Small roofs remain the weak case.

## 5. Facing direction, edges and pitch

### 5.1 Facing direction

The Stage 4 models were right on 18–39% of labelled roof pixels and, more tellingly, only 33–58% on
their own training houses, with large differences between compass directions. The problem was
learning, not generalisation.

Two causes, both in the loss. The orientation cross-entropy ran over the whole image, where about
85% of pixels are easy background already handled by the roof output, so the roof-pixel signal
was weak. And checkpoint selection used a validation loss in which roof-pixel orientation barely
registered, so whichever orientation quality a run reached was partly luck.

Computing the orientation loss on roof pixels only made orientation better and stable (58–59%
across seeds) but dominated the shared network: a 9-way choice worth ~1.0 against a roof loss of
0.1–0.2. Roof IoU fell to 0.51–0.57 and area error rose to 26–36%. Weighting orientation at 0.25
recovered most of it (orientation 64%, area 11.8% against 9.8% on the same fold), but a trade-off
remained. We therefore use two models: the Stage 4 models for extent, pitch and area, and separate
orientation models (roof-pixel loss, weight 0.25). The orientation models reach 64–65% on every
held-out area. On sloped pixels, 65–75% are exactly right, 5–12% one octant off and 10–19% off by
90°, the last mostly adjacent facets of hipped roofs.

### 5.2 Edge types

Facets are connected regions of one predicted orientation (fragments under ~1 m² merged into a
neighbour). Each boundary is typed geometrically: between two sloped facets that both fall away
from it, a ridge if they face apart by 135° or more and a hip otherwise; a valley if both fall
towards it; a break for a step; a perimeter, eave or rake against the outside, depending on the
local outward direction relative to the slope.

Scoring: a predicted edge pixel counts if a labelled edge of the same type lies within 3 px
(24 cm). An oracle applies the same rules to the true labels and gives the rules' ceiling.

| edge type | model precision / recall | oracle precision / recall |
|---|---|---|
| any | 0.53 / 0.43 | 0.99 / 0.77 |
| ridge | 0.50 / 0.34 | 0.96 / 0.74 |
| eave | 0.50 / 0.43 | 0.94 / 0.85 |
| rake | 0.46 / 0.27 | 0.84 / 0.81 |
| hip | 0.24 / 0.30 | 0.98 / 0.62 |
| valley | 0.07 / 0.03 | 0.91 / 0.40 |
| break | 0.03 / 0.04 | 0.45 / 0.18 |

The rules are sound; predicted outline and orientation are the bottleneck. "Break" is a gap in the
rules themselves: in the labels it is mostly a pitch change between parts facing the same way,
which orientation classes cannot see.

Three attempts to do better did not beat the rules overall (west fold):

| attempt | any-type precision / recall | what happened |
|---|---|---|
| straighten facets (polygonise, snap to the house axes) | 0.53 / 0.40 | exact on synthetic shapes, but on real predictions mostly fell back to plain simplification |
| learned edge-type output | 0.66 / 0.20 | predicts only eaves and ridges; rare types never beat "no edge" |
| learned edge location, typed by the rules | 0.38 / 0.75 at threshold 0.5; 0.48 / 0.43 at 0.7 | finds far more edges, better for hips and valleys, worse for ridges and eaves; thinning to 1 px did not restore precision |

Choosing a threshold or a per-type mix from these tables would tune on test houses, so edges stay
rule-based.

### 5.3 Pitch per facet

Mean predicted pitch over each reliable facet's true outline, against its LiDAR pitch,
area-weighted, out of fold: 8.5° error in Bilthoven against 13.4° for a constant 38.6°. Predictions
regress strongly towards typical pitches (slope 0.50 of predicted on true): flat roofs read 12–16°,
roofs over 50° read about 41°. Pitch from a near-vertical photograph is underdetermined; the model
hedges.

## 6. A second town

### 6.1 Bilthoven model in Heemstede, nothing re-tuned

Heemstede/Aerdenhout, ~50 km west, coastal dune villages, same LiDAR and photo year. Stage 2 with
identical settings kept 147 of 529 houses (28%; Bilthoven 35%), with the same registration
behaviour (median shift 0.50 m, pure relief better on 63%).

| | Heemstede (147) | Bilthoven (351) |
|---|---|---|
| model median area error | 11.8% | 8.9% |
| baseline | 14.0% | 11.5% |
| model minus baseline | −2.2 (−5.3 to +1.2) | −2.5 (−4.1 to −0.3) |
| model bias | −10.7% | +0.4% |
| roof IoU | 0.837 | 0.82–0.84 |
| orientation accuracy | 71% | 64–65% |
| edges, any type | 0.57 / 0.41 | 0.53 / 0.43 |

What the model reads directly from the image transfers fully. Area does not. Per facet, roofs that
are truly 45.5° read 38.9° in Bilthoven and 31.8° in Heemstede, about 7° lower in every pitch band.
Heemstede's roofs are somewhat steeper (median 41.5° vs 38.6°), but an offset that is the same for
equally steep roofs is about appearance. The most likely carrier of pitch in an orthophoto is
shading, which depends on sun elevation and roofing material; we did not have flight times to
confirm it.

### 6.2 Training on both towns

Same recipe, seed 0, with Heemstede houses added. Each fold pairs a column of Bilthoven sub-tiles
with a column of Heemstede sub-tiles, so every test house is in an area its model never saw.
Compared with the Bilthoven-only seed-0 model of the same fold on the same houses, calibrated as in
4.4:

| | Heemstede (147) | Bilthoven (351) |
|---|---|---|
| two-town model | **8.0%** | **8.1%** |
| Bilthoven-only model | 11.0% | 8.5% |
| baseline (each model's own training houses) | 13.9% | 11.2% |
| two-town minus baseline | −5.9 (−8.5 to −2.9), p = 0.0005 | −3.1 (−4.8 to −1.6), p = 0.0005 |
| two-town minus Bilthoven-only | −3.0 (−5.2 to −0.5), p = 0.05 | −0.4 (−1.6 to +0.6) |
| pitch per facet, Heemstede | 8.8° (bias −2.7°) | |
| same, Bilthoven-only model | 13.1° (bias −8.9°) | |

The pitch offset closes (45.5° roofs read 38.9° in both towns) and the home town loses nothing.
Heemstede is no longer an unseen town after this, so the result shows that local training data
removes the gap. It does not show that two towns suffice for a third.

## 7. What the numbers support

- From 8 cm orthophotos and a building outline, roof area can be measured to a median 8–9% of a
  LiDAR measurement in towns represented in training, about 3 points better than a footprint
  factor, with the largest gains on flat and mixed roofs.
- Roof extent, facing direction and edge geometry transfer to a new town without retraining.
  Pitch, and through it area, does not; a modest amount of labelled data from the new town fixes
  it.
- Edge typing works for ridges and eaves and poorly for valleys and breaks, limited by predicted
  orientation more than by the typing rules.

## 8. Negative results and withdrawn conclusions

These shaped the method more than the successes did.

| what we believed or tried | what happened |
|---|---|
| the Delft shifts came from east–west flight strips | withdrawn: 1,000 houses show a radial pattern around frame nadirs |
| a 4-parameter relief model per house | unstable; replaced by pure relief once the ground check fixed s0 = 0 |
| the labeller's settings transfer between densities | they don't; a connectivity radius tuned at 25–50 pts/m² fragments planes at 15 |
| dropping houses with uncertain facets keeps labels clean | it dropped 25% of usable houses and biased the set to simple roofs; ignoring those facets kept 105 more houses |
| ignoring uncertain facets is enough | the model then learned to omit uncertain roof; extent had to be labelled separately |
| calibrating on one validation sub-tile | over-corrected; replaced by cross-fold calibration |
| "training noise is about 2 points" | withdrawn: seeds vary 0.5–1 point; the 2-point gap compared different training sets |
| rotation augmentation scrambles the sun-shading cue | withdrawn: the seed alone moved orientation by 20 points; with three seeds per setting, rotations won on every measure |
| learned edge bands were too thick | withdrawn: thinning to 1 px raised precision by only 2–8 points |
| Heemstede's area error comes from steeper roofs | only in part: the main cause is a 7° town-wide pitch offset |

Two process lessons. First, a single training run cannot separate two methods whose difference is
smaller than the seed-to-seed spread; several conclusions above were drawn too early from one run.
Second, the oracle analyses (rules on true labels, edges on true orientation) were the fastest way
to locate a bottleneck.

## 9. Limitations

- **Two towns, both Dutch, both 2023 imagery.** Transfer to a third town never trained on, to a
  different photo year, and to non-Dutch building stock is untested.
- **The truth is a pipeline.** Errors are measured against `roofrecon`, not against surveyed
  roofs. Its small-facet error (23% median under 10 m²) and traced outlines are in the labels; the
  quality gates and ignore masks reduce this but do not remove it. The hand-measurement protocol in
  the LiDAR project was never carried out.
- **Building outlines are an input.** The area method needs a footprint to select the house and
  the baseline needs one to exist at all. Outside the Netherlands, OpenStreetMap is the likely
  substitute, with uneven coverage.
- **Pitch is the fragile output.** It regresses to the training distribution and shifts with
  imaging conditions.
- **Edges.** Valleys and breaks are close to unusable; "break" needs pitch-aware typing.
- **Single seeds** for the two-town models.

## 10. Future work

1. A third town flown in a different year (Vught or Oisterwijk, AHN5 2024), with the two-town
   models unchanged: the direct test of transfer.
2. For Lagos and similar settings: a small labelled sample from the target region, given 6.2.
   Satellite imagery is coarser than 8 cm, roofing is mostly corrugated metal and flat concrete,
   and there is no LiDAR to score against, so the first test there is qualitative.
3. Pitch that is less dependent on shading: training on more flights, or photometric
   normalisation per flight.
4. Edge typing with a per-type choice between learned locations and rules, selected on validation
   houses; pitch-aware breaks.
5. A three-seed release of the two-town models.

## 11. Reproducing the results

Hardware: Apple M4, 16 GB, PyTorch MPS. A 40-epoch training run takes 30–55 minutes; disk peaks
at about 8 GB per AHN sheet while cropping.

```sh
uv sync
uv run pytest                                                       # 69 tests

# data
uv run python scripts/download_ahn.py --sheet 32CN1
uv run python scripts/crop_sheet.py --sheet data/raw/ahn5/2023_C_32CN1.LAZ --subtiles 03 04 05 08 09 10 13 15 20
uv run python scripts/stage2_build_dataset.py --tile data/raw/ahn5/32CN1_04.LAZ     # per sub-tile
uv run python scripts/stage2_summary.py 32CN1_03 32CN1_04 ...

# baseline
uv run python scripts/stage3_baseline.py

# area models: 3 folds x 3 seeds (seed 1 shown), then pooled
uv run python scripts/stage4_train.py --run cv_west_s1 --seed 1 --orient-loss image \
    --test 32CN1_03 32CN1_08 32CN1_13 --val 32CN1_10
uv run python scripts/stage4_evaluate.py --run cv_west_s1
uv run python scripts/stage4_evaluate.py --run cv_west_s1 --on val
uv run python scripts/stage4_cv_summary.py cv_west,cv_west_s1,cv_west_s2 cv_centre,... cv_east,...

# orientation models, edges, pitch
uv run python scripts/stage4_train.py --run s5w_west_geom_s0 --orient-weight 0.25 \
    --test 32CN1_03 32CN1_08 32CN1_13 --val 32CN1_10
uv run python scripts/stage5_edges.py
uv run python scripts/pitch_per_facet.py

# second town
uv run python scripts/new_town_eval.py --name heemstede 25CN1_01 25CN1_02 25CN1_06 25CN1_07 25CN1_13
uv run python scripts/two_town_compare.py
```

The Stage 4 area models were trained before the orientation-loss option existed; `--orient-loss
image` reproduces their loss. Per-stage details, including every intermediate number, are in
`docs/stage1_report.md` to `docs/stage4_report.md`, `docs/stage5_notes.md` and
`docs/second_town_report.md`.

## Appendix: repository map

```
src/roofphoto/
  io/          pdok.py (WMS to GeoTIFF), lidar.py (labeller outputs, one-pass tile reader)
  geo/         raster.py (pixel/world), align.py + register.py (edge-piece registration), ground.py
  data/        teacher.py (labeller in memory), candidates.py, quality.py, chips.py (label rasters)
  models/      dataset.py (targets, augmentation), unet.py (model, losses)
  baseline.py, edges.py, regularise.py, stats.py
scripts/       one entry point per step, listed in section 11
tests/         69 tests: coordinate transforms, registration on synthetic images, label
               rasterisation, augmentation equivariance, edge typing, the labeller's reproduction
docs/          stage reports, Stage 5 notes, second-town report, this report
```
