# Stage 5 notes (in progress): orientation

Stage 5 adds edge types and per-facet pitch, both of which depend on knowing which way each roof
surface faces. The Stage 4 models get orientation (flat, or sloped towards one of 8 compass
directions) right on only 18–39% of labelled roof pixels, so orientation comes first.

## Diagnosis (no training): `scripts/stage5_orientation_diagnosis.py`

On the Stage 4 seed-0 models, per fold:

- **Weak on training houses too** (33–58%), not just on held-out areas: a learning or setup
  problem, not only generalisation.
- **Flat vs sloped is mostly right** (83–97%). Most errors are a neighbouring direction; opposite
  directions are rare.
- **Lopsided across directions**: north, south-west and north-west are recognised far better than
  east, west and south, although rotation augmentation shows every roof facing every way.
- 32% of sloped roof area lies within 5° of an octant boundary (22% if uniform), which adds some
  unavoidable near-misses.

## A hypothesis, then a correction

**Hypothesis:** the sun. Slopes facing away from it are dark, slopes facing it bright, and
rotating a training image keeps the shading but changes the label, so rotation augmentation
would scramble the strongest orientation cue.

**Test 1** (west fold, seed 0): colour-only augmentation raised orientation accuracy from 33% to
82% on training houses and from 32% to 49% on test, but cost roof extent (IoU 0.844 → 0.797) and
area (9.5% → 10.7% median error).

**Test 2** (west, seed 0): "mixed" augmentation (rotations on half the images, with roof
orientation ignored on those) did worse on orientation than both: 31% train, 20% test.

**Correction.** The two other Stage 4 rotation models of the west fold (seeds 1 and 2) reach
**54% and 49%** on test, against 32% for seed 0. The seed alone moves orientation accuracy by
about 20 points, so single runs can't tell these augmentation schemes apart. The sun-shading
hypothesis is **not established**: colour-only's 49% is inside the rotation runs' own range. The
error pattern also changes with the seed. Seed 0 is mostly one step off; seeds 1 and 2 are mostly
exactly 90° or 180° off, which suggests the model reads roof shape but confuses symmetric sides.

## Why orientation is unstable, and the change

1. The orientation loss was computed over the whole image, where about 85% of pixels are
   background: easy, and already handled by the roof output. The signal for roof pixels was weak.
2. The kept checkpoint was chosen by total validation loss, which barely reflected roof-pixel
   orientation, so which orientation quality a run ended up with was partly luck.

**Change** (`roofphoto.models.unet`): orientation loss over labelled roof pixels only, among the
9 roof classes; validation logs roof-pixel orientation accuracy; with orientation now a real part
of the validation loss, checkpoint selection takes it into account.

**Next test:** with the new loss, rotations vs colour-only augmentation, 3 seeds each, on the
west fold (runs `s5_west_geom_s{0,1,2}` and `s5_west_colour_s{0,1,2}`).

## Result of the roof-only orientation loss (west fold, 3 seeds each)

| | Stage 4 | new loss, rotations | new loss, colour only |
|---|---|---|---|
| orientation, test (mean, range) | 18–54% | 58% (49–64%) | 59% (57–60%) |
| orientation, training | 31–66% | 79% | 95% |
| roof IoU | 0.84 | 0.57 | 0.51 |
| median area error | 9.5% | 26.5% | 35.5% |
| pitch error, median per house | 8.6° | 13.8° | 12.6° |

Orientation is better and far more stable, colour-only especially (57–60% across seeds), but
roof extent, pitch and area got much worse. The roof-only orientation loss is a hard 9-way
choice, worth about 1.0 against the roof loss's 0.1–0.2, so it dominates the shared network
(validation roof loss 0.17–0.22 against Stage 4's 0.069). Checkpoint selection follows it too:
colour-only kept epoch 15, when orientation started to overfit and roof was still half-trained.

**Next:** weight orientation at about 0.25, and choose checkpoints by that same weighted total;
repeat the 3-seed comparison. Fallback if that can't keep both: separate models for roof/pitch
and for orientation.

## Weighted orientation loss (0.25), west fold, 3 seeds each

| mean of 3 seeds (range) | Stage 4 | weighted, rotations | weighted, colour only |
|---|---|---|---|
| orientation, test | 45% (32–54%) | **64% (61–67%)** | 60% (58–62%) |
| roof IoU | **0.84** | 0.80 | 0.75 |
| median area error | **9.8%** | 11.8% (11.0–13.2%) | 14.7% |
| 90th percentile area error | **21.1%** | 26.7% | 35.0% |
| pitch error, median per house | **8.5°** | 9.9° | 10.4° |

- **Rotations beat colour-only on every measure, orientation included.** The sun-shading
  hypothesis is ruled out; the earlier colour-only advantage was seed noise and the loss setup.
- Weighting gives much better and stable orientation, but one shared network still trades it
  against roof extent, area (+2 points) and pitch (+1.4°).

**Decision proposed:** two models. Area, roof extent and pitch from the Stage 4 models (unchanged,
8.9% median area error); orientation from separate models trained with the weighted loss and
rotations, used inside the Stage 4 roof mask. Remaining work: orientation models for the centre
and east folds (3 seeds each).

## Orientation models for all folds (weighted loss 0.25, rotations, 3 seeds each)

| fold | test orientation accuracy (mean, range) | Stage 4 seed 0 |
|---|---|---|
| west | 64% (61–67%) | 32% |
| centre | 65% (62–67%) | 39% |
| east | 65% (62–69%) | 18% |

All nine runs land at 61–69%. On sloped test pixels: 65–75% exactly right, 5–12% one step off,
10–19% off by 90° (the largest remaining error, likely adjacent facets of hipped roofs), 3–7%
opposite. Flat vs sloped is right 87–91% of the time. Runs: `s5w_<fold>_geom_s{0,1,2}`.

## Edge types by geometry (`roofphoto.edges`, `scripts/stage5_edges.py`)

Roof from the 3 Stage 4 models (averaged), orientation from the 3 orientation models (averaged);
facets are connected regions of one orientation (fragments under ~1 m² merged); boundaries are
typed by which way the facets on either side slope (ridge/hip/valley/break between facets;
eave/rake/perimeter against the outside, from the local outward direction). Scored on all 351
test houses: a predicted edge pixel is right if a labelled edge of the same type is within 24 cm.
The oracle applies the same rules to the true roof and orientation.

| edge type | model precision | model recall | oracle precision | oracle recall |
|---|---|---|---|---|
| any edge | 0.53 | 0.43 | 0.99 | 0.77 |
| ridge | 0.50 | 0.34 | 0.96 | 0.74 |
| eave | 0.50 | 0.43 | 0.94 | 0.85 |
| rake | 0.46 | 0.27 | 0.84 | 0.81 |
| hip | 0.24 | 0.30 | 0.98 | 0.62 |
| perimeter | 0.41 | 0.20 | 0.98 | 0.68 |
| valley | 0.07 | 0.03 | 0.91 | 0.40 |
| break | 0.03 | 0.04 | 0.45 | 0.18 |

- **The rules are sound**: on true orientation they reproduce the LiDAR edges (precision
  0.84–0.99) for every type but break.
- **Break is a rule gap**: in the LiDAR labels it is mostly a pitch change between parts facing
  the same way, which orientation classes can't see (both get the same class and merge).
- **Predictions are the bottleneck.** The previews show rounded, wobbly roof outlines (eaves and
  rakes drift out of tolerance), ridges found in the right place but wavy, and spurious internal
  lines where orientation flickers (most false valleys and breaks). Consistent across folds
  (any-edge precision 0.51–0.55).

**Next:** regularise the geometry before typing edges: turn facets into polygons, simplify them,
and square them to the house's main axes (from the BAG outline), so outlines and ridges become
straight lines. No training needed. A dedicated edge output trained on the edge labels is the
heavier alternative.
