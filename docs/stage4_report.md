# Stage 4: a U-Net that measures roofs from the photo

**Question.** Can a model that sees only the aerial photo measure roof area better than the
footprint baseline from Stage 3?

**Answer.** Yes. In a 3-fold spatial cross-validation over all 351 Bilthoven houses, each
predicted by models that never saw its area, the final model (3 training seeds averaged, area
calibrated on the other folds' houses) has a median roof-area error of **8.9%**, against 11.5%
for footprint × 1.45. The difference is −2.5 points (95% interval −4.1 to −0.3), and the model is
closer on 204 of 351 houses (sign test p = 0.001). Its 90th-percentile error is 21.8% against
27.7%. It beats the baseline in all three areas, narrowly in the east (8.6% vs 8.9%). On the 159
houses whose LiDAR truth is most reliable, it is **8.2% vs 13.5%** (7.4% uncalibrated). Its
biggest advantage is flat roofs (10.4% vs 34.9%); it is worse on small roofs and orientation is
weak.

## Setup

**Model.** A U-Net with a ResNet-34 encoder pretrained on ImageNet (`segmentation-models-pytorch`).
It takes a 512 × 512 photo chip (8 cm pixels) and outputs, per pixel:

- **roof**: this house's roof or not. Trained on every facet, the uncertain ones included; only
  neighbouring buildings are ignored.
- **orientation**: flat (pitch under 5°) or sloped facing one of 8 compass directions (downhill).
  Uncertain facets are ignored. A facet is a connected region of one class, since adjacent
  facets almost always face different ways.
- **pitch**, in degrees, on reliable facets.

Losses: cross-entropy (orientation), binary cross-entropy (roof), smooth L1 (pitch), each skipping
unlabelled pixels. Rotations and flips by multiples of 90° remap the orientation classes with the
roof (tested against synthetic height fields for every direction). AdamW, learning rate 3 × 10⁻⁴
with cosine decay, batch 8, 40 epochs; the epoch with the lowest validation loss is kept. About
40–90 s per epoch on an Apple M4 GPU (MPS).

**Roof area per house.** The predicted roof regions that touch the house's BAG outline + 1.5 m
(room for the lean), each pixel's plan area divided by cos(predicted pitch). Two alternatives were
also scored: the median training pitch for every sloped pixel ("fallback"), and footprint × a
factor chosen by the predicted flat share ("hybrid").

**Truth and baseline** as in Stage 3: the LiDAR roof area, and B1 = footprint × k with k fitted
on the same training houses as the model. The *clean* subset is houses with at most 10% of their
roof in uncertain facets.

**Spatial cross-validation.** The nine sub-tiles are grouped by grid column into three blocks;
each fold trains on two blocks and tests on the third:

| fold | test sub-tiles | test houses | validation sub-tile |
|---|---|---|---|
| west | 03, 08, 13 | 134 | 10 |
| centre | 04, 09 | 141 | 13 |
| east | 05, 10, 15, 20 | 76 | 13 |

## A first attempt that lost to the baseline

The first model had no roof output: roof extent came from the orientation classes, which ignore
uncertain facets. On the east split it **underestimated** roof area (bias −6.5%) and lost to the
baseline, 11.1% vs 8.5%. It was worst on houses with many uncertain facets (18.3% vs 12.1%). The
previews showed why: it missed dark, shadowed roof sections that had been labelled *ignore*. An
uncertain facet is still roof; only its outline and orientation are doubtful. Ignoring it taught
the model nothing about it, so it learned to leave such roof out. Adding a roof-extent band to
the labels (every facet of the house, `labels_*.tif` band 3) and a roof output to the model fixed
this: roof IoU went from 0.79 to 0.84, and the east-split error from 11.1% to 7.7%.

## Results: single model, seed 0 (3-fold cross-validation, 351 houses)

These are the first full cross-validation results, one model per fold, uncalibrated. The final
results, with 3 seeds and calibration, are further down.

| | photo model | fallback pitch | hybrid | **baseline B1** |
|---|---|---|---|---|
| **all 351**: median \|error\| | **9.4%** | 11.1% | 11.5% | 11.5% |
| 90th percentile | **24.2%** | 28.3% | 28.6% | 27.7% |
| within 10% | **53%** | 46% | 44% | 47% |
| bias (median signed) | −6.3% | −0.8% | +4.5% | +0.3% |
| **clean 159**: median \|error\| | **7.1%** | 12.5% | 12.3% | 13.5% |
| 90th percentile | **23.0%** | 31.1% | 34.5% | 31.2% |

**Against the baseline, house by house:**

| | difference in median error | 95% interval (paired bootstrap) | model better on | sign test (one-sided) |
|---|---|---|---|---|
| all 351 | −2.1 points | −3.9 to +0.1 | 185 | p = 0.17 |
| clean 159 | −6.4 points | −8.9 to −2.1 | 98 | p = 0.002 |

**Per fold** (median |error|, model vs baseline): west 9.5% vs 12.3%, centre 8.9% vs 11.7%,
**east 9.5% vs 8.9%**. The east is the area where the footprint factor fits best (Stage 3), and
there the model doesn't beat it.

**Pixel quality** per fold: roof IoU 0.82–0.84. Predicted pitch is off by a median of 8.4–8.6°
per house. Orientation accuracy is only 18–39%.

## Where it wins and loses (seed 0, uncalibrated)

Median |error|, model vs baseline:

| | houses | model | baseline |
|---|---|---|---|
| mostly flat roof (> 50% flat) | 22 | **9.7%** | 34.9% |
| some flat (5–50%) | 129 | 9.7% | 9.5% |
| no flat (< 5%) | 200 | **8.7%** | 11.5% |
| uncertain facets ≤ 10% | 159 | **7.1%** | 13.5% |
| uncertain 10–30% | 108 | 9.1% | 9.5% |
| uncertain 30–50% | 84 | 11.9% | 11.2% |
| few unlabelled points (trees) < 2% | 225 | **8.3%** | 11.5% |
| unlabelled points 2–5% | 126 | 10.6% | 11.4% |
| roof under 150 m² | 20 | 14.0% | 11.6% |
| 150–250 m² | 199 | **8.0%** | 11.7% |
| over 250 m² | 132 | 11.4% | 11.0% |

- **Flat roofs** are where the photo earns its keep: a single footprint factor assumes a pitched
  roof and overestimates flat ones by a third.
- **The cleaner the truth, the bigger the model's lead.** Some of the apparent ties on houses with
  many uncertain facets are disagreements with a truth that is itself approximate.
- **Trees and small roofs** are weak spots: hidden or small roof sections are missed, which
  feeds the overall underestimate.

## Caveats

- **Seed noise is 0.5–1 point per fold** (pooled 9.4–10.0%), smaller than the gain over the
  baseline. An earlier, larger difference (7.7% vs 9.5% on the east) came from a different
  training set (sub-tile 05 in training), not from the seed alone.
- **Calibration depends on the folds agreeing.** The three factors (1.05–1.08) are close, but a
  new area or photo year would need its own check.
- **Orientation is poor** (18–39% of roof pixels). It barely affects area, which depends on roof
  extent and pitch, but it matters for facets and edge types in Stage 5.
- **The truth is the LiDAR pipeline,** not a measured roof. On houses where it is uncertain, a
  "loss" may be the truth's error.

## Calibration from one validation sub-tile: tried, not adopted

The model underestimates (median bias −6.3%). One factor per fold, c = median(LiDAR area /
model area) over that fold's validation houses (30, 24, 24), was applied to its test houses
(`model_pitch_cal_m2` in `cv_summary.json`). Neither the model nor c saw the test houses.

| median \|error\| | uncalibrated | calibrated | baseline |
|---|---|---|---|
| all 351 | 9.4% | 9.1% | 11.5% |
| difference from baseline | −2.1 (−3.9 to +0.1) | −2.4 (−4.0 to −0.4), p = 0.004 | |
| clean 159 | 7.1% | 9.9% | 13.5% |
| west / centre / east | 9.5 / 8.9 / 9.5% | 7.3 / 12.1 / 9.2% | 12.3 / 11.7 / 8.9% |

The factors (×1.10, ×1.12, ×1.10) are larger than the test underestimate, and the bias flips to
+3.8%. Each rests on 24–30 houses from a single sub-tile, and sub-tile 13 serves two folds. The
overall result improves, but at the cost of the reliable-truth houses and the centre fold, so it
is not adopted. The cross-fold calibration below uses ten times as many houses and replaces it.

## Final: 3 seeds and cross-fold calibration

**Seeds.** Each fold was trained three times (seeds 0, 1, 2), and each house's predicted area is
the mean of its fold's three models. Pooled over all houses, the individual seeds score 9.4%,
9.8% and 10.0%; within a fold they spread 0.5–1.0 points (west 9.5/9.7/10.2, centre 8.9/9.5/9.9,
east 9.5/10.4/10.2). Seed 0 was the luckiest of the three.

**Cross-fold calibration.** Fold F's area factor is the median of LiDAR area / model area over
the *other two folds'* test houses (210–275 houses), each predicted by models that never saw it.
None of them is in F's test set, so F's houses are never used to correct themselves. The factors
are ×1.05, ×1.08 and ×1.05 (west, centre, east), in line with the model's underestimate.

| | ensemble + calibration | ensemble | baseline B1 |
|---|---|---|---|
| **all 351**: median \|error\| | **8.9%** | 9.2% | 11.5% |
| 90th percentile | **21.8%** | 21.9% | 27.7% |
| within 10% | 53% | 54% | 47% |
| bias (median signed) | +0.4% | −5.5% | +0.3% |
| difference from baseline | **−2.5 (−4.1 to −0.3)** | −2.3 (−4.2 to +0.1) | |
| houses closer than baseline | **204 / 351, p = 0.001** | 186 / 351, p = 0.14 | |
| **clean 159**: median \|error\| | 8.2% | **7.4%** | 13.5% |
| difference from baseline | −5.3 (−7.3 to −1.5) | −6.1 (−8.4 to −2.3) | |
| **west / centre / east** | **8.6 / 10.5 / 8.6%** | 9.4 / 8.6 / 10.3% | 12.3 / 11.7 / 8.9% |

Calibration removes the underestimate (bias −5.5% to +0.4%) and makes the model beat the baseline
in every area. It costs a little on the clean houses, which the model underestimated less than
average, so one shared factor pushes them slightly high.

**Where the final model wins and loses** (median |error|, model vs baseline):

| | houses | model | baseline |
|---|---|---|---|
| mostly flat roof (> 50% flat) | 22 | **10.4%** | 34.9% |
| some flat (5–50%) | 129 | 9.2% | 9.5% |
| no flat (< 5%) | 200 | **8.5%** | 11.5% |
| uncertain facets ≤ 10% | 159 | **8.2%** | 13.5% |
| uncertain 10–30% | 108 | 10.8% | 9.5% |
| uncertain 30–50% | 84 | **8.2%** | 11.2% |
| unlabelled points < 2% | 225 | **9.0%** | 11.5% |
| unlabelled points 2–5% (trees) | 126 | **8.5%** | 11.4% |
| roof under 150 m² | 20 | 15.3% | 11.6% |
| 150–250 m² | 199 | **8.0%** | 11.7% |
| over 250 m² | 132 | **10.0%** | 11.0% |

Averaging seeds closed the gap on trees (now a win). Small roofs remain the clear weak spot.

## Next

1. Done: 3 seeds per fold and cross-fold calibration (see Final).
2. Orientation: weight its loss up, or train it separately from roof extent, and check whether
   the 8-way classes are too fine for 8 cm photos.
3. Stage 5 (edge types, then pitch per facet), which needs better orientation first.
4. More data: a second AHN sheet in another town would test generalisation beyond Bilthoven.

## Reproduce

```sh
uv run python scripts/stage4_train.py --run cv_west --test 32CN1_03 32CN1_08 32CN1_13 --val 32CN1_10
uv run python scripts/stage4_evaluate.py --run cv_west
# same for cv_centre (--test 32CN1_04 32CN1_09 --val 32CN1_13)
# and cv_east (--test 32CN1_05 32CN1_10 32CN1_15 32CN1_20 --val 32CN1_13)
uv run python scripts/stage4_cv_summary.py cv_west cv_centre cv_east
# final: seeds 1 and 2 with --seed, runs named cv_<fold>_s1 / _s2, each also scored with --on val
uv run python scripts/stage4_cv_summary.py cv_west,cv_west_s1,cv_west_s2 \
    cv_centre,cv_centre_s1,cv_centre_s2 cv_east,cv_east_s1,cv_east_s2
```

Each fold takes 40–60 minutes on an M4 under `caffeinate`, with about 6 GB of memory free. The
first attempt was stopped once when memory ran low with a second editor window open. Outputs:
`outputs/stage4/<run>/` (model, history, per-house results, previews) and
`outputs/stage4/cv_summary.json`.
