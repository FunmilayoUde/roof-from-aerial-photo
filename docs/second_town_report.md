# Second town: Heemstede / Aerdenhout

**Question.** Does the Bilthoven model hold in a town it has never seen, with nothing tuned on it?

**Answer.** What the model reads directly from the photo transfers fully: roof outline, facing
direction and edges are as good as in Bilthoven, or better. Roof area transfers only partly. The
model still beats the footprint baseline by about the same margin (11.8% vs 14.0% median error),
but it underestimates area by 10.7%, so the gain is no longer statistically clear over all 147
houses. On the 59 houses with reliable LiDAR truth it is clearly better (10.2% vs 19.1%). The
underestimate comes at least partly from pitch: Heemstede's roofs are about 3° steeper, and the
model under-reads pitches steeper than those it learned.

## Data

| | |
|---|---|
| Town | Heemstede / Aerdenhout, about 50 km west of Bilthoven, coastal dune villages |
| LiDAR | AHN5 sheet 25CN1, flown 2023 (same year as Bilthoven and the photos), 6.0 GB |
| Sub-tiles | 01, 02, 06, 07, 13 (the five with most candidates near the town centre), 25 m overlap, checked point by point against the sheet; sheet then deleted |
| Photos | PDOK 2023_orthoHR, 8 cm, two renderings per house, as in Stage 2 |
| Houses | 529 candidates, **147 kept** (28%; Bilthoven 35%). Same settings and quality checks, nothing re-tuned |

Chosen from six scouted towns for being the same LiDAR and photo year as Bilthoven (so only the
place changes), having the most candidates, and being a different setting. Laren/Blaricum (close
to Bilthoven) and Vught/Oisterwijk (flown 2024) were the alternatives; Haren had no AHN5 yet.

Stage 2 behaved as in Bilthoven: median shift 0.50 m, pure relief better than a constant shift on
63% of houses (Bilthoven 66%).

## Method (`scripts/new_town_eval.py`)

- **Model:** all 9 Bilthoven area models (3 folds × 3 seeds), none of which saw Heemstede, areas
  averaged; calibration factor ×1.059 from all 351 Bilthoven out-of-fold predictions.
  Orientation from the 9 Bilthoven orientation models averaged; edges by the Stage 5 rules.
- **Baseline:** footprint × 1.454, fitted on Bilthoven.
- **Reference, not a fair competitor:** footprint × a factor fitted on Heemstede's own LiDAR
  (1.491). It shows what local LiDAR would buy a footprint method.

## Results

Sign-test p-values in this report are one-sided (model closer more often). The paper
(`docs/paper/paper.md`) reports two-sided values: 0.74 for 76/147, 0.0009 for 94/147 and 207/351,
0.10 for 84/147.

| | Heemstede (147) | Bilthoven (351, Stage 4) |
|---|---|---|
| model median area error | **11.8%** | 8.9% |
| model, without calibration | 15.6% | 9.2% |
| baseline (footprint × 1.454) | 14.0% | 11.5% |
| reference: footprint × local factor | 13.4% | |
| model minus baseline (95% interval) | −2.2 (−5.3 to +1.2) | −2.5 (−4.1 to −0.3) |
| houses where the model is closer | 76 / 147 (p = 0.37) | 204 / 351 (p = 0.001) |
| model median bias | **−10.7%** | +0.4% |
| 90th percentile area error, model / baseline | 26.2% / 31.1% | 21.8% / 27.7% |
| **clean houses**: model / baseline | **10.2% / 19.1%** (59 houses; −8.9, interval −12.1 to −1.8) | 8.2% / 13.5% |
| roof IoU | 0.837 | 0.82–0.84 |
| orientation accuracy (roof pixels) | **71%** | 64–65% |
| pitch error, median per house | **11.1°** | about 8.5° |
| edges, any type: precision / recall | 0.57 / 0.41 | 0.53 / 0.43 |
| ridge / eave / hip precision | 0.59 / 0.54 / 0.36 | 0.50 / 0.50 / 0.24 |

**Roof pitch in the two towns** (LiDAR, reliable facets, area-weighted):

| | 25th pct | median | 75th pct | sloped share of area |
|---|---|---|---|---|
| Bilthoven | 25.5° | 38.6° | 46.4° | 88.7% |
| Heemstede | 28.9° | 41.5° | 48.6° | 88.8% |

## What it means

- **Reading the roof transfers.** Outline, orientation and edges are as good as or better than at
  home. The model has learned what roofs look like, not just Bilthoven's.
- **Measuring area depends on pitch, and pitch doesn't fully transfer.** About 3° steeper roofs
  mean roughly 4–5% more true area for the same footprint; a model trained on Bilthoven pulls
  pitch towards what it saw there, so it under-reads steeper roofs. Pitch error rose from about
  8.5° to 11.1°. Calibration fitted on Bilthoven can't correct a bias that comes from the new
  town's roofs.
- **The footprint baseline also degrades** (11.5% to 14.0%), and even a factor fitted on local
  LiDAR only reaches 13.4%: Heemstede's roofs vary more than Bilthoven's. The model beats both.
- **The overall claim, restated honestly:** across two towns the model's median area error is
  about 2 points below a footprint baseline; within its training town that gain is clear, in a
  new town it is the same size but uncertain, because of a pitch-driven underestimate.

## Next

1. Train on both towns (a three-fold split per town, or leave-one-town-out), which gives the
   model a wider range of pitches; check whether the underestimate shrinks.
2. Look at pitch directly: predicted against LiDAR pitch by facet, to confirm the pull towards
   Bilthoven's typical pitch.
3. A third town flown in a different year (Vught or Oisterwijk, 2024) would test the photo year
   as well.

## Follow-up: pitch per facet

`scripts/pitch_per_facet.py` (details in `docs/stage5_notes.md`). For equally steep roofs (true
45.5°), the model reads 38.9° in Bilthoven and 31.8° in Heemstede: a ~7° town-wide offset across
every pitch band, on top of the pull towards typical values seen in both towns. So the
underestimate is mainly about how the photo looks (shading under different light, roof material),
not only steeper roofs. Pitch from appearance is the part that transfers worst, which also matters
for Lagos. Training on both towns is the direct test of whether varied data fixes it.

## Training on both towns (`scripts/two_town_compare.py`)

*Superseded:* the table below is the original one-seed comparison. With three seeds per side
(paper §7.6, Table 6) the two-town ensemble reaches 7.3% in Heemstede and 7.5% in Bilthoven, against
11.1% and 8.9% for the Bilthoven-only ensembles (p = 0.0005 and 0.042).

Same Stage 4 recipe (rotations, Stage 4's orientation loss), seed 0, with Heemstede's houses added
to training. Each fold pairs a Bilthoven block with a Heemstede block, so every test house is in an
area its model never saw: west = Bilthoven 03, 08, 13 + Heemstede 01, 06; centre = 04, 09 + 02, 07;
east = 05, 10, 15, 20 + 13. Compared with the Bilthoven-only seed-0 model of the same fold, on the
same houses. Calibrated as in Stage 4 (factor from the other folds' houses).

| median area error | Heemstede (147) | Bilthoven (351) |
|---|---|---|
| **two-town model** | **8.0%** | **8.1%** |
| Bilthoven-only model | 11.0% | 8.5% |
| footprint baseline (each model's own training houses) | 13.9% | 11.2% |
| two-town minus baseline (95% interval) | −5.9 (−8.5 to −2.9), 94/147, p = 0.0005 | −3.1 (−4.8 to −1.6), 207/351, p = 0.0005 |
| two-town minus Bilthoven-only | −3.0 (−5.2 to −0.5), p = 0.05 | −0.4 (−1.6 to +0.6) |
| bias before calibration, two-town / Bilthoven-only | −6.0% / −17.4% | −4.3% / −6.3% |
| 90th percentile, two-town / Bilthoven-only | 20.1% / 27.9% | 21.4% / 23.0% |

Pitch per facet on Heemstede (685 facets): error **8.8°** against 13.1° for the Bilthoven-only
model, bias **−2.7°** against −8.9°. Roofs that are truly 45.5° read 38.9° (as in Bilthoven)
instead of 30.8°: **the town-wide pitch offset is gone.**

One Bilthoven-only prediction found no roof at all on one Heemstede house; it counts
as 100% off in the errors and is left out of the calibration factor only.

**What it shows.** The pitch problem was a matter of training variety: photos from one flight taught
a pitch reading that didn't transfer; photos from two transfer between both, and the home town
doesn't lose anything. **What it doesn't show:** Heemstede is no longer an unseen town (other parts
of it were in training), so this proves that local training data closes the gap, not that two towns
are enough for a third. That needs a third town, never trained on. One seed per fold, but the gains
are large against the 0.5–1 point seed noise measured in Stage 4.
