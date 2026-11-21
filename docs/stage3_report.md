# Stage 3: the non-ML baseline

**Question.** How well does a house's footprint alone predict its roof area? That's the bar a
photo model must clear before it is worth anything.

**Answer.** Footprint × 1.45 misses the LiDAR roof area by a **median of 11.4%** over all 351
Bilthoven houses, when each house is predicted by a factor fitted without its own sub-tile. The
90th percentile is 27.9%, and 46% of houses are within 10%. On the held-out eastern area alone it
does better (8.5%), because the east happens to be the easiest area. Construction year adds
nothing worth having.

## Setup

- **Houses:** the 351 Bilthoven houses Stage 2 kept, plus Delft's 3 as an out-of-region check.
- **Truth:** the LiDAR pipeline's total roof area. Two sets:
  - *all* houses;
  - *clean*: houses where small or traced facets, whose area is approximate, are at most 10% of
    the roof (159 houses), so the truth itself is reliable.
- **Baselines** (`roofphoto.baseline`), all using the BAG footprint only:
  - **B0** footprint as-is: flat, no overhang.
  - **B1** footprint × k, with k the median roof/footprint ratio on training houses. One factor
    covers both slope and eave overhang.
  - **B2** k per construction-year band (BAG `bouwjaar`: before 1930, 1930–69, 1970–99, 2000 on).
- **Error:** estimate / truth − 1, per house. Reported as the median and 90th percentile of the
  absolute error, the median signed error (bias), and the share within 10%.

**Spatial splits.** Neighbouring houses look alike (same builder, same era, same roof style). A
random split puts near-copies in both train and test, and flatters any method. Holding out whole
areas tests what matters: houses from places the method never saw.

- *East:* train on central Bilthoven (sub-tiles 03, 04, 05, 08, 09, 13), test on the eastern
  sub-tiles 10, 15, 20 (62 houses). These are physically separate, towards Zeist.
- *Leave one sub-tile out (LOSO):* each of the nine sub-tiles held out in turn. The pooled result
  predicts every house from a fit that never saw its area.

## Results

| | B0 | **B1** | B2 |
|---|---|---|---|
| **LOSO pooled, all 351**: median \|error\| | 31.2% | **11.4%** | 10.7% |
| 90th percentile | 43.9% | **27.9%** | 27.1% |
| within 10% | 7% | **46%** | 47% |
| bias (median signed) | −31.2% | −0.3% | +0.1% |
| **LOSO pooled, clean 159**: median \|error\| | 29.1% | **13.3%** | 12.8% |
| **East, all 62**: median \|error\| | 31.4% | **8.5%** | 8.3% |
| 90th percentile | 41.9% | **19.7%** | 20.0% |
| within 10% | 3% | **58%** | 57% |
| **East, clean 33**: median \|error\| | 30.0% | **8.8%** | 8.5% |

B1's factor is 1.45 with any training set (1.44–1.47 across the LOSO folds).

**By held-out sub-tile** (B1, median |error|): 03 15.0%, 04 12.4%, 05 11.7%, 08 11.5%,
09 11.2%, 13 12.1%, **10 8.0%, 15 9.4%, 20 9.0%**. The three eastern sub-tiles are the three
easiest, so the east split on its own flatters the baseline.

**Delft** (3 houses, fitted on all of Bilthoven): B1 11.3%, B2 18.2%. Too few houses to read
anything into, except that the factor isn't wildly off in another town.

## What it means

- **The bar for Stage 4.** A photo model is worth having if it beats a median of **11.4%** (90th
  percentile 27.9%) on a leave-one-sub-tile-out evaluation, and **8.5%** (19.7%) on the east
  split. Both should be reported, because the east alone is easy.
- **Why a single factor fails.** The roof/footprint ratio runs from about 1.0 (flat roofs) to 2.3
  (steep roofs with deep overhangs); see `outputs/stage3/baseline.png`. One number sits in the
  middle. A photo shows what the footprint can't: whether a roof is flat or pitched, where the
  ridges and hips are, and how far the eaves overhang.
- **Construction year doesn't help.** The per-band factors (1.38–1.60) are close to the overall
  1.45, and B2 gains under a point. Not worth carrying forward.
- **The clean subset is harder, not easier** (13.3% vs 11.4%). Noisy truth would do the opposite.
  A plausible reason, untested: houses with few uncertain facets are simpler roofs with one
  dominant pitch, which can be much steeper or flatter than average.

## Reproduce

```sh
uv run python scripts/stage3_baseline.py    # needs the Stage 2 outputs
```

Writes `outputs/stage3/results.json` (every number above) and `outputs/stage3/baseline.png`.
