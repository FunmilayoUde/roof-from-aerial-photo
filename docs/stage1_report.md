# Stage 1: aligning aerial photos with the LiDAR roof labels

**Question.** Do the PDOK aerial photos line up with the LiDAR roof outlines well enough to use the
LiDAR output as training labels?

**Answer.** Not as delivered. Roofs in the photo sit **0.45–1.0 m** (6–12 pixels) from the LiDAR
roof, in a direction and by an amount that change from house to house. The cause is **relief
displacement**: ground features line up to within 0.2 m on all four houses while the roofs do
not. After a per-house correction estimated from the photo, roof edges agree to **0.07–0.23 m
RMS** (1–3 pixels). Within a house, the displacement does not always grow with height the way the
textbook model says; Stage 2 should choose the correction model on many houses, not four.

## What was done

1. Loaded the LiDAR pipeline's output for the four houses that have a report (delft_01, 02, 03,
   05; delft_04 is excluded in the LiDAR project). Facet polygons and edges are stored in local
   coordinates; adding `source.offset_xyz` back gives RD New (EPSG:28992) x, y and NAP height.
2. Downloaded the aerial photo around each house from the PDOK WMS and saved it as a
   GeoTIFF in EPSG:28992, plus a crop to footprint + 1.0 m (mitred), the LiDAR pipeline's crop.
3. Took the local ground height from AHN ground points (class 2) in a 2–8 m ring around the
   walls, so roof heights can be expressed above ground.
4. Measured the offset between each LiDAR roof edge and the roof edge in the photo, and fitted
   three models: a constant shift, pure relief displacement (`h·v`), and `s0 + h·v` with both free.
5. Measured the offset **at ground level**, by matching an image of LiDAR ground-point
   intensity to the photo, to tell relief displacement from a coordinate error.
6. Drew overlays: the photo, facets (filled), edges coloured by type, and the BAG walls (dashed),
   as is and with each fitted shift applied.

## Photo layer

| | |
|---|---|
| Service | `https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0` (WMS 1.3.0) |
| Layer | **`2023_orthoHR`**, "Luchtfoto 2023 Ortho 8cm RGB" |
| Pixel size | 0.08 m, requested on a grid snapped to multiples of 8 cm |
| CRS | EPSG:28992 (RD New), same as AHN and BAG |
| Format | JPEG (the only format offered), stored as a deflate-compressed GeoTIFF |
| Licence | CC-BY 4.0, Beeldmateriaal Nederland / PDOK |

Layers on offer run from 2016 to 2026; HR (5–8 cm) exists for 2021–2026. 2023 matches AHN5's
nominal year. AHN5 was flown over several years, so the exact acquisition date of tile (withheld)
is still unchecked (open question 2).

**WMS vs. WMTS.** A WMS draws an image for any bounding box and pixel size you ask for. A WMTS
serves pre-cut tiles on a fixed grid. For one house at a time, WMS is simpler: we ask for exactly
the area and resolution we want, in RD New.

**GeoTIFF georeferencing.** A GeoTIFF stores the pixels plus an *affine transform* (where pixel
(0, 0)'s top-left corner is, and how big a pixel is) and a CRS. `src/roofphoto/geo/raster.py`
does every pixel ↔ world conversion through that transform, and its tests pin the half-pixel
convention: the centre of pixel [r, c] is at (c + 0.5, r + 0.5). Getting that half pixel wrong
would shift every overlay by 4 cm.

## Roof offsets: how they were measured

Each LiDAR edge (minus 0.2 m at each end, where corners mix several lines) is cut into ~1 m
pieces. Each piece is slid along its own normal across the photo, and the offset where the photo
has the strongest edge **parallel to the piece** is kept (the image gradient across the piece,
not its overall strength, which keeps tree texture from winning). A piece only tells us the
shift *across* itself, so each gives one equation `n · shift = d`.

- **Pass 1** searches ±1.2 m. **Pass 2** searches ±0.4 m around where pass 1's fit puts each
  edge. In pass 1 many pieces locked onto stronger lines nearby: a gutter, chimneys, the white
  trim of a flat roof. Pass 2 cut the residual RMS on delft_03 from 0.34 m to 0.11 m.
- Edges of type `unclassified` are left out. They come from the pipeline's traced fallback
  outlines, which it flags as approximate, so they would measure the pipeline's error, not the photo's.
- Fits are least squares, refitted without pieces beyond 3 robust standard deviations.
- Tests (`tests/test_align.py`) recover known shifts from anti-aliased synthetic roofs to ±3 cm,
  including diagonal edges, and pin the sign of the north gradient (image rows run south).

Constant shift per house (photo minus LiDAR; x east, y north):

| house | RD y | shift (m) | size | roof height above ground |
|---|---|---|---|---|
| delft_02 | 445 161 | (−0.07, **+0.62**) | 0.63 m | 2.7–5.8 m |
| delft_01 | 445 456 | (+0.10, **+0.44**) | 0.45 m | 3.6–7.4 m |
| delft_03 | 445 759 | (−0.10, **−0.99**) | 1.00 m | 3.4–8.2 m |
| delft_05 | 445 793 | (−0.04, **−0.77**) | 0.77 m | 5.4–7.5 m |

All four shifts are almost purely north–south: the southern houses lean north, the northern ones
south. A single coordinate error would move them all the same way.

## Relief displacement or a coordinate error? The ground check

**Relief displacement.** An orthophoto is corrected against a model of the *ground*. Anything
above the ground, like a roof, isn't corrected. It appears pushed away from the point straight
below the camera (the nadir) by about `height × distance from nadir / flying height`. The
building seems to lean outward, and the wall facing the camera can show.

The two explanations differ **at ground level**. Relief displacement leaves ground features where
they belong. A coordinate error moves them with the roofs. LiDAR records how strongly each
laser pulse came back (*intensity*), so ground points show road paint, kerbs and paving edges.
`scripts/stage1_ground_check.py` makes an intensity image of ground points within 25 m of each
house (0.25 m cells) and matches its edges to the photo's edges by normalised cross-correlation,
over shifts up to ±1.6 m.

| house | ground shift (m) | roof shift (m) | NCC: no shift / roof shift | "no shift" wins |
|---|---|---|---|---|
| delft_01 | (−0.12, +0.16) | (+0.10, +0.44) | 0.278 / 0.227 | 100% |
| delft_02 | (−0.12, +0.08) | (−0.07, +0.62) | 0.386 / 0.283 | 100% |
| delft_03 | (−0.04, −0.14) | (−0.10, −0.99) | 0.188 / 0.149 | 98.3% |
| delft_05 | (−0.02, 0.00) | (−0.04, −0.77) | 0.250 / 0.156 | 100% |

"Wins" is a **block bootstrap**: the ground is cut into 5 m blocks, blocks are resampled 1000
times, and we count how often "no shift" matches better than the roof's shift. Blocks, not single
cells, are resampled because neighbouring cells see the same kerb and aren't independent
evidence. This is the same reason Stage 4 will need spatial train/test splits.

**Conclusion: relief displacement.** Ground features are in place to within 0.2 m (1–2 pixels)
on every house, while roofs are off by 0.45–1.0 m. delft_03 is the weakest case: its ground is
dominated by a street running north-north-west, and a straight kerb only fixes the shift across
itself, so the correlation peak is a ridge along the street (see its `ground_check.png`). Even
so, "no shift" wins 98% of resamples.

> **Correction (Stage 2).** With 1,000 Bilthoven houses the pattern is clearly *radial*: roofs
> lean away from the nadir of the photo the mosaic took them from (see
> `docs/stage2_report.md`, "The lean is radial"). The flight-strip explanation below was a guess
> from four houses that happened to lie north or south of their nadirs.

Supporting signs: the north–south pattern across houses fits an orthophoto mosaic built from
east–west flight strips. Along a strip, photos overlap heavily, so each house is seen from nearly
overhead east–west. Across strips, a house can be hundreds of metres from the nadir, and houses
on opposite sides of a strip's centre lean opposite ways. On delft_02, a light band runs along
the south side only, between the roof edge and the BAG wall line, which is where the south
facade should show if the roof leans north.

## Within a house: does the offset grow with height?

Pure relief displacement says it should: shift = `h·v`, zero at the ground. Knowing that the
ground is in place fixes `s0 = 0`, which leaves two parameters per house. Fitting `s0` as well
was unstable: a rectangular house has nearly all its edges along two directions, and
extrapolating from roofs 3–8 m up to the ground gave `s0` values over 1.5 m.

RMS residual on the same pieces (those both fits keep):

| house | constant shift | pure relief `h·v` | `v` (m per m of height) |
|---|---|---|---|
| delft_02 | 0.148 | **0.070** | 0.13 towards 343° |
| delft_05 | 0.154 | **0.076** | 0.10 towards 186° |
| delft_01 | **0.231** | 0.256 | 0.10 towards 12° |
| delft_03 | **0.109** | 0.239 | 0.17 towards 187° |

Pure relief fits two houses about twice as well and two houses worse. On delft_03 the
difference is the low flat extension (3.4 m above ground): pure relief puts it 0.56 m off, but it
is measured close to the main roof's ~0.9 m (residual 0.45 m against 0.14 m for a constant
shift). delft_01's low flat garage roof shows the same thing, less strongly. Possible reasons,
not yet tested:

- the photo's edge on a low flat roof is a trim, parapet or facade line, not the LiDAR edge;
- a mosaic seam runs through the house, so its parts come from different source photos with
  different `v`;
- the orthophoto's elevation model is not bare ground everywhere (around buildings, say).

The directions of `v` (north for the two southern houses, south for the two northern ones) agree
with the per-house constant shifts and the flight-strip explanation.

## What this means for Stage 2

- **Labels can't be used where they lie.** A 0.5–1 m offset is 6–12 pixels at 8 cm, enough to
  smear every roof boundary in a segmentation mask.
- **Shift the labels to the photo, not the other way round.** At inference the model only sees
  photos, so labels belong in photo space.
- **Pick the correction model on many houses.** A constant shift per house gets edges to 0.11–0.23
  m RMS; pure relief gets 0.07–0.26 m. With hundreds of houses, Stage 2 can measure which is
  better overall and by roof type. It can also pool `v` across neighbours in the same flight
  strip, so a house with few clean edges borrows from its neighbours.
- **Quality gate.** Keep a house only if its fit is well constrained in both axes (enough inlier
  pieces across each) and its residual is low. Set thresholds from the Stage 2 distribution.
- **LiDAR label defects show up too.** On delft_02 the upper roof's polygon crosses over the lower
  roof (see its overlay). That's a pipeline error, not an alignment one, and the kind of label
  noise the Stage 2 filter has to catch.

## Open questions

1. Why do low flat roofs on delft_03 and delft_01 move almost as much as the main roof? Checking
   for a mosaic seam through the house, and looking at flat-roof edges in the photo, would help.
2. When was AHN5 tile (withheld) flown, and in what season was the 2023 photo taken? Leaf-on
   trees hide eaves (the north side of delft_03).
3. Is flight metadata available (strip lines, nadir points, or seamlines) for the 2023 HR
   orthophoto? It would let us predict `v` instead of fitting it.
4. Is there a true orthophoto (corrected with a surface model, so roofs aren't displaced)
   anywhere in Beeldmateriaal's offering? It isn't in this WMS.

## Reproduce

```sh
uv run python scripts/stage1_align.py --house-id delft_03   # also delft_01, delft_02, delft_05
uv run python scripts/stage1_ground_check.py                # needs stage1_align.py outputs
uv run pytest
```

The first run per house downloads the photo (`data/raw/2023_orthoHR/`) and scans the AHN tile for
ground points (cached in `data/processed/<house>/`, about 40 s per pass). The ground check reads
the tile once for all houses.
