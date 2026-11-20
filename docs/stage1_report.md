# Stage 1: aligning aerial photos with the LiDAR roof labels

**Question.** Do the PDOK aerial photos line up with the LiDAR roof outlines well enough to use the
LiDAR output as training labels?

**Answer.** Not as delivered. The roofs in the photo sit **0.45–1.0 m** (6–12 pixels) from the LiDAR
roof, in a direction and by an amount that change from house to house. After a per-house shift
estimated from the photo itself, the edges agree to about **0.1–0.25 m RMS** (1–3 pixels). The
pattern across houses points to **relief displacement**, not a coordinate error, though one house
does not show the clean height dependence that pure relief predicts (see below).

## What was done

1. Loaded the LiDAR pipeline's output for the four houses that have a report (delft_01, 02, 03,
   05; delft_04 is excluded in the LiDAR project). Facet polygons and edges are stored in local
   coordinates; adding `source.offset_xyz` back gives RD New (EPSG:28992) x, y and NAP height.
2. Downloaded the aerial photo around each house from the PDOK WMS and saved it as a
   GeoTIFF in EPSG:28992, plus a crop to footprint + 1.0 m (mitred), the LiDAR pipeline's crop.
3. Took the local ground height from AHN ground points (class 2) in a 2–8 m ring around the
   walls, so roof heights can be expressed above ground.
4. Measured the offset between each LiDAR roof edge and the roof edge in the photo (method below),
   and fitted two models: a constant shift, and a shift that grows with height.
5. Drew overlays: the photo, facets (filled), edges coloured by type, and the BAG walls (dashed),
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
is still unchecked (open question 3).

**WMS vs. WMTS.** A WMS draws an image for any bounding box and pixel size you ask for. A WMTS
serves pre-cut tiles on a fixed grid. For one house at a time, WMS is simpler: we ask for exactly
the area and resolution we want, in RD New.

**GeoTIFF georeferencing.** A GeoTIFF stores the pixels plus an *affine transform* (where pixel
(0, 0)'s top-left corner is, and how big a pixel is) and a CRS. `src/roofphoto/geo/raster.py`
does every pixel ↔ world conversion through that transform, and its tests pin the half-pixel
convention: the centre of pixel [r, c] is at (c + 0.5, r + 0.5). Getting that half pixel wrong
would shift every overlay by 4 cm.

## How the offset was measured

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

## Results

Constant shift per house (photo minus LiDAR; x east, y north):

| house | RD y | shift (m) | size | RMS | inlier pieces | roof height above ground |
|---|---|---|---|---|---|---|
| delft_02 | 445 161 | (−0.07, **+0.62**) | 0.63 m | 0.17 m | 49 / 49 | 2.7–5.8 m |
| delft_01 | 445 456 | (+0.10, **+0.44**) | 0.45 m | 0.25 m | 73 / 73 | 3.6–7.4 m |
| delft_03 | 445 759 | (−0.10, **−0.99**) | 1.00 m | 0.11 m | 49 / 60 | 3.4–8.2 m |
| delft_05 | 445 793 | (−0.04, **−0.77**) | 0.77 m | 0.18 m | 32 / 37 | 5.4–7.5 m |

Overlays and per-house numbers: `outputs/stage1/<house>/overlay.png`, `relief.png`,
`alignment.json`.

## Relief displacement or a coordinate error?

**Relief displacement.** An orthophoto is corrected against a model of the *ground*. Anything
above the ground, like a roof, isn't corrected. It appears pushed away from the point straight
below the camera (the nadir) by about `height × distance from nadir / flying height`. The
building seems to lean outward, and the wall facing the camera can show.

Evidence **for** relief displacement:

1. **No single coordinate error explains the four houses.** The two southern houses shift
   north and the two northern houses shift south. A datum or georeferencing error would
   shift them all the same way.
2. **The shifts are almost purely north–south**, which is what an orthophoto mosaic built from
   east–west flight strips produces. Photos overlap heavily along a strip, so the mosaic can use
   one taken nearly above each house in the east–west direction. Across strips the overlap is
   smaller, so a house can be hundreds of metres north or south of the camera that photographed
   it. Houses on opposite sides of a strip's centre line lean opposite ways.
3. **On delft_02 a light band runs along the south side only**, between the roof edge and the
   BAG wall line. That band is where the south facade should appear if the roof leans north. It
   could also be a parapet, so this is suggestive, not proof.
4. **The sizes are plausible.** 0.5–1.0 m at 5–8 m height means the house sat at about 0.1–0.16
   × flying height from the nadir, which is within a normal aerial frame.

Evidence **against** pure relief displacement:

- Within delft_03, the best-measured house, the offset across the house is −1.00 m at the ridge
  (8.2 m above ground) and −0.85 m at the eaves (5.1 m). Pure relief through zero at the ground
  would put the eaves at −0.62 m. Possible causes:
  - Eave edges are biased: the LiDAR outline sits slightly inside the true edge, and gutters
    show in the photo but are thin in LiDAR.
  - The orthophoto's ground model isn't bare ground.
  - There is also a constant shift on top of the relief.
  This is unresolved.
- Fitted one house at a time, the height-dependent model (`s0 + h·v`) is unstable. A
  rectangular house has nearly all its edges along two directions, so the shift along the long
  axis rests on a handful of short edges (6 inlier pieces on delft_03, 7 on delft_02). The fit
  extrapolates wildly to ground level (`s0` of 1.6 m) and skews the overlay. Per-house `v`
  values should not be trusted. The along-axis and across-axis numbers in `alignment.json` show
  where support is thin.

A ground-level check would settle it. Relief displacement means features *on the ground*
(kerbs, road markings, paving edges) sit where they should in the photo, while a coordinate
error moves them too. That check isn't done yet (see next steps).

## What this means for Stage 2

- **Labels can't be used where they lie.** A 0.5–1 m offset is 6–12 pixels at 8 cm, enough to
  smear every roof boundary in a segmentation mask.
- **Shift the labels to the photo, not the other way round.** At inference the model only sees
  photos, so labels belong in photo space. A per-house constant shift from this edge matching
  brings edges to 0.1–0.25 m RMS, about 1–3 pixels.
- **Quality gate.** Keep a house only if the fit is well constrained in both axes (enough inlier
  pieces across each) and the RMS is low. The thresholds should be set from the Stage 2
  distribution, not guessed now.
- **Residual skew.** A constant shift ignores the difference between eave and ridge
  displacement, about 0.15 m on delft_03. With hundreds of houses, `v` can be estimated per
  photo strip, pooled across houses, which constrains it far better than one house does.
- **LiDAR label defects show up here too.** On delft_02 the upper roof's polygon crosses over the
  lower roof (see its overlay). That's a pipeline error, not an alignment one, and it's the kind
  of label noise the Stage 2 filter has to catch.

## Open questions

1. Do ground features line up (offset ≈ 0)? This would confirm relief displacement.
2. Can we get flight metadata (strip lines, nadir points, or per-pixel source-photo footprints)
   for the 2023 HR ortho? It would let us predict the displacement instead of fitting it.
3. When was AHN5 tile (withheld) flown, and in what season was the 2023 photo taken? Leaf-on
   trees hide eaves (the north side of delft_03).
4. Is there a true orthophoto (corrected with a surface model, so roofs aren't displaced)
   anywhere in Beeldmateriaal's offering? It isn't in this WMS.

## Reproduce

```sh
uv run python scripts/stage1_align.py --house-id delft_03   # also delft_01, delft_02, delft_05
uv run pytest
```

The first run per house downloads the photo (`data/raw/2023_orthoHR/`) and scans the AHN tile
once for ground points (cached in `data/processed/<house>/ground.json`, about 40 s).
