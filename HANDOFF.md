# Project handoff: Roof measurement from aerial photos

## 1. What this project is

I built a pipeline that measures roofs from aerial LiDAR alone (separate repo, described below).
This new project trains an ML model to measure roofs from **aerial photos alone**, using that
LiDAR pipeline's output as training labels.

**Why:** LiDAR is accurate but expensive and missing in most of the world (including Lagos, where I live).
Aerial and satellite photos exist almost everywhere. If a model can learn roof geometry from photos,
roof measurement works where LiDAR doesn't.

**One-line pitch:** "I used my LiDAR roof pipeline to automatically label thousands of roofs, then trained a
model that measures roofs from aerial photos so it works where LiDAR doesn't exist."

**I want to learn a lot from this.** I'm an ML engineer, but newer to geospatial work. Teach as you go (see section 7).

---

## 2. The existing LiDAR pipeline (the "teacher")

Local folder: `~/3D reconstruction` (check the exact name: earlier commands showed a
**trailing space**, i.e. `"3D reconstruction "`. Run `ls -la ~/` to confirm. Always quote this path.)
Write-up: https://roof-measurement-lidar.netlify.app/

What's in that folder (already set up with **uv** and a `pyproject.toml`, so it's installable as-is):
```
3D reconstruction/
  pyproject.toml, uv.lock, .python-version   # uv-managed package
  src/          # pipeline code (import from here)
  scripts/      # runnable entry points
  tests/        # existing test suite
  data/         # AHN5 + BAG inputs, interim outputs (e.g. data/interim/delft_03/stage3/)
  validation/   # hand_measurements.csv + PROTOCOL.md (hand measurements; useful ground truth)
  docs/, site/, .netlify/   # write-up and the published site
  README.md, RESULTS.md
```
Read README.md, RESULTS.md and validation/PROTOCOL.md first.

What it does:
- **Data:** AHN5 (2023) airborne LiDAR, tile (withheld) (Delft), using only points classified as building.
  BAG building footprints (Kadaster) via PDOK isolate each house.
- **Isolate:** crops each house by its footprint plus a 1.0 m buffer, and shifts coordinates near the origin.
- **Find planes:** RANSAC (0.12 m threshold, 1.0 m² minimum plane), then cleanup.
- **Topology:** internal edges from plane intersections, typed by slope. Outer edges from alpha shapes,
  snapped to eave or rake within 10°.
- **Output:** one JSON per house with surface areas (m², ft², roofing squares), pitch (degrees and rise/12),
  edge types (ridge, hip, valley, eave, rake) and lengths, and per-surface confidence.
- **Tools:** Python, numpy, scipy, Open3D, shapely.

Known weaknesses (these matter, because they become label noise):
- Surfaces under 10 m² (dormers, for example) are poor: about 23% median area error, and some fail to close.
- Only 4 houses validated so far, all in one area, with no valleys or dormers.
- Curved roofs are a weak case.
- It trusts AHN building classification, so unlabelled chimneys leave holes.
- No independent ground truth yet.

---

## 3. Goal and success criteria

The model takes an aerial photo crop of one house and predicts, in this order of priority:
1. **Roof surface outlines** (segmentation)
2. **Edge types** (ridge, hip, valley, eave, rake)
3. **Surface area**
4. **Pitch** (stretch goal: hard from a top-down photo, see risks)

Success means I can say: "On houses from areas it never saw in training, the model measures roof area
within X% of LiDAR, beating a simple non-ML baseline by Y." I can also explain where and why it fails.

---

## 4. Repo setup

- This is a **new repo in its own folder, next to the LiDAR folder** (not inside it):
  ```
  ~/
    3D reconstruction/          # existing LiDAR project: READ-ONLY from this project
    roof-from-aerial-photo/     # this project (HANDOFF.md lives here)
  ```
- **Do not modify anything in the LiDAR folder.** It's a finished portfolio piece with a live site.
  If a change there seems necessary, stop and ask me first.
- Use **uv**, same as the LiDAR project, and the same Python version (copy its `.python-version`).
- Install the LiDAR project as an editable dependency so changes there are picked up automatically:
  `uv add --editable "../3D reconstruction"` (adjust for the exact folder name; keep the quotes).
  **Import from it; never copy its code.** Read the LiDAR inputs and outputs from its `data/` folder
  rather than duplicating them.
- Layout for this repo (adjust if you have a better idea, and explain why):
  ```
  roof-from-aerial-photo/
    HANDOFF.md            # this file
    README.md             # what the project is, how to run each stage
    pyproject.toml        # uv-managed; depends on the LiDAR package
    .python-version
    .gitignore            # ignores data/, outputs/, .venv/
    src/roofphoto/        # package code
      io/                 # PDOK download, GeoTIFF read/write, LiDAR output loading
      geo/                # coordinate transforms, cropping, alignment
      viz/                # overlays and figures
      data/               # dataset building (Stage 2)
      models/             # ML models (Stage 4+)
    scripts/              # one runnable entry point per stage: stage1_align.py, stage2_build_dataset.py, ...
    tests/                # mirrors src/; coordinate transforms must be tested
    data/                 # gitignored
      raw/                # downloaded aerial photos
      processed/          # crops + labels per house
    outputs/              # gitignored; overlays, figures, metrics per stage
      stage1/
    docs/
      stage1_report.md    # one report per stage
  ```
- Data and large files stay out of git. Every output must be reproducible by running a script.
- Set up git, ruff and pytest the same way the LiDAR project does.

---

## 5. Roadmap

Do one stage at a time. **Stop after each stage**, summarise what we learned, and wait for me.

| Stage | What | Done when |
|---|---|---|
| **1. Alignment (current)** | Line up one house's aerial photo with its LiDAR roof output | Overlay image + measured offset + short report |
| 2. Dataset | Run the LiDAR pipeline on a few hundred to a few thousand houses; keep only trustworthy results; save photo crop + label per house | Dataset with a quality filter, plus stats on what was kept and dropped |
| 3. Baseline | Non-ML area estimate (footprint area × typical-pitch factor) | Baseline error numbers on a spatial test split |
| 4. First model | U-Net segmentation of roof surfaces, trained and tested with spatial splits | Beats baseline; error analysis written up |
| 5. Extend | Edge types, then area, then pitch; maybe try on Lagos imagery | Each addition evaluated separately |

---

## 6. Stage 1 in detail (do this now)

**Goal:** check whether aerial photos line up well enough with the LiDAR labels to be used as training input.

1. Set up this repo as described in section 4. Read the LiDAR project and summarise how it works and
   where per-house outputs live. Install it into this repo as an editable dependency.
2. Pick one of the four processed Delft houses.
3. Download a PDOK aerial photo (Beeldmateriaal RGB) covering that house.
   - Service: `https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0` (the WMTS is also available).
   - Call GetCapabilities to list layers. Choose the year closest to 2023 (to match AHN5) and the highest
     resolution available (HR, about 5–8 cm, if offered; otherwise 25 cm). Record which layer you used.
   - Request in EPSG:28992 (RD New), the same system as AHN and BAG. Save it as a georeferenced GeoTIFF.
   - The imagery is CC-BY 4.0. Note the attribution in the README.
4. Crop to the same footprint plus 1.0 m buffer the pipeline uses.
5. Make an overlay PNG: photo + roof surfaces (filled, semi-transparent) + edges coloured by type + BAG footprint outline.
6. Measure misalignment between the LiDAR roof outline and the roof in the photo, in metres
   (automatic edge matching, or hand-picked corners if simpler). Explain whether it looks like
   **relief displacement** (buildings appear to lean away from the image centre) or a coordinate error.
7. Write `docs/stage1_report.md`: what was done, the layer, year and resolution used, the offset found, and what it means for Stage 2.

**Tests:** add tests for anything that transforms coordinates (crop bounds, pixel↔world conversion),
so I can trust that overlays sit in the right place.

---

## 7. How to work with me

- **Teach as you go.** The first time a geospatial or ML concept comes up, explain it in 1–2 sentences
  (coordinate systems, GeoTIFF georeferencing, WMS vs WMTS, orthophotos and relief displacement,
  spatial splits, and so on). Point me to the relevant reading in section 9 when it helps.
- **Ask before big decisions** (new dependencies, changing data sources, changing the plan).
- **Never fake data.** If a service is down or a layer is missing, tell me and suggest alternatives.
- **Small, reviewable commits** with clear messages.
- Keep code simple and typed; prefer readable over clever.
- At the end of each stage: summary of what was done, what surprised you, open questions, and a proposed next step.

---

## 8. Known risks (keep these in mind)

- **Relief displacement:** standard orthophotos show roofs shifted from their footprints. LiDAR labels may not line up with photo pixels. Stage 1 exists to measure this.
- **Date mismatch:** roofs can change between the LiDAR year and the photo year. Match years as closely as possible.
- **Pitch from a top-down photo is ambiguous:** steep and shallow roofs can look alike from above. Expect pitch to be the hardest output; treat it as a stretch goal.
- **Label noise:** the LiDAR pipeline's mistakes (small surfaces, curved roofs, chimneys) become training labels. Filter on confidence, and measure the effect.
- **Spatial leakage:** neighbouring houses look alike. Train/test splits must be by area, never random.

---

## 9. Reading list (read when the stage needs it, not up front)

- Stage 1: *Geographic Data Science with Python*, ch. 3 "Spatial Data": https://geographicdata.science/book/intro.html ;
  rasterio quickstart: https://rasterio.readthedocs.io
- Stage 4: TorchGeo intro: https://torchgeo.readthedocs.io/en/stable/tutorials/torchgeo.html ;
  U-Net: https://arxiv.org/abs/1505.04597 ; PyTorch U-Net: https://github.com/milesial/Pytorch-UNet
- Stages 2–4: noisy labels survey: https://arxiv.org/abs/2007.08199 ;
  spatial-kfold: https://pypi.org/project/spatial-kfold/
- Stage 5: HEAT (roof graphs from images): https://arxiv.org/abs/2111.15143 ;
  Building3D (roof wireframes dataset): https://arxiv.org/abs/2307.11914
- Background: Deep Learning in Remote Sensing review: https://arxiv.org/abs/1710.03959

---

**Start now with Stage 1. Begin by reading the LiDAR repo and summarising it back to me before writing any code.**
