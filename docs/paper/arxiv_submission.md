# arXiv submission sheet

Everything the arXiv submission form asks for, ready to paste. Upload `paper.pdf` from this folder.

## Steps
1. Create an account at https://arxiv.org/user/register (use the email you want public on arXiv).
2. Start a new submission: https://arxiv.org/submit
3. Licence: choose **CC BY 4.0**.
4. Archive and category: primary **cs.CV** (Computer Vision and Pattern Recognition);
   cross-list **eess.IV** (Image and Video Processing).
5. If arXiv asks for an **endorsement** (usual for a first submission to cs.CV), it emails you a
   code. Send it to someone who has posted in cs.CV, typically a lecturer or supervisor; they
   endorse at https://arxiv.org/auth/endorse with that code.
6. Upload: `docs/paper/paper.pdf` (made from HTML, not TeX, so arXiv accepts the PDF alone).
7. Fill the metadata below, preview, submit. It appears after the next daily announcement.

## Metadata

**Title**

Roof Geometry from Aerial Orthophotos under Airborne LiDAR Supervision

**Authors**

Oluwatifunmilayo Ude

**Comments**

27 pages, 12 figures, 8 tables. Code: https://github.com/FunmilayoUde/roof-from-aerial-photo ; LiDAR teacher: https://github.com/FunmilayoUde/roof-measurement-from-lidar

**Report number, journal reference, DOI, ACM/MSC class**

Leave empty.

**Abstract** (plain-ASCII version for the form; 1803 characters, limit 1,920)

Airborne LiDAR measures roof geometry well but covers little of the world; aerial imagery covers almost all of it. We train a convolutional network to measure roofs from 8 cm orthophotos with the output of a LiDAR roof-reconstruction pipeline as its only supervision, without manual annotation. Three mismatches separate the LiDAR measurements from usable image labels. Relief displacement shifts roofs by 0.2-0.8 m; a ground-level test confirms the cause, and a one-parameter relief model fitted by edge registration corrects it per house. Registration depends on the JPEG encoding of each image request, so every house is registered on two renderings and kept only if the shift repeats. The LiDAR pipeline degrades at lower point density, so its neighbourhood radii are rescaled to the measured point spacing. A U-Net trained on the resulting 351 houses predicts roof extent, pitch and facing direction per pixel. Under spatially blocked cross-validation in Bilthoven (NL), it estimates roof surface area with a median absolute error of 8.9% against LiDAR, versus 11.5% for the standard footprint-times-factor method (difference -2.5 points, 95% CI -4.1 to -0.3; closer on 204 of 351 houses, two-sided sign test p = 0.003; robust to resampling 250 m blocks). The largest gain is on flat roofs. Rule-typed roof edges reach 0.53 precision and 0.43 recall within 24 cm; the same rules reach 0.84-0.98 precision on the LiDAR labels. Applied to a second town without retraining, the model keeps its segmentation and orientation quality but reads pitch about 7 degrees low on equally steep roofs, biasing area by -10.7%; adding that town's houses to training removes the offset (7.3% area error there, 7.5% in Bilthoven). We also report the negative results and withdrawn conclusions that shaped the method.
