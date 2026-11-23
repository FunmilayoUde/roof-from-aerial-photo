"""U-Net with a pretrained ResNet-34 encoder: orientation classes plus pitch, per pixel.

One network, 12 output channels: 10 orientation logits (see dataset.py), 1 pitch channel and
1 roof logit. Pitch is predicted in units of 45° so it sits on the same scale as the logits early
in training.

Losses, each skipping pixels without a label:
  orientation  cross-entropy over the 9 roof classes, on labelled roof pixels only (Stage 5:
               over the whole image, easy background pixels dominated it and roof-pixel
               orientation came out unstable from seed to seed)
  pitch        smooth L1, only on roof pixels of reliable facets (NaN target elsewhere)
  roof         binary cross-entropy, ignore 255 (neighbours only: uncertain facets are roof)
Roof area comes from the roof output, which sees every facet; orientation only says, within the
roof, which pixels are flat.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from roofphoto.data.chips import EDGE_CLASSES
from roofphoto.models.dataset import IGNORE, N_CLASSES

PITCH_SCALE_DEG = 45.0
N_EDGE = max(EDGE_CLASSES.values()) + 1  # 0 = no edge, then one class per edge type
NO_EDGE_WEIGHT = 0.1  # "no edge" pixels outnumber edge pixels ~30:1


def build(pretrained: bool = True, edges: bool = False) -> torch.nn.Module:
    """edges=True adds N_EDGE edge-type logits after the 12 standard channels (Stage 5)."""
    return smp.Unet(
        "resnet34",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=N_CLASSES + 2 + (N_EDGE if edges else 0),
    )


def edge_logits(out: torch.Tensor) -> torch.Tensor | None:
    """(B, N_EDGE, H, W) edge-type logits, or None for a model without the edge output."""
    return out[:, N_CLASSES + 2 :] if out.shape[1] > N_CLASSES + 2 else None


def split(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(orientation logits (B, 10, H, W), pitch in degrees (B, H, W), roof logit (B, H, W))."""
    return out[:, :N_CLASSES], out[:, N_CLASSES] * PITCH_SCALE_DEG, out[:, N_CLASSES + 1]


def loss(
    out: torch.Tensor,
    orient: torch.Tensor,
    pitch: torch.Tensor,
    roof: torch.Tensor,
    pitch_weight: float = 1.0,
    roof_weight: float = 1.0,
    orient_weight: float = 1.0,
    edges: torch.Tensor | None = None,
    edge_weight: float = 0.0,
    edge_binary: bool = False,
    orient_loss: str = "roof",
) -> dict:
    """orient_loss: "roof" (Stage 5) over labelled roof pixels among the 9 roof classes; "image"
    (Stage 4) over every labelled pixel among all 10 classes, background included."""
    logits, pitch_pred, roof_logit = split(out)
    on_roof = (orient >= 1) & (orient != IGNORE)
    if orient_loss == "image":
        ce = F.cross_entropy(logits, orient, ignore_index=IGNORE)
    elif on_roof.any():
        roof_logits = logits.permute(0, 2, 3, 1)[on_roof][:, 1:]  # (pixels, 9): flat + 8 directions
        ce = F.cross_entropy(roof_logits, orient[on_roof] - 1)
    else:
        ce = torch.zeros((), device=out.device)
    known = roof != IGNORE
    rl = F.binary_cross_entropy_with_logits(roof_logit[known], roof[known].float())
    has_pitch = torch.isfinite(pitch)
    if has_pitch.any():
        pl = F.smooth_l1_loss(pitch_pred[has_pitch] / PITCH_SCALE_DEG, pitch[has_pitch] / PITCH_SCALE_DEG)
    else:
        pl = torch.zeros((), device=out.device)
    total = orient_weight * ce + pitch_weight * pl + roof_weight * rl
    result = {"orientation": ce, "pitch": pl, "roof": rl}
    el = edge_logits(out)
    if edges is not None and el is not None and edge_weight > 0:
        if edge_binary:  # where edges are: the first two edge channels, edge vs none
            el = el[:, :2]
        w = torch.ones(el.shape[1], device=out.device)
        w[0] = NO_EDGE_WEIGHT
        ec = F.cross_entropy(el, edges, weight=w, ignore_index=IGNORE)
        total = total + edge_weight * ec
        result["edges"] = ec
    return {"total": total} | result


def roof_orientation_hits(out: torch.Tensor, orient: torch.Tensor) -> tuple[int, int]:
    """(correct, total) labelled roof pixels where the orientation (among the 9 roof classes) is right."""
    logits = split(out)[0]
    on_roof = (orient >= 1) & (orient != IGNORE)
    pred = logits[:, 1:].argmax(1) + 1
    return int((pred[on_roof] == orient[on_roof]).sum()), int(on_roof.sum())
