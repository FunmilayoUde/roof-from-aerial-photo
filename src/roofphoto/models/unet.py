"""U-Net with a pretrained ResNet-34 encoder: orientation classes plus pitch, per pixel.

One network, 12 output channels: 10 orientation logits (see dataset.py), 1 pitch channel and
1 roof logit. Pitch is predicted in units of 45° so it sits on the same scale as the logits early
in training.

Losses, each skipping pixels without a label:
  orientation  cross-entropy, ignore 255 (neighbours, low-confidence facets)
  pitch        smooth L1, only on roof pixels of reliable facets (NaN target elsewhere)
  roof         binary cross-entropy, ignore 255 (neighbours only: uncertain facets are roof)
Roof area comes from the roof output, which sees every facet; orientation only says, within the
roof, which pixels are flat.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from roofphoto.models.dataset import IGNORE, N_CLASSES

PITCH_SCALE_DEG = 45.0


def build(pretrained: bool = True) -> torch.nn.Module:
    return smp.Unet(
        "resnet34",
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=N_CLASSES + 2,
    )


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
) -> dict:
    logits, pitch_pred, roof_logit = split(out)
    ce = F.cross_entropy(logits, orient, ignore_index=IGNORE)
    known = roof != IGNORE
    rl = F.binary_cross_entropy_with_logits(roof_logit[known], roof[known].float())
    has_pitch = torch.isfinite(pitch)
    if has_pitch.any():
        pl = F.smooth_l1_loss(pitch_pred[has_pitch] / PITCH_SCALE_DEG, pitch[has_pitch] / PITCH_SCALE_DEG)
    else:
        pl = torch.zeros((), device=out.device)
    return {"total": ce + pitch_weight * pl + roof_weight * rl, "orientation": ce, "pitch": pl, "roof": rl}
