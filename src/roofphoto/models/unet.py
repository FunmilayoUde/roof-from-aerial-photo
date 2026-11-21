"""U-Net with a pretrained ResNet-34 encoder: orientation classes plus pitch, per pixel.

One network, 11 output channels: 10 orientation logits (see dataset.py) and 1 pitch channel.
Pitch is predicted in units of 45° so it sits on the same scale as the logits early in training.

Losses, both skipping pixels without a label:
  orientation  cross-entropy, ignore_index 255 (neighbours, low-confidence facets)
  pitch        smooth L1, only on roof pixels of reliable facets (NaN target elsewhere)
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
        classes=N_CLASSES + 1,
    )


def split(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(logits (B, 10, H, W), pitch in degrees (B, H, W))."""
    return out[:, :N_CLASSES], out[:, N_CLASSES] * PITCH_SCALE_DEG


def loss(out: torch.Tensor, orient: torch.Tensor, pitch: torch.Tensor, pitch_weight: float = 1.0) -> dict:
    logits, pitch_pred = split(out)
    ce = F.cross_entropy(logits, orient, ignore_index=IGNORE)
    has_pitch = torch.isfinite(pitch)
    if has_pitch.any():
        pl = F.smooth_l1_loss(pitch_pred[has_pitch] / PITCH_SCALE_DEG, pitch[has_pitch] / PITCH_SCALE_DEG)
    else:
        pl = torch.zeros((), device=out.device)
    return {"total": ce + pitch_weight * pl, "orientation": ce, "pitch": pl}
