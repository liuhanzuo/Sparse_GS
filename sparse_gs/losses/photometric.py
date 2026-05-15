"""Photometric loss: ``(1 - lambda) * L1 + lambda * (1 - SSIM)``.

This is the standard 3DGS reconstruction loss (Kerbl et al. 2023). All
inputs are HxWx3 in [0,1].
"""

from __future__ import annotations

import torch

from ..utils.metrics import hwc_to_bchw, ssim


def photometric_loss(
    pred_rgb: torch.Tensor,         # (H, W, 3) in [0, 1]
    gt_rgb: torch.Tensor,           # (H, W, 3) in [0, 1]
    ssim_lambda: float = 0.2,
) -> torch.Tensor:
    l1 = (pred_rgb - gt_rgb).abs().mean()
    if ssim_lambda <= 0.0:
        return l1
    s = ssim(hwc_to_bchw(pred_rgb), hwc_to_bchw(gt_rgb))
    return (1.0 - ssim_lambda) * l1 + ssim_lambda * (1.0 - s)
