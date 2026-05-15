"""Differentiable Augmentation for Data-Efficient GAN Training (DiffAug).

Based on Zhao et al., NeurIPS 2020 (arXiv:2006.10738). Specialised for our
PVD setup where:

* "real" = ground-truth training-view RGB (only 8 of them, very few
  samples) and "fake" = pseudo-view renders. Both go through the same
  Patch-GAN discriminator, so we MUST apply identical augmentations to
  real and fake otherwise the discriminator learns to detect the
  augmentation rather than floater artifacts.

* All operations are differentiable so gradients flow back to G when the
  generator-side adversarial loss is enabled.

Three policies are supported (toggle independently via the cfg):

    ``color``        — random brightness / saturation / contrast
    ``translation``  — small affine shift (zero-padded), helps low-data
    ``cutout``       — randomly zero out a rectangular region

A single augmentation pass is parameterised by a sampled ``params`` dict
so it can be applied to multiple tensors with the *same* randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DiffAugConfig:
    color: bool = True
    translation: float = 0.125  # max shift as a fraction of H/W
    cutout: float = 0.5         # cutout side as a fraction of H/W (0 disables)
    brightness: float = 0.5     # +/- this in [0,1] space (clipped later)
    saturation: float = 2.0     # multiplicative
    contrast: float = 1.5       # multiplicative

    @classmethod
    def from_cfg(cls, cfg: Optional[Dict[str, Any]]) -> "DiffAugConfig":
        cfg = cfg or {}
        return cls(
            color=bool(cfg.get("color", True)),
            translation=float(cfg.get("translation", 0.125)),
            cutout=float(cfg.get("cutout", 0.5)),
            brightness=float(cfg.get("brightness", 0.5)),
            saturation=float(cfg.get("saturation", 2.0)),
            contrast=float(cfg.get("contrast", 1.5)),
        )

    @property
    def enabled(self) -> bool:
        return self.color or self.translation > 0 or self.cutout > 0


# ----------------------------------------------------------------------
# parameter sampling (one set per call; shared across real/fake pairs)
# ----------------------------------------------------------------------


def sample_params(
    cfg: DiffAugConfig,
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Sample one batch of augmentation parameters on ``device``."""
    rand = lambda *shape: torch.rand(*shape, device=device, generator=generator)
    out: Dict[str, torch.Tensor] = {}
    if cfg.color:
        # brightness in [-b, +b]
        out["bright"] = (rand(batch, 1, 1, 1) - 0.5) * 2.0 * cfg.brightness
        # saturation factor in [1-s, 1+s] (s is e.g. 2 -> [−1, 3], we clip)
        out["sat"] = rand(batch, 1, 1, 1) * cfg.saturation
        # contrast factor in [0.5, 1.5] for default contrast=1.5
        out["cont"] = rand(batch, 1, 1, 1) * cfg.contrast + (1.0 - cfg.contrast / 2.0)
    if cfg.translation > 0:
        max_dx = max(1, int(width * cfg.translation))
        max_dy = max(1, int(height * cfg.translation))
        out["dx"] = torch.randint(
            -max_dx, max_dx + 1, (batch, 1, 1), device=device, generator=generator,
        )
        out["dy"] = torch.randint(
            -max_dy, max_dy + 1, (batch, 1, 1), device=device, generator=generator,
        )
    if cfg.cutout > 0:
        cut_h = max(1, int(height * cfg.cutout))
        cut_w = max(1, int(width * cfg.cutout))
        out["cut_y"] = torch.randint(0, height, (batch, 1, 1), device=device, generator=generator)
        out["cut_x"] = torch.randint(0, width, (batch, 1, 1), device=device, generator=generator)
        out["cut_h"] = torch.tensor(cut_h, device=device)
        out["cut_w"] = torch.tensor(cut_w, device=device)
    return out


# ----------------------------------------------------------------------
# apply
# ----------------------------------------------------------------------


def _color(x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "bright" not in params:
        return x
    x = x + params["bright"]
    mean = x.mean(dim=1, keepdim=True)
    x = (x - mean) * params["sat"] + mean
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - mean) * params["cont"] + mean
    return x


def _translation(x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "dx" not in params:
        return x
    b, c, h, w = x.shape
    dx = params["dx"].view(b)
    dy = params["dy"].view(b)
    # build grids
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device),
        torch.arange(w, device=x.device),
        indexing="ij",
    )
    # shifted indices per-batch
    yy = yy.unsqueeze(0).expand(b, -1, -1) - dy.view(b, 1, 1)
    xx = xx.unsqueeze(0).expand(b, -1, -1) - dx.view(b, 1, 1)
    # zero-pad: invalid positions -> we'll mask
    valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    yy_c = yy.clamp(0, h - 1)
    xx_c = xx.clamp(0, w - 1)
    # gather: build flat index
    idx = (yy_c * w + xx_c).unsqueeze(1).expand(-1, c, -1, -1)
    flat = x.reshape(b, c, h * w)
    out = torch.gather(flat, 2, idx.reshape(b, c, h * w)).reshape(b, c, h, w)
    out = out * valid.unsqueeze(1).to(out.dtype)
    return out


def _cutout(x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "cut_y" not in params:
        return x
    b, c, h, w = x.shape
    cy = params["cut_y"].view(b)
    cx = params["cut_x"].view(b)
    ch_ = int(params["cut_h"].item())
    cw_ = int(params["cut_w"].item())
    # build mask: 1 outside the cut box, 0 inside
    yy = torch.arange(h, device=x.device).view(1, h, 1).expand(b, -1, w)
    xx = torch.arange(w, device=x.device).view(1, 1, w).expand(b, h, -1)
    y0 = (cy - ch_ // 2).clamp(0, h - 1).view(b, 1, 1)
    y1 = (cy + ch_ // 2).clamp(0, h - 1).view(b, 1, 1)
    x0 = (cx - cw_ // 2).clamp(0, w - 1).view(b, 1, 1)
    x1 = (cx + cw_ // 2).clamp(0, w - 1).view(b, 1, 1)
    inside = (yy >= y0) & (yy <= y1) & (xx >= x0) & (xx <= x1)
    mask = (~inside).to(x.dtype).unsqueeze(1)  # (b,1,h,w)
    return x * mask


def apply(
    x: torch.Tensor, cfg: DiffAugConfig, params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply DiffAug to a BCHW tensor using the pre-sampled ``params``."""
    if not cfg.enabled:
        return x
    if cfg.color:
        x = _color(x, params)
    if cfg.translation > 0:
        x = _translation(x, params)
    if cfg.cutout > 0:
        x = _cutout(x, params)
    return x


def apply_pair(
    real: torch.Tensor,
    fake: torch.Tensor,
    cfg: DiffAugConfig,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample one set of params and apply identically to ``real`` and ``fake``.

    real and fake must share spatial dims (BCHW); they typically come from
    the co-cropped patches in PVD.
    """
    if not cfg.enabled:
        return real, fake
    b, _, h, w = real.shape
    assert fake.shape[0] == b and fake.shape[2] == h and fake.shape[3] == w, \
        f"shape mismatch real={tuple(real.shape)} fake={tuple(fake.shape)}"
    params = sample_params(cfg, b, h, w, real.device, generator=generator)
    return apply(real, cfg, params), apply(fake, cfg, params)
