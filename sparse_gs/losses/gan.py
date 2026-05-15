"""Small PatchGAN regularizer for appearance-only sparse-view experiments.

This module is deliberately conservative: the discriminator operates on RGB
patches, while the generator-side adversarial loss can be applied to a
separate render with geometry detached. That lets the loss sharpen appearance
without directly steering Gaussian means/scales/quats or gsplat's densification
statistics.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _hwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(2, 0, 1).unsqueeze(0).contiguous()


class PatchDiscriminator(nn.Module):
    """A compact PatchGAN discriminator with optional spectral norm.

    Two architectures are supported via ``arch``:

    - ``"default"`` (legacy): ``n_layers`` 4x4-stride-2 convs followed by a
      3x3-stride-1 head. With ``n_layers=3`` the actual receptive field on
      input is ~38x38 — useful but not the canonical pix2pix PatchGAN.

    - ``"pix2pix70"``: faithful 70x70 PatchGAN from Isola et al. The
      structure is::

          conv 4x4 s=2 (C64)   -> LReLU
          conv 4x4 s=2 (C128)  -> LReLU
          conv 4x4 s=2 (C256)  -> LReLU
          conv 4x4 s=1 (C512)  -> LReLU
          conv 4x4 s=1 (1)     (head, no activation)

      Any unit in the output has a 70x70 receptive field on the input,
      i.e. each output position scores a 70x70 image patch — exactly what
      we want for floater detection (very local artifacts) under sparse
      GS supervision.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        max_channels: int = 256,
        n_layers: int = 3,
        spectral_norm: bool = True,
        arch: str = "default",
    ) -> None:
        super().__init__()
        self.arch = str(arch).lower()

        def conv(cin: int, cout: int, stride: int, ksize: int = 4) -> nn.Module:
            pad = 1 if ksize == 4 else (ksize // 2)
            layer = nn.Conv2d(cin, cout, kernel_size=ksize, stride=stride, padding=pad)
            return nn.utils.spectral_norm(layer) if spectral_norm else layer

        if self.arch == "pix2pix70":
            # Strictly the canonical 70x70 PatchGAN. Channels are derived
            # from ``base_channels`` (default 32 -> 32/64/128/256, the
            # original pix2pix uses 64 -> 64/128/256/512; both work).
            c1 = base_channels
            c2 = min(base_channels * 2, max_channels)
            c3 = min(base_channels * 4, max_channels)
            c4 = min(base_channels * 8, max_channels)
            layers = [
                conv(in_channels, c1, stride=2), nn.LeakyReLU(0.2, inplace=True),
                conv(c1, c2, stride=2), nn.LeakyReLU(0.2, inplace=True),
                conv(c2, c3, stride=2), nn.LeakyReLU(0.2, inplace=True),
                conv(c3, c4, stride=1), nn.LeakyReLU(0.2, inplace=True),
                # head: 4x4 stride=1 to a single channel; receptive field 70.
                conv(c4, 1, stride=1, ksize=4),
            ]
            self.net = nn.Sequential(*layers)
        else:
            layers = [
                conv(in_channels, base_channels, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = base_channels
            for i in range(1, max(int(n_layers), 1)):
                nxt = min(base_channels * (2 ** i), max_channels)
                layers += [conv(ch, nxt, stride=2), nn.LeakyReLU(0.2, inplace=True)]
                ch = nxt
            head = nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1)
            layers.append(nn.utils.spectral_norm(head) if spectral_norm else head)
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchGANController:
    """Owns discriminator, optimizer, crops, augmentation, and GAN losses."""

    def __init__(self, cfg: Dict[str, Any], device: torch.device) -> None:
        self.cfg = cfg
        self.weight = float(cfg.get("weight", 0.001))
        self.start_iter = int(cfg.get("start_iter", 4000))
        self.every = max(1, int(cfg.get("every", 4)))
        self.patch_size = int(cfg.get("patch_size", 128))
        self.augment = bool(cfg.get("augment", True))
        self.detach_geometry = bool(cfg.get("detach_geometry", True))
        self.loss_type = str(cfg.get("loss", "hinge")).lower()

        self.disc = PatchDiscriminator(
            base_channels=int(cfg.get("base_channels", 32)),
            max_channels=int(cfg.get("max_channels", 256)),
            n_layers=int(cfg.get("n_layers", 3)),
            spectral_norm=bool(cfg.get("spectral_norm", True)),
        ).to(device)
        self.opt = torch.optim.Adam(
            self.disc.parameters(),
            lr=float(cfg.get("lr", 2.0e-4)),
            betas=(float(cfg.get("beta1", 0.0)), float(cfg.get("beta2", 0.99))),
        )

    def active(self, step: int) -> bool:
        return step >= self.start_iter and step % self.every == 0 and self.weight > 0.0

    def _set_requires_grad(self, enabled: bool) -> None:
        for p in self.disc.parameters():
            p.requires_grad_(enabled)

    def _crop_pair(self, real: torch.Tensor, fake: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return same random crop from real/fake BCHW tensors."""
        _, _, h, w = real.shape
        p = min(self.patch_size, h, w)
        if p <= 0:
            raise ValueError(f"invalid patch_size={self.patch_size} for image {h}x{w}")
        top = 0 if h == p else random.randint(0, h - p)
        left = 0 if w == p else random.randint(0, w - p)
        real_p = real[:, :, top:top + p, left:left + p]
        fake_p = fake[:, :, top:top + p, left:left + p]
        if self.augment and random.random() < 0.5:
            real_p = torch.flip(real_p, dims=[3])
            fake_p = torch.flip(fake_p, dims=[3])
        return real_p.contiguous(), fake_p.contiguous()

    def discriminator_step(self, real_hwc: torch.Tensor, fake_hwc: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        self._set_requires_grad(True)
        real = _hwc_to_bchw(real_hwc.detach().clamp(0.0, 1.0))
        fake = _hwc_to_bchw(fake_hwc.detach().clamp(0.0, 1.0))
        real_p, fake_p = self._crop_pair(real, fake)

        self.opt.zero_grad(set_to_none=True)
        real_logits = self.disc(real_p)
        fake_logits = self.disc(fake_p)
        if self.loss_type == "bce":
            d_loss = (
                F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            )
        else:
            d_loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
        d_loss.backward()
        self.opt.step()
        return d_loss.detach(), {
            "patch_gan/d_loss": float(d_loss.detach().item()),
            "patch_gan/real_logit": float(real_logits.detach().mean().item()),
            "patch_gan/fake_logit": float(fake_logits.detach().mean().item()),
        }

    def generator_loss(self, fake_hwc: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        self._set_requires_grad(False)
        fake = _hwc_to_bchw(fake_hwc.clamp(0.0, 1.0))
        _, fake_p = self._crop_pair(fake, fake)
        logits = self.disc(fake_p)
        if self.loss_type == "bce":
            raw = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        else:
            raw = -logits.mean()
        weighted = self.weight * raw
        self._set_requires_grad(True)
        return weighted, {
            "patch_gan/g_loss": float(weighted.detach().item()),
            "patch_gan/g_raw": float(raw.detach().item()),
        }
