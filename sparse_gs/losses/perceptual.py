"""Perceptual (LPIPS-VGG) loss usable as a *train-time* regularizer.

Unlike :class:`sparse_gs.utils.metrics.LPIPSMetric` (which is wrapped in
``@torch.no_grad`` and used only at eval), this module returns a tensor
that participates in the backward graph so the renderer is pushed toward
matching VGG features of the GT.

Why this exists
---------------
SparseGS / FSGS / FreeNeRF and friends all observe that a small
perceptual term on top of the L1 + SSIM photometric loss substantially
reduces "blurry / floaty" artefacts under sparse-view supervision —
because LPIPS-VGG features are far more sensitive to high-frequency
geometric mistakes than per-pixel L1 is.

The implementation re-uses the ``lpips`` PyPI package (same VGG weights
that are loaded for eval, so no extra download), but **re-creates the
network in train mode with all parameters frozen + ``requires_grad =
False``**, so gradients flow only into the inputs (the renderer's RGB
output) and not into the VGG weights themselves.

Cost: the LPIPS-VGG forward pass on a 800x800 image is ~30-50 ms on a
3060/4070 class GPU, so applying it every step roughly doubles the
per-iter time. To keep training time reasonable we expose:

* ``every`` — only compute every N steps (default 4)
* ``start_iter`` — only kick in after warmup
* ``downsample`` — bilinearly downsample to (e.g.) 256x256 first
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _hwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.permute(2, 0, 1).unsqueeze(0).contiguous()
    if x.dim() == 4:
        # already (B, H, W, C) or (B, C, H, W)?
        # We standardize on HWC inputs from the renderer, so dim==4 here
        # means (B, H, W, C).
        return x.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"perceptual: unexpected input shape {tuple(x.shape)}")


class PerceptualLoss:
    """Differentiable LPIPS-VGG perceptual loss for train-time use.

    Parameters live in ``cfg``; all keys are optional with sensible
    defaults:

    * ``weight`` (float, default 0.05): coefficient on the perceptual term
      in the total loss.
    * ``net`` (str, default ``"vgg"``): LPIPS backbone (``vgg`` / ``alex``).
    * ``every`` (int, default 4): only compute every N train steps.
    * ``start_iter`` (int, default 1000): warmup steps before kicking in.
    * ``downsample`` (int, default 0): if >0, bilinearly downsample the
      pair to this side length before LPIPS to save time.
    * ``patch_size`` (int, default 0): if >0, use a single random crop of
      this size instead of the full image (cheaper, also a form of
      regularization since LPIPS is patch-local anyway).

    Returns ``(0., {})`` when not active, so call sites can just do
    ``term, logs = ploss(step, render, gt)`` and add ``term`` to the
    total loss unconditionally.
    """

    def __init__(self, cfg: Dict[str, Any], device: torch.device) -> None:
        self.cfg = cfg or {}
        self.weight = float(self.cfg.get("weight", 0.05))
        self.net = str(self.cfg.get("net", "vgg"))
        self.every = max(1, int(self.cfg.get("every", 4)))
        self.start_iter = int(self.cfg.get("start_iter", 1000))
        self.downsample = int(self.cfg.get("downsample", 0))
        self.patch_size = int(self.cfg.get("patch_size", 0))
        self.device = device

        self.model = None
        self.available = False
        self._err: Optional[str] = None
        try:
            import lpips as _lpips  # type: ignore
            m = _lpips.LPIPS(net=self.net, verbose=False)
            # Freeze weights but keep eval mode so dropout/BN don't fluctuate.
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
            self.model = m.to(device)
            self.available = True
        except Exception as e:  # noqa: BLE001
            self._err = f"{type(e).__name__}: {e}"
            self.model = None
            self.available = False

    def __repr__(self) -> str:
        if not self.available:
            return f"PerceptualLoss(disabled: {self._err})"
        return (
            f"PerceptualLoss(net={self.net!r}, w={self.weight}, every={self.every}, "
            f"start_iter={self.start_iter}, downsample={self.downsample}, "
            f"patch_size={self.patch_size})"
        )

    # ------------------------------------------------------------------
    def active(self, step: int) -> bool:
        return (
            self.available
            and self.weight > 0.0
            and step >= self.start_iter
            and (step % self.every) == 0
        )

    # ------------------------------------------------------------------
    def _maybe_downsample(self, p: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.downsample <= 0:
            return p, g
        h, w = p.shape[-2:]
        side = max(h, w)
        if side <= self.downsample:
            return p, g
        scale = self.downsample / side
        new_h = max(int(round(h * scale)), 16)
        new_w = max(int(round(w * scale)), 16)
        p = F.interpolate(p, size=(new_h, new_w), mode="bilinear", align_corners=False)
        g = F.interpolate(g, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return p, g

    def _maybe_crop(self, p: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.patch_size <= 0:
            return p, g
        _, _, h, w = p.shape
        # VGG-LPIPS does 5 maxpools (factor 32 total) and then needs at
        # least 1x1 spatial output. Using <32 here would crash deep in
        # the VGG forward, so we floor the effective patch size.
        ps = min(self.patch_size, h, w)
        ps = max(ps, 32)
        if ps <= 0 or ps >= min(h, w):
            return p, g
        # same random crop for both — no augmentation here, that's GAN's job.
        top = torch.randint(0, h - ps + 1, (1,)).item()
        left = torch.randint(0, w - ps + 1, (1,)).item()
        p = p[:, :, top:top + ps, left:left + ps].contiguous()
        g = g[:, :, top:top + ps, left:left + ps].contiguous()
        return p, g

    # ------------------------------------------------------------------
    def __call__(
        self, step: int, render_hwc: torch.Tensor, gt_hwc: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return ``(weighted_loss, logs)``.

        When inactive returns a 0-tensor (still on the device, with
        ``requires_grad=False`` so adding it to the total loss is a
        zero-cost no-op). Logs are empty in the inactive case.
        """
        if not self.active(step):
            zero = torch.zeros((), device=self.device, dtype=render_hwc.dtype)
            return zero, {}

        # render keeps gradients; gt is detached anyway.
        p = _hwc_to_bchw(render_hwc)
        with torch.no_grad():
            g = _hwc_to_bchw(gt_hwc)

        p, g = self._maybe_downsample(p, g)
        p, g = self._maybe_crop(p, g)

        # LPIPS expects [-1, 1].
        p_n = p.clamp(0.0, 1.0) * 2.0 - 1.0
        g_n = g.clamp(0.0, 1.0) * 2.0 - 1.0

        d = self.model(p_n, g_n)              # (B, 1, 1, 1) typically
        raw = d.mean()
        weighted = self.weight * raw

        return weighted, {
            "perceptual/raw": float(raw.detach().item()),
            "perceptual/weighted": float(weighted.detach().item()),
        }
