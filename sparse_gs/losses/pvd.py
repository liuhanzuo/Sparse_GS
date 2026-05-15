"""Pseudo-View Discriminator (PVD).

Motivation
----------
Earlier ``PatchGANController`` paired ``real = train-view GT`` with
``fake = train-view render``. Under sparse-view 3DGS, training-view
renders already match the GT closely (PSNR ~30+) so the discriminator
sees almost no signal — adversarial loss collapsed and helped nothing.

PVD swaps the *negative* distribution to **pseudo-view renders**, which
is exactly where floaters / ghost artifacts surface most strongly under
n=8 supervision. The discriminator now learns a meaningful "real scene
texture vs. floater-corrupted render" boundary, and we can use its
spatial logit map as a soft floater map to *guide* depth-based pruning.

Two consumers
~~~~~~~~~~~~~
1. ``generator_loss(fake_pseudo_rgb)`` — optional G-side adversarial
   term (small weight; geometry can be detached for safety).
2. ``floater_score_map(fake_pseudo_rgb)`` — full-resolution per-pixel
   "fakeness" score in [0, 1]; used by the trainer to bias which pixels
   contribute to the SparseGS-style floater mask.

The discriminator architecture is reused from ``PatchDiscriminator`` so
we don't introduce a second model on disk.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from . import diffaug as _diffaug
from .gan import PatchDiscriminator, _hwc_to_bchw


class PseudoViewDiscriminator:
    """Owns D, optimizer, training/inference paths against pseudo views.

    Config keys (all optional, with sane defaults)::

        ssl:
          pvd:
            enabled: true
            start_iter: 2000      # delay until densify has produced enough
            every: 4              # train D every K iterations
            weight: 0.0           # G-side adv weight (0 = D-only, no G push)
            patch_size: 128
            augment: true
            detach_geometry: true # G adv term must not steer geometry
            loss: hinge           # hinge | bce
            lr: 2.0e-4
            beta1: 0.0
            beta2: 0.99
            base_channels: 32
            n_layers: 3
            spectral_norm: true

            # ---- floater-mask coupling (the whole point) ----
            floater_guidance:
              enabled: true
              gate_thresh: 0.6    # only pixels with sigmoid(logit) > thresh
                                  # are treated as "fake" by floater stage
              warmup_iter: 4000   # don't trust D before this step
              min_fake_logit_gap: 0.5   # also require D actually distinguishes
                                          (mean(real) - mean(fake) >= gap)

    The "guidance" path is opt-in: when ``floater_guidance.enabled`` is
    True and D is past warmup *and* meets the discrimination-gap check,
    the trainer multiplies its raw floater-pixel mask by the D-derived
    spatial fake-prob mask. Pixels that D thinks are real get exempted
    from being declared floaters.
    """

    def __init__(self, cfg: Dict[str, Any], device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.enabled = bool(cfg.get("enabled", False))
        self.start_iter = int(cfg.get("start_iter", 2000))
        self.every = max(1, int(cfg.get("every", 4)))
        self.weight = float(cfg.get("weight", 0.0))
        self.patch_size = int(cfg.get("patch_size", 128))
        self.augment = bool(cfg.get("augment", True))
        self.detach_geometry = bool(cfg.get("detach_geometry", True))
        self.loss_type = str(cfg.get("loss", "hinge")).lower()
        self.arch = str(cfg.get("arch", "default")).lower()

        # DiffAug: stronger, differentiable, real/fake-shared augmentation
        # to compensate for our tiny (n=8) real GT distribution. Disabled
        # by default for backward compat with v4 configs.
        diffaug_cfg = cfg.get("augment_diffaug", None)
        if isinstance(diffaug_cfg, dict) and bool(diffaug_cfg.get("enabled", False)):
            self.diffaug_cfg = _diffaug.DiffAugConfig.from_cfg(diffaug_cfg)
        else:
            self.diffaug_cfg = None

        self.disc = PatchDiscriminator(
            base_channels=int(cfg.get("base_channels", 32)),
            max_channels=int(cfg.get("max_channels", 256)),
            n_layers=int(cfg.get("n_layers", 3)),
            spectral_norm=bool(cfg.get("spectral_norm", True)),
            arch=self.arch,
        ).to(device)
        self.opt = torch.optim.Adam(
            self.disc.parameters(),
            lr=float(cfg.get("lr", 2.0e-4)),
            betas=(float(cfg.get("beta1", 0.0)), float(cfg.get("beta2", 0.99))),
        )

        # Running discrimination-gap monitor (real_logit - fake_logit).
        # Used by floater_guidance gate to refuse trusting an untrained D.
        self._ema_real_logit: Optional[float] = None
        self._ema_fake_logit: Optional[float] = None
        self._ema_alpha = 0.05

        guid = cfg.get("floater_guidance", {}) or {}
        self.guidance_enabled = bool(guid.get("enabled", False))
        self.guidance_thresh = float(guid.get("gate_thresh", 0.6))
        self.guidance_warmup = int(guid.get("warmup_iter", self.start_iter + 2000))
        self.guidance_min_gap = float(guid.get("min_fake_logit_gap", 0.5))

    # ------------------------------------------------------------------
    # scheduling
    # ------------------------------------------------------------------
    def active(self, step: int) -> bool:
        return self.enabled and step >= self.start_iter and step % self.every == 0

    def guidance_active(self, step: int) -> bool:
        if not (self.enabled and self.guidance_enabled):
            return False
        if step < self.guidance_warmup:
            return False
        if self._ema_real_logit is None or self._ema_fake_logit is None:
            return False
        return (self._ema_real_logit - self._ema_fake_logit) >= self.guidance_min_gap

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _set_requires_grad(self, enabled: bool) -> None:
        for p in self.disc.parameters():
            p.requires_grad_(enabled)

    def _crop_pair(
        self, real: torch.Tensor, fake: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random co-cropped patches from real/fake BCHW tensors.

        Real and fake do NOT need to be the same image since they come
        from different cameras (train GT vs. pseudo render); we still
        share the spatial crop window for stability.
        """
        _, _, hr, wr = real.shape
        _, _, hf, wf = fake.shape
        h = min(hr, hf)
        w = min(wr, wf)
        p = min(self.patch_size, h, w)
        if p <= 0:
            raise ValueError(f"invalid patch_size={self.patch_size}")
        top_r = 0 if hr == p else random.randint(0, hr - p)
        left_r = 0 if wr == p else random.randint(0, wr - p)
        top_f = 0 if hf == p else random.randint(0, hf - p)
        left_f = 0 if wf == p else random.randint(0, wf - p)
        real_p = real[:, :, top_r:top_r + p, left_r:left_r + p].contiguous()
        fake_p = fake[:, :, top_f:top_f + p, left_f:left_f + p].contiguous()
        if self.augment and random.random() < 0.5:
            real_p = torch.flip(real_p, dims=[3])
            fake_p = torch.flip(fake_p, dims=[3])
        if self.diffaug_cfg is not None:
            real_p, fake_p = _diffaug.apply_pair(real_p, fake_p, self.diffaug_cfg)
        return real_p, fake_p

    # ------------------------------------------------------------------
    # public: D step (real = train GT, fake = pseudo-view render)
    # ------------------------------------------------------------------
    def discriminator_step(
        self, real_train_gt_hwc: torch.Tensor, fake_pseudo_rgb_hwc: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self._set_requires_grad(True)
        real = _hwc_to_bchw(real_train_gt_hwc.detach().clamp(0.0, 1.0))
        fake = _hwc_to_bchw(fake_pseudo_rgb_hwc.detach().clamp(0.0, 1.0))
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

        rl = float(real_logits.detach().mean().item())
        fl = float(fake_logits.detach().mean().item())
        # EMA update of the discrimination gap
        a = self._ema_alpha
        self._ema_real_logit = rl if self._ema_real_logit is None else (1 - a) * self._ema_real_logit + a * rl
        self._ema_fake_logit = fl if self._ema_fake_logit is None else (1 - a) * self._ema_fake_logit + a * fl

        gap = (self._ema_real_logit or 0.0) - (self._ema_fake_logit or 0.0)
        return d_loss.detach(), {
            "pvd/d_loss": float(d_loss.detach().item()),
            "pvd/real_logit": rl,
            "pvd/fake_logit": fl,
            "pvd/ema_gap": gap,
        }

    # ------------------------------------------------------------------
    # public: G adversarial term (optional; weight=0 means D-only mode)
    # ------------------------------------------------------------------
    def generator_loss(
        self, fake_pseudo_rgb_hwc: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.weight <= 0.0:
            return torch.zeros((), device=fake_pseudo_rgb_hwc.device), {}
        self._set_requires_grad(False)
        fake = _hwc_to_bchw(fake_pseudo_rgb_hwc.clamp(0.0, 1.0))
        # For the G term we only need a single random crop of the fake
        _, _, hf, wf = fake.shape
        p = min(self.patch_size, hf, wf)
        top = 0 if hf == p else random.randint(0, hf - p)
        left = 0 if wf == p else random.randint(0, wf - p)
        fake_p = fake[:, :, top:top + p, left:left + p].contiguous()
        # Apply DiffAug to keep G-side input distribution consistent with
        # what D was trained on; sampling differs between iterations but
        # within this call real/fake-shared params are not needed (no real).
        if self.diffaug_cfg is not None:
            params = _diffaug.sample_params(
                self.diffaug_cfg, fake_p.shape[0], fake_p.shape[2], fake_p.shape[3],
                fake_p.device,
            )
            fake_p = _diffaug.apply(fake_p, self.diffaug_cfg, params)

        logits = self.disc(fake_p)
        if self.loss_type == "bce":
            raw = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        else:
            raw = -logits.mean()
        weighted = self.weight * raw
        self._set_requires_grad(True)
        return weighted, {
            "pvd/g_loss": float(weighted.detach().item()),
            "pvd/g_raw": float(raw.detach().item()),
        }

    # ------------------------------------------------------------------
    # public: per-pixel "fake probability" map for floater guidance
    # ------------------------------------------------------------------
    @torch.no_grad()
    def floater_score_map(
        self, rgb_hwc: torch.Tensor, target_h: int, target_w: int,
    ) -> torch.Tensor:
        """Score map in [0, 1]; high value = D thinks the pixel is fake.

        Returns a (H, W) float32 tensor on rgb_hwc.device, upsampled to
        ``(target_h, target_w)``. The PatchGAN architecture produces a
        spatial logit map at ~stride 2**n_layers; we bilinearly upsample
        and pass through sigmoid (so "more positive logit" = "more real",
        and we return 1 - sigmoid = "fakeness").
        """
        self._set_requires_grad(False)
        x = _hwc_to_bchw(rgb_hwc.detach().clamp(0.0, 1.0))
        logits = self.disc(x)             # (1, 1, h', w')
        # Higher D-logit means "real". We want "fakeness".
        fake_prob = torch.sigmoid(-logits)
        fake_prob = F.interpolate(
            fake_prob, size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        )
        self._set_requires_grad(True)
        return fake_prob[0, 0].contiguous()

    # ------------------------------------------------------------------
    # state dict (so a future v5 can resume without retraining D)
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "disc": self.disc.state_dict(),
            "opt": self.opt.state_dict(),
            "ema_real_logit": self._ema_real_logit,
            "ema_fake_logit": self._ema_fake_logit,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.disc.load_state_dict(d["disc"])
        self.opt.load_state_dict(d["opt"])
        self._ema_real_logit = d.get("ema_real_logit")
        self._ema_fake_logit = d.get("ema_fake_logit")
