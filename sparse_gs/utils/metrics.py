"""Image-quality metrics for training/eval.

We implement PSNR and SSIM in pure PyTorch so they are differentiable
(SSIM is also used inside the photometric loss as ``1 - SSIM``).

LPIPS is provided by the optional ``lpips`` PyPI package (VGG backbone).
If unavailable, :class:`LPIPSMetric` degrades to a no-op that returns
``None`` so callers can skip it without branching at every call site.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """PSNR in dB. Tensors are in [0,1] with shape (..., H, W, C) or (..., C, H, W)."""
    mse = (pred - gt).pow(2).mean()
    if mse <= 0:
        return torch.tensor(99.0, device=pred.device)
    return 10.0 * torch.log10(max_val * max_val / mse)


def _gauss_kernel(window_size: int = 11, sigma: float = 1.5, channels: int = 3,
                  device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    kernel_2d = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return kernel_2d


def ssim(pred: torch.Tensor, gt: torch.Tensor,
         window_size: int = 11, max_val: float = 1.0) -> torch.Tensor:
    """SSIM for 4D (B,C,H,W) tensors, returning the mean over batch + spatial."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
    assert pred.dim() == 4 and gt.shape == pred.shape, \
        f"shape mismatch: {pred.shape} vs {gt.shape}"

    C = pred.shape[1]
    kernel = _gauss_kernel(window_size, 1.5, C, device=pred.device, dtype=pred.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(gt,   kernel, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(gt   * gt,   kernel, padding=pad, groups=C) - mu2_sq
    sigma12   = F.conv2d(pred * gt,   kernel, padding=pad, groups=C) - mu1_mu2

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return (num / den).mean()


def hwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Convert a (H,W,C) or (B,H,W,C) tensor to NCHW."""
    if x.dim() == 3:
        return x.permute(2, 0, 1).unsqueeze(0).contiguous()
    if x.dim() == 4:
        return x.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"unexpected shape {tuple(x.shape)}")


# --------------------------------------------------------------------------
# LPIPS (optional, perceptual metric)
# --------------------------------------------------------------------------
class LPIPSMetric:
    """Thin wrapper around the ``lpips`` package (VGG backbone by default).

    The upstream package expects tensors in ``[-1, 1]`` with shape ``(B, 3, H, W)``.
    We accept ``(H, W, 3)`` in ``[0, 1]`` (our renderer's native layout) and
    convert internally.

    If the ``lpips`` package is not installed or the backbone fails to load
    (e.g. offline without cached weights), :attr:`available` is ``False`` and
    :meth:`__call__` returns ``None`` — callers should treat ``None`` as
    "skip this metric" rather than as an error.
    """

    def __init__(self, net: str = "vgg", device: Optional[torch.device] = None) -> None:
        self.net = net
        self.device = device
        self.model = None
        self.available = False
        self._err: Optional[str] = None
        try:
            import lpips as _lpips  # type: ignore
            # ``verbose=False`` suppresses the "Loading model from..." banner
            # that otherwise prints on every construction.
            self.model = _lpips.LPIPS(net=net, verbose=False).eval()
            if device is not None:
                self.model = self.model.to(device)
            # freeze
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.available = True
        except Exception as e:  # noqa: BLE001 — we intentionally swallow
            self._err = f"{type(e).__name__}: {e}"
            self.model = None
            self.available = False

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> Optional[float]:
        """Return LPIPS distance (lower=better), or ``None`` if unavailable.

        Inputs are expected in ``[0, 1]``, shape ``(H, W, 3)`` or ``(B, 3, H, W)``.
        """
        if not self.available or self.model is None:
            return None
        p = pred if pred.dim() == 4 else hwc_to_bchw(pred)
        g = gt   if gt.dim()   == 4 else hwc_to_bchw(gt)
        if self.device is not None:
            p = p.to(self.device); g = g.to(self.device)
        # [0,1] -> [-1,1]
        p = p.clamp(0.0, 1.0) * 2.0 - 1.0
        g = g.clamp(0.0, 1.0) * 2.0 - 1.0
        d = self.model(p, g)
        return float(d.mean().item())

    def __repr__(self) -> str:
        state = "ok" if self.available else f"disabled ({self._err})"
        return f"LPIPSMetric(net={self.net!r}, {state})"
