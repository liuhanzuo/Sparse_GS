"""GaussianModel — the only learnable state of a 3DGS scene.

We follow the parameter layout used by gsplat's official ``simple_trainer``:

    means      : (N, 3)        world-space xyz
    scales     : (N, 3)        log-space, exp() before rasterization
    quats      : (N, 4)        wxyz, normalized inside gsplat
    opacities  : (N,)          sigmoid space, sigmoid() before rasterization
    sh0        : (N, 1, 3)     DC SH coefficient (band 0)
    shN        : (N, K, 3)     higher-order SH (K = (sh_degree+1)^2 - 1)

This is *exactly* the parameter dict that ``DefaultStrategy`` knows how to
densify / prune in-place, so we can hand `self.params` directly to the
strategy. Crucially, every parameter must have its OWN optimizer (handled
in ``strategies/densify.py``).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def _rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB in [0,1] to the DC SH coefficient used by gsplat.

    gsplat's SH evaluation multiplies sh0 by the DC SH basis ``C0 = 0.282095``
    (i.e. ``1 / (2*sqrt(pi))``) and then adds ``0.5`` before clamping. So to
    initialize colors equal to ``rgb`` we set ``sh0 = (rgb - 0.5) / C0``.
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


class GaussianModel(nn.Module):
    """Holds Gaussian parameters as ``nn.ParameterDict``.

    Use ``model.params`` to pass to ``gsplat`` strategies / optimizers.
    """

    def __init__(self, sh_degree: int = 3):
        super().__init__()
        self.sh_degree = int(sh_degree)
        self.active_sh_degree = 0          # we ramp this up during training
        self.params: nn.ParameterDict = nn.ParameterDict()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_random(
        self,
        num_points: int,
        extent: float = 1.5,
        rgb_init: float = 0.5,
        scale_init_factor: float = 0.01,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize uniformly inside an axis-aligned cube of half-side ``extent``."""
        N = int(num_points)
        K = (self.sh_degree + 1) ** 2 - 1   # higher-order SH count

        means = (torch.rand(N, 3) * 2.0 - 1.0) * extent
        # Isotropic scale: ``scale_init_factor`` in world units, in log-space.
        log_scale = math.log(max(scale_init_factor, 1e-6))
        scales = torch.full((N, 3), log_scale)
        quats = torch.zeros(N, 4); quats[:, 0] = 1.0          # identity (wxyz)
        # opacity ~ 0.1 in real space => logit(0.1) ≈ -2.197
        opacities = torch.full((N,), float(np.log(0.1 / 0.9)))
        rgb = torch.full((N, 3), float(rgb_init))
        sh0 = _rgb_to_sh0(rgb).unsqueeze(1)                    # (N,1,3)
        shN = torch.zeros(N, K, 3)

        self._set_params(means, scales, quats, opacities, sh0, shN, device=device)

    @torch.no_grad()
    def init_from_points(
        self,
        xyz: np.ndarray | torch.Tensor,
        rgb: Optional[np.ndarray | torch.Tensor] = None,
        scale_init_factor: float = 0.01,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize from an explicit point cloud (e.g. SfM output)."""
        xyz_t = torch.as_tensor(xyz, dtype=torch.float32)
        N = xyz_t.shape[0]
        K = (self.sh_degree + 1) ** 2 - 1
        if rgb is None:
            rgb_t = torch.full((N, 3), 0.5)
        else:
            rgb_t = torch.as_tensor(rgb, dtype=torch.float32)
            if rgb_t.max() > 1.5:               # 0..255 -> 0..1
                rgb_t = rgb_t / 255.0
        log_scale = math.log(max(scale_init_factor, 1e-6))
        scales = torch.full((N, 3), log_scale)
        quats = torch.zeros(N, 4); quats[:, 0] = 1.0
        opacities = torch.full((N,), float(np.log(0.1 / 0.9)))
        sh0 = _rgb_to_sh0(rgb_t).unsqueeze(1)
        shN = torch.zeros(N, K, 3)
        self._set_params(xyz_t, scales, quats, opacities, sh0, shN, device=device)

    def _set_params(self, means, scales, quats, opacities, sh0, shN, device) -> None:
        device = torch.device(device)
        self.params = nn.ParameterDict({
            "means":     nn.Parameter(means.to(device).contiguous()),
            "scales":    nn.Parameter(scales.to(device).contiguous()),
            "quats":     nn.Parameter(quats.to(device).contiguous()),
            "opacities": nn.Parameter(opacities.to(device).contiguous()),
            "sh0":       nn.Parameter(sh0.to(device).contiguous()),
            "shN":       nn.Parameter(shN.to(device).contiguous()),
        })

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    @property
    def num_points(self) -> int:
        return int(self.params["means"].shape[0]) if "means" in self.params else 0

    def state_for_render(self, detach_geometry: bool = False) -> Dict[str, torch.Tensor]:
        """Decoded tensors ready for ``gsplat.rasterization``.

        Note we do *not* exp / sigmoid here — gsplat expects raw scales and
        opacities? No — gsplat expects already-decoded ``scales`` (positive)
        and ``opacities`` in [0,1]. So we apply the activations here.

        Args:
            detach_geometry: when True, ``means / scales / quats`` are
                detached from the autograd graph. Used by SSL passes that
                should not affect the geometry parameters (and therefore
                must not feed gsplat's density-control 2D-mean gradient).
                Colors (sh0, shN) and opacities remain trainable.
        """
        means = self.params["means"]
        scales = torch.exp(self.params["scales"])
        quats = self.params["quats"]                       # gsplat normalizes internally
        opacities = torch.sigmoid(self.params["opacities"])
        sh0 = self.params["sh0"]
        shN = self.params["shN"]
        if detach_geometry:
            means = means.detach()
            scales = scales.detach()
            quats = quats.detach()
        colors = torch.cat([sh0, shN], dim=1)              # (N, K+1, 3)
        return {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "colors": colors,
        }

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------
    def save(self, path) -> None:
        torch.save({
            "sh_degree": self.sh_degree,
            "active_sh_degree": self.active_sh_degree,
            "params": {k: v.detach().cpu() for k, v in self.params.items()},
        }, str(path))

    def load(self, path, device="cuda") -> None:
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        self.sh_degree = int(state["sh_degree"])
        self.active_sh_degree = int(state.get("active_sh_degree", self.sh_degree))
        p = state["params"]
        self._set_params(
            p["means"], p["scales"], p["quats"],
            p["opacities"], p["sh0"], p["shN"],
            device=device,
        )
