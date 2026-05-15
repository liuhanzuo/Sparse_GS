"""Thin wrapper around ``gsplat.rasterization``.

Returns a dict ``{rgb, depth, alpha, info}`` so the trainer + future SSL
losses can read whatever channel they need without re-rendering.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from gsplat import rasterization

from ..models.gaussians import GaussianModel


class GSplatRenderer:
    def __init__(
        self,
        sh_degree: int,
        near_plane: float = 0.01,
        far_plane: float = 1.0e10,
        rasterize_mode: str = "antialiased",
        packed: bool = True,
        absgrad: bool = True,
        render_mode: str = "RGB+ED",
    ) -> None:
        self.sh_degree = int(sh_degree)
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.rasterize_mode = rasterize_mode
        self.packed = bool(packed)
        self.absgrad = bool(absgrad)
        self.render_mode = render_mode

    @torch.no_grad()
    def _validate_render_mode(self) -> None:
        assert self.render_mode in {"RGB", "D", "ED", "RGB+D", "RGB+ED"}

    def render(
        self,
        gaussians: GaussianModel,
        viewmat: torch.Tensor,        # (4,4) world->camera, OpenCV
        K: torch.Tensor,              # (3,3)
        width: int,
        height: int,
        active_sh_degree: Optional[int] = None,
        background: Optional[torch.Tensor] = None,
        detach_geometry: bool = False,
    ) -> Dict[str, Any]:
        self._validate_render_mode()
        st = gaussians.state_for_render(detach_geometry=detach_geometry)

        # Add batch dim: gsplat wants (C, 4, 4) and (C, 3, 3).
        viewmats = viewmat.unsqueeze(0).contiguous()
        Ks = K.unsqueeze(0).contiguous()

        sh_deg = gaussians.sh_degree if active_sh_degree is None else int(active_sh_degree)

        bg = None
        if background is not None:
            bg = background
            # gsplat 1.5.x expects ``backgrounds`` of shape ``(C, channels)``
            # regardless of ``packed=True/False``. (Internally it concatenates
            # a ``(C, 1)`` zero column for the depth channel in RGB+ED mode,
            # which fails with shape mismatch if we pass a 1D ``(3,)``.)
            # Depth-only renders ignore backgrounds.
            if bg.dim() > 1:
                bg = bg.flatten()                                 # (3,)
            if self.render_mode in ("D", "ED"):
                bg = None
            else:
                bg = bg.unsqueeze(0).contiguous()                 # (C=1, 3)

        render_colors, render_alphas, info = rasterization(
            means=st["means"],
            quats=st["quats"],
            scales=st["scales"],
            opacities=st["opacities"],
            colors=st["colors"],
            viewmats=viewmats,
            Ks=Ks,
            width=int(width),
            height=int(height),
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            sh_degree=sh_deg,
            packed=self.packed,
            absgrad=self.absgrad,
            rasterize_mode=self.rasterize_mode,
            backgrounds=bg,
            render_mode=self.render_mode,
        )

        # Split RGB / depth depending on render_mode.
        # render_colors: (C, H, W, channels)
        rgb = depth = None
        if self.render_mode == "RGB":
            rgb = render_colors[..., :3]
        elif self.render_mode in ("D", "ED"):
            depth = render_colors[..., 0:1]
        elif self.render_mode in ("RGB+D", "RGB+ED"):
            rgb = render_colors[..., :3]
            depth = render_colors[..., 3:4]

        return {
            "rgb": rgb.squeeze(0) if rgb is not None else None,        # (H,W,3)
            "depth": depth.squeeze(0) if depth is not None else None,  # (H,W,1)
            "alpha": render_alphas.squeeze(0),                          # (H,W,1)
            "info": info,
        }
