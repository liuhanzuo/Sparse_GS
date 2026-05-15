"""Wrap ``gsplat.DefaultStrategy`` together with its required per-parameter
Adam optimizers.

Why per-parameter optimizers? ``DefaultStrategy.step_post_backward`` mutates
optimizer state (Adam's running moments) *in place* during densification
and pruning. To do that reliably, gsplat needs one optimizer per Gaussian
parameter — this is the same pattern as gsplat's official
``simple_trainer``. Sharing one optimizer across all params will silently
break state tracking on densify.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from gsplat import DefaultStrategy

from ..models.gaussians import GaussianModel


# Per-parameter learning rates (read from cfg["optim"] keys named ``<name>_lr``).
_LR_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def build_strategy_and_optimizers(
    gaussians: GaussianModel,
    cfg: Dict[str, Any],
    scene_scale: float,
) -> Tuple[DefaultStrategy, Dict[str, torch.optim.Optimizer], Dict[str, Any]]:
    """Returns (strategy, optimizers_by_param_name, strategy_state)."""

    strat_cfg = cfg.get("strategy", {})
    optim_cfg = cfg.get("optim", {})

    if str(strat_cfg.get("type", "default")).lower() != "default":
        raise NotImplementedError(
            "only DefaultStrategy is wired in this baseline; "
            "MCMCStrategy etc. left for a later commit."
        )

    strategy = DefaultStrategy(
        prune_opa=float(strat_cfg.get("prune_opa", 0.005)),
        grow_grad2d=float(strat_cfg.get("grow_grad2d", 0.0002)),
        grow_scale3d=float(strat_cfg.get("grow_scale3d", 0.01)),
        prune_scale3d=float(strat_cfg.get("prune_scale3d", 0.1)),
        refine_start_iter=int(strat_cfg.get("refine_start_iter", 500)),
        refine_stop_iter=int(strat_cfg.get("refine_stop_iter", 15000)),
        refine_every=int(strat_cfg.get("refine_every", 100)),
        reset_every=int(strat_cfg.get("reset_every", 3000)),
        absgrad=bool(strat_cfg.get("absgrad", True)),
        revised_opacity=bool(strat_cfg.get("revised_opacity", False)),
        verbose=bool(strat_cfg.get("verbose", False)),
    )

    # gsplat 1.5: check_sanity raises if any param/optimizer is missing.
    eps = float(optim_cfg.get("eps", 1e-15))
    optimizers: Dict[str, torch.optim.Optimizer] = {}
    for name in _LR_KEYS:
        if name not in gaussians.params:
            raise KeyError(f"GaussianModel is missing '{name}' parameter")
        lr = float(optim_cfg.get(f"{name}_lr"))
        # gsplat's convention: scale the means LR by the scene extent so that
        # the same relative motion is allowed across scenes of different size.
        if name == "means":
            lr = lr * float(scene_scale)
        optimizers[name] = torch.optim.Adam(
            [{"params": [gaussians.params[name]], "name": name, "lr": lr}],
            eps=eps,
        )

    strategy.check_sanity(gaussians.params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=float(scene_scale))

    # Optional hard cap on Gaussian count. ``DefaultStrategy`` itself has no
    # ``cap_max`` (only ``MCMCStrategy`` does), but on Windows + recent CUDA
    # drivers a runaway Gaussian count is one of the most common sources of
    # transient TDR resets on long single rasterization kernels. We expose
    # ``strategy.cap_max`` purely as a tag attached to the state dict; the
    # trainer reads it back and prunes by lowest opacity after every
    # ``step_post_backward`` call (see ``Trainer._enforce_cap_max``).
    cap_max = strat_cfg.get("cap_max", None)
    if cap_max is not None:
        strategy_state["cap_max"] = int(cap_max)
    return strategy, optimizers, strategy_state
