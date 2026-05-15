"""Post-densify pruning utilities for W2 (unseen + floater prune).

This module is intentionally **pure-functional** and **rasterizer-agnostic**:

* It does **not** import ``gsplat`` and does **not** know how Gaussians are
  rendered. The trainer is responsible for collecting visibility / depth
  signals from whatever renderer it uses, and then calling the helpers here
  to derive the prune masks.
* All helpers run on CPU as well as CUDA (no kernel calls), so they can be
  unit-tested without a GPU.

Design notes
------------
1. ``compute_unseen_mask`` is the DNGaussian ``clean_views`` semantics:
   accumulate visibility across all *training* cameras, then prune Gaussians
   that were never seen by *any* of them.

2. ``compute_floater_mask`` is the **lightweight** SparseGS
   ``identify_floaters`` semantics: alpha-mask -> normalize depth -> take
   the lower ``thresh_bin`` quantile as floater pixels -> back-project to
   Gaussians via the per-pixel ``gaussian_ids`` (packed=True) or via a
   ``radii > 0`` projector (packed=False, single view fallback).
   We deliberately **drop** SparseGS' second-stage alpha re-projection
   (which depends on ``conic_opacity`` / per-mode IDs that gsplat 1.5.3
   does not expose).

3. ``apply_safety_cap`` is a hard guard against catastrophic over-pruning:
   if the prune mask covers more than ``max_ratio`` of all Gaussians, we
   refuse to apply it and let the caller log + skip. Empirically a single
   ``floater`` step pruning > 5% of the model usually means the depth
   histogram is bimodal in a *bad* way (unconverged geometry) and applying
   the mask deletes good points.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# 1) Unseen mask (DNGaussian-style clean_views)
# ---------------------------------------------------------------------------
def compute_unseen_mask(visibility_count: Tensor) -> Tensor:
    """Return a bool mask of Gaussians that were never visible to any camera.

    Args:
        visibility_count: (N,) int / long / float tensor with the number of
            cameras that saw each Gaussian. ``count == 0`` => unseen.

    Returns:
        (N,) bool tensor where ``True`` marks an unseen Gaussian
        (i.e. a candidate for pruning).
    """
    if visibility_count.dim() != 1:
        raise ValueError(
            f"visibility_count must be 1-D (N,), got shape {tuple(visibility_count.shape)}"
        )
    return visibility_count == 0


# ---------------------------------------------------------------------------
# 2) Floater mask (SparseGS-style identify_floaters, lightweight)
# ---------------------------------------------------------------------------
def _normalize_depth(depth: Tensor, eps: float = 1e-8) -> Tensor:
    """Min-max normalize a (H, W) depth map to [0, 1]."""
    d_min = depth.min()
    d_max = depth.max()
    return (depth - d_min) / (d_max - d_min + eps)


def _floater_pixel_mask_from_depth_alpha(
    depth_hw: Tensor,
    alpha_hw: Tensor,
    *,
    alpha_thresh: float,
    thresh_bin: float,
) -> Tensor:
    """Identify floater pixels from a single (H, W) depth + alpha.

    Steps:
        1. valid = alpha > alpha_thresh
        2. normalize depth over valid pixels
        3. cut-off = ``thresh_bin``-quantile of normalized valid depth
        4. floater_pixels = (normalized_depth < cut_off) & valid

    Returns:
        (H, W) bool tensor.
    """
    if depth_hw.shape != alpha_hw.shape:
        raise ValueError(
            f"depth/alpha shape mismatch: {tuple(depth_hw.shape)} vs {tuple(alpha_hw.shape)}"
        )

    valid = alpha_hw > float(alpha_thresh)
    if valid.sum() == 0:
        return torch.zeros_like(valid)

    # Compute min/max only on valid pixels so background does not bias things.
    d_valid = depth_hw[valid]
    d_min, d_max = d_valid.min(), d_valid.max()
    rng = (d_max - d_min).clamp_min(1e-8)
    d_norm = (depth_hw - d_min) / rng       # may go <0 or >1 outside valid; OK
    cut_off = torch.quantile(
        ((d_valid - d_min) / rng).float(),
        float(thresh_bin),
    )
    floater_pix = (d_norm < cut_off) & valid
    return floater_pix


def compute_floater_mask(
    *,
    depth: Tensor,                              # (H, W) or (H, W, 1)
    alpha: Tensor,                              # (H, W) or (H, W, 1)
    n_gaussians: int,
    gaussian_ids: Optional[Tensor] = None,      # (P,) long, packed=True path
    radii: Optional[Tensor] = None,             # (N,) numeric, packed=False path
    means2d: Optional[Tensor] = None,           # (N, 2) or (P, 2), pixel coords
    alpha_thresh: float = 0.5,
    thresh_bin: float = 0.13,
    packed: bool = True,
) -> Tensor:
    """Lightweight SparseGS-style floater detection on a *single* view.

    Two back-projection paths are supported:

    * ``packed=True``  : ``gaussian_ids`` is a flat (P,) tensor of GLOBAL
      indices for the Gaussians that contributed to this view. We mark a
      Gaussian as floater iff its ``means2d`` falls inside a floater pixel.
    * ``packed=False`` : ``radii > 0`` already gives us the (N,) per-Gaussian
      visibility for this view; ``means2d`` is (N, 2). Same back-projection,
      restricted to the visible subset.

    Returns:
        (N,) bool tensor; ``True`` marks Gaussians selected as floaters in
        this single view (the trainer should ``|`` across views).
    """
    # --- normalize tensor shapes ---
    if depth.dim() == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    if alpha.dim() == 3 and alpha.shape[-1] == 1:
        alpha = alpha.squeeze(-1)
    if depth.dim() != 2:
        raise ValueError(f"depth must be (H,W) or (H,W,1), got {tuple(depth.shape)}")

    H, W = depth.shape
    out = torch.zeros(int(n_gaussians), dtype=torch.bool, device=depth.device)

    # --- step 1: per-pixel floater mask ---
    floater_pix = _floater_pixel_mask_from_depth_alpha(
        depth, alpha, alpha_thresh=alpha_thresh, thresh_bin=thresh_bin
    )                                                                  # (H, W) bool
    if floater_pix.sum() == 0:
        return out

    # --- step 2: back-project floater pixels -> Gaussians via means2d ---
    if means2d is None:
        # No means2d => we can't back-project; conservatively prune nothing.
        return out

    if packed:
        if gaussian_ids is None:
            return out
        # means2d in packed mode is (P, 2) aligned with gaussian_ids.
        if means2d.dim() != 2 or means2d.shape[0] != gaussian_ids.shape[0]:
            return out
        candidate_global_ids = gaussian_ids
        cand_xy = means2d
    else:
        if radii is None:
            return out
        # gsplat 1.5.3 packed=False: radii is (B, N, 2) with per-axis screen
        # extents; a Gaussian is visible iff EITHER axis > 0.
        r = radii
        if r.dim() == 3:
            # (B, N, 2) -> (N, 2); take batch 0 (we render single-view).
            r = r[0]
        if r.dim() == 2 and r.shape[-1] == 2:
            vis = (r > 0).any(dim=-1)            # (N,)
        else:
            vis = (r.reshape(-1) > 0)
        if int(vis.sum()) == 0 or means2d is None:
            return out
        # gsplat un-packed means2d is (B, N, 2) or (N, 2). Strip batch.
        m2d = means2d
        if m2d.dim() == 3:
            m2d = m2d[0]
        if m2d.dim() != 2 or m2d.shape[-1] != 2:
            return out
        candidate_global_ids = torch.nonzero(vis, as_tuple=False).reshape(-1)
        cand_xy = m2d[vis]

    # Round to integer pixel and bounds-check.
    px = cand_xy[:, 0].round().long().clamp_(0, W - 1)
    py = cand_xy[:, 1].round().long().clamp_(0, H - 1)
    is_floater_at = floater_pix[py, px]                                # (P,) bool
    selected = candidate_global_ids[is_floater_at]
    if selected.numel() > 0:
        out[selected] = True
    return out


# ---------------------------------------------------------------------------
# 3) Safety cap
# ---------------------------------------------------------------------------
def apply_safety_cap(mask: Tensor, max_ratio: float) -> Tuple[Tensor, bool]:
    """If ``mask`` would prune more than ``max_ratio`` of all Gaussians, drop it.

    Args:
        mask: (N,) bool prune mask. ``True`` = prune.
        max_ratio: float in (0, 1). Hard cap on the fraction of Gaussians a
            *single* prune step may remove.

    Returns:
        (final_mask, over_aggressive). ``over_aggressive=True`` means the
        original mask exceeded ``max_ratio`` and ``final_mask`` is now an
        all-False mask (the caller must log + skip, **not** apply).
    """
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be bool, got {mask.dtype}")
    n = mask.numel()
    if n == 0:
        return mask, False
    ratio = float(mask.sum().item()) / float(n)
    if ratio > float(max_ratio):
        return torch.zeros_like(mask), True
    return mask, False
