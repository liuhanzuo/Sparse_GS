"""Per-image monocular depth prior for SSL.

Two backends are supported, selected by ``cfg.data.depth_prior.backend``:

  ``alpha``
      Derive a pseudo-disparity directly from the RGBA alpha channel of
      NeRF-Synthetic-style images: ``disparity = alpha`` (in [0, 1], with
      a small constant added to the background so the SSI-loss has a
      well-defined background plane). This is **free** — no external
      model, no download — and is surprisingly effective on synthetic
      foreground / background scenes because it provides a strong
      silhouette cue. It does **not** disambiguate depth *within* the
      foreground, so it should be combined with multi-view consistency
      (which is what the existing ``multiview_photo`` loss does).

  ``cache``
      Read pre-computed depth maps from ``<scene>/_depth_cache/<tag>/<image_name>.npz``.
      The npz must contain a single array ``depth`` of shape (H, W) or
      (H, W, 1). See ``scripts/precompute_depth.py`` for a generator.
      Use this for real-data scenes (DTU, Mip-NeRF-360, ...) where there
      is no alpha channel and you want a real prior (e.g. DepthAnything-V2).

Either backend produces a tensor stored on ``Camera.mono_depth`` with
``Camera.depth_kind`` set to ``"disparity"`` or ``"metric"``. The loss
code does scale-shift-invariant matching, so the absolute scale does
not matter — only the *relative ordering* of depths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------
# backend: alpha-derived pseudo-disparity (NeRF-Synthetic only)
# ---------------------------------------------------------------------
def _alpha_to_disparity(alpha: torch.Tensor, bg_value: float = 0.0) -> torch.Tensor:
    """Map an alpha mask in [0,1] to a per-pixel pseudo-disparity in [bg, 1].

    Foreground (alpha=1) -> disparity 1 (closest)
    Background (alpha=0) -> disparity bg_value (farthest, default 0)

    Returns (H, W, 1) float32.
    """
    if alpha.dim() == 2:
        alpha = alpha.unsqueeze(-1)
    a = alpha.clamp(0.0, 1.0).float()
    return a * (1.0 - bg_value) + bg_value


# ---------------------------------------------------------------------
# backend: load pre-computed depth from .npz cache
# ---------------------------------------------------------------------
def _load_cached_depth(npz_path: Path) -> Optional[torch.Tensor]:
    if not npz_path.is_file():
        return None
    try:
        with np.load(str(npz_path)) as f:
            if "depth" in f:
                d = f["depth"]
            elif "disparity" in f:
                d = f["disparity"]
            else:
                # take the first array
                key = list(f.keys())[0]
                d = f[key]
        d = np.asarray(d, dtype=np.float32)
        if d.ndim == 2:
            d = d[..., None]
        elif d.ndim == 3 and d.shape[-1] != 1:
            # if (1,H,W) -> (H,W,1)
            if d.shape[0] == 1:
                d = np.transpose(d, (1, 2, 0))
            else:
                d = d[..., :1]
        return torch.from_numpy(np.ascontiguousarray(d)).float()
    except Exception as e:  # noqa: BLE001
        print(f"[depth_prior] failed to load {npz_path}: {e}")
        return None


def cache_path_for(
    scene_root: Path, tag: str, image_name: str, split: str = "train"
) -> Path:
    """Where the .npz lives. Mirrors the layout used by precompute_depth.py.

    ``image_name`` is e.g. ``r_0`` (no extension) per NeRF-Synthetic
    convention; we strip a trailing ``.png`` if present for safety.
    """
    stem = Path(image_name).stem
    return scene_root / "_depth_cache" / tag / split / f"{stem}.npz"


# ---------------------------------------------------------------------
# main entry point used by NerfSyntheticDataset
# ---------------------------------------------------------------------
def attach_mono_depth_to_cameras(
    cameras: List[Any],            # List[Camera]; avoid circular import
    scene_root: Path,
    split: str,
    cfg: Dict[str, Any],
) -> None:
    """Mutate ``cameras`` in place, setting ``mono_depth`` and ``depth_kind``.

    Config schema (under ``data.depth_prior``):

        enabled:    bool       — global switch
        backend:    str        — "alpha" | "cache"
        tag:        str        — sub-folder under _depth_cache/ for "cache"
        kind:       str        — "disparity" | "metric"; default depends on
                                 backend ("disparity" for alpha, "metric"
                                 for cache unless overridden)
        bg_value:   float      — for "alpha", the background disparity
                                 (default 0.0). Setting to e.g. 0.05
                                 prevents log(0) downstream.
        required:   bool       — if True, raise when no prior is found
                                 for some camera; if False (default), just
                                 leave mono_depth=None and warn once.
    """
    backend = str(cfg.get("backend", "alpha"))
    if backend == "alpha":
        kind = str(cfg.get("kind", "disparity"))
        bg_v = float(cfg.get("bg_value", 0.0))
        n_ok, n_skip = 0, 0
        for cam in cameras:
            if cam.alpha is None:
                n_skip += 1
                continue
            cam.mono_depth = _alpha_to_disparity(cam.alpha, bg_value=bg_v)
            cam.depth_kind = kind
            n_ok += 1
        print(f"[depth_prior] backend=alpha split={split}: "
              f"{n_ok} attached, {n_skip} skipped (no alpha channel)")
        return

    if backend == "cache":
        tag = str(cfg.get("tag", "depth_anything_v2_small"))
        kind = str(cfg.get("kind", "disparity"))
        required = bool(cfg.get("required", False))
        n_ok, n_miss = 0, 0
        first_miss: Optional[Path] = None
        for cam in cameras:
            p = cache_path_for(scene_root, tag, cam.image_name, split=split)
            d = _load_cached_depth(p)
            if d is None:
                n_miss += 1
                if first_miss is None:
                    first_miss = p
                continue
            # sanity-check shape vs image
            if d.shape[0] != cam.height or d.shape[1] != cam.width:
                # bilinearly resize -> (H, W, 1)
                d_chw = d.permute(2, 0, 1).unsqueeze(0)
                d_chw = torch.nn.functional.interpolate(
                    d_chw, size=(cam.height, cam.width),
                    mode="bilinear", align_corners=False,
                )
                d = d_chw.squeeze(0).permute(1, 2, 0).contiguous()
            cam.mono_depth = d
            cam.depth_kind = kind
            n_ok += 1
        msg = (f"[depth_prior] backend=cache tag={tag} split={split}: "
               f"{n_ok} attached, {n_miss} missing")
        if first_miss is not None:
            msg += f" (first miss: {first_miss})"
        print(msg)
        if required and n_miss > 0:
            raise FileNotFoundError(
                f"depth_prior.required=true but {n_miss} cache files missing; "
                f"run scripts/precompute_depth.py first."
            )
        return

    raise ValueError(f"unknown depth_prior.backend: {backend!r}")
