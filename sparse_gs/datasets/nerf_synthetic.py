"""NeRF-Synthetic loader (Blender-style ``transforms_*.json``).

Conventions
-----------
- gsplat expects **OpenCV** camera coordinates (X right, Y down, Z forward)
  and a *world-to-camera* viewmat.
- NeRF-Synthetic stores **OpenGL / Blender** camera-to-world poses
  (X right, Y up, Z back).
- We convert OpenGL c2w -> OpenCV c2w by flipping Y and Z of the camera
  axes (right-multiply the rotation by ``diag(1, -1, -1)``), then invert
  to obtain the world-to-camera viewmat.
- Backgrounds: NeRF-Synthetic PNGs are RGBA on a transparent background.
  When ``white_background=True`` we composite onto white at load time
  (the standard convention used in the original NeRF / 3DGS papers).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from ..utils.io import load_image_rgba
from .sparse_sampler import sparse_view_indices


# OpenGL (Blender) -> OpenCV: flip Y and Z of camera axes.
_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


@dataclass
class Camera:
    """A single image + camera. All tensors live on CPU until moved.

    Optional fields:
        alpha:       (H, W, 1) float32 in [0, 1] — original RGBA alpha if
                     present in the source image (NeRF-Synthetic). Useful as
                     a silhouette / foreground mask.
        mono_depth:  (H, W, 1) float32 — disparity-like or metric depth
                     prior loaded from cache (see ``utils.depth_prior``).
                     Higher = closer for disparity, larger = farther for
                     metric. Loss code applies a scale-shift-invariant
                     match so either convention works.
        depth_kind:  one of {"disparity", "metric", None}. Optional hint
                     so loss code knows whether to flip the sign before
                     SSI matching against the rendered (metric) depth.
    """

    image: torch.Tensor          # (H, W, 3), float32 in [0,1]
    viewmat: torch.Tensor        # (4, 4), world-to-camera, OpenCV
    K: torch.Tensor              # (3, 3) intrinsics
    width: int
    height: int
    image_name: str
    cam_idx: int                 # original index in the JSON
    alpha: Optional[torch.Tensor] = None         # (H, W, 1) float32 in [0,1]
    mono_depth: Optional[torch.Tensor] = None    # (H, W, 1) float32
    depth_kind: Optional[str] = None             # "disparity" | "metric" | None

    def to(self, device) -> "Camera":
        return Camera(
            image=self.image.to(device, non_blocking=True),
            viewmat=self.viewmat.to(device, non_blocking=True),
            K=self.K.to(device, non_blocking=True),
            width=self.width,
            height=self.height,
            image_name=self.image_name,
            cam_idx=self.cam_idx,
            alpha=self.alpha.to(device, non_blocking=True) if self.alpha is not None else None,
            mono_depth=(
                self.mono_depth.to(device, non_blocking=True)
                if self.mono_depth is not None else None
            ),
            depth_kind=self.depth_kind,
        )


def _focal_from_fovx(width: int, fov_x: float) -> float:
    return 0.5 * width / math.tan(0.5 * fov_x)


def _load_split(
    scene_root: Path,
    split: str,
    image_downsample: int,
    white_background: bool,
) -> List[Camera]:
    json_path = scene_root / f"transforms_{split}.json"
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fov_x = float(meta["camera_angle_x"])
    frames = meta["frames"]

    cameras: List[Camera] = []
    for i, fr in enumerate(frames):
        # File path: NeRF-Synthetic uses paths like "./train/r_0" (no extension).
        rel = fr["file_path"]
        candidates = [
            scene_root / (rel + ".png"),
            scene_root / rel,
            scene_root / Path(rel).with_suffix(".png"),
        ]
        img_path = next((p for p in candidates if p.is_file()), None)
        if img_path is None:
            raise FileNotFoundError(f"image not found for frame: {rel}")

        img = load_image_rgba(img_path)            # HxWx{3,4} float32
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        alpha_chan: Optional[np.ndarray] = None
        if img.shape[-1] == 4:
            rgb, alpha_chan = img[..., :3], img[..., 3:4]
            if white_background:
                img = rgb * alpha_chan + (1.0 - alpha_chan)  # composite on white
            else:
                img = rgb
        # Downsample if requested.
        if image_downsample > 1:
            ds = image_downsample
            h, w = img.shape[:2]
            img = img[: h - h % ds, : w - w % ds]
            img = img.reshape(h // ds, ds, w // ds, ds, 3).mean(axis=(1, 3))
            if alpha_chan is not None:
                ah, aw = alpha_chan.shape[:2]
                alpha_chan = alpha_chan[: ah - ah % ds, : aw - aw % ds]
                alpha_chan = alpha_chan.reshape(
                    ah // ds, ds, aw // ds, ds, 1
                ).mean(axis=(1, 3))
        H, W = img.shape[:2]

        # Pose: 4x4 c2w in OpenGL.
        c2w_gl = np.array(fr["transform_matrix"], dtype=np.float32)
        c2w_cv = c2w_gl @ _GL2CV                    # OpenGL -> OpenCV
        viewmat = np.linalg.inv(c2w_cv).astype(np.float32)

        fx = fy = _focal_from_fovx(W, fov_x)
        cx, cy = 0.5 * W, 0.5 * H
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        cameras.append(
            Camera(
                image=torch.from_numpy(np.ascontiguousarray(img)).float(),
                viewmat=torch.from_numpy(viewmat).float(),
                K=torch.from_numpy(K).float(),
                width=W,
                height=H,
                image_name=Path(rel).name,
                cam_idx=i,
                alpha=(
                    torch.from_numpy(np.ascontiguousarray(alpha_chan)).float()
                    if alpha_chan is not None else None
                ),
            )
        )

    return cameras


class NerfSyntheticDataset:
    """Holds all three splits in CPU memory.

    Sparse-view subsampling is applied to the *train* split only.
    """

    def __init__(
        self,
        root: str | Path,
        scene: str,
        n_train_views: int = 6,
        train_view_ids: Optional[Sequence[int]] = None,
        image_downsample: int = 1,
        white_background: bool = True,
        seed: int = 42,
        sparse_mode: str = "uniform",
        depth_prior: Optional[dict] = None,
    ) -> None:
        scene_root = Path(root) / scene
        if not scene_root.is_dir():
            raise FileNotFoundError(
                f"Scene directory not found: {scene_root}\n"
                f"Expected layout: <root>/<scene>/transforms_train.json + train/*.png"
            )
        self.scene_root = scene_root
        self.scene = scene

        all_train = _load_split(scene_root, "train", image_downsample, white_background)
        self.val   = _load_split(scene_root, "val",   image_downsample, white_background)
        self.test  = _load_split(scene_root, "test",  image_downsample, white_background)

        kept = sparse_view_indices(
            n_total=len(all_train),
            n_train=n_train_views,
            explicit=train_view_ids,
            seed=seed,
            mode=sparse_mode,
        )
        self.train: List[Camera] = [all_train[i] for i in kept]
        self.train_view_ids: List[int] = list(kept)
        self.n_total_train = len(all_train)

        # Optionally attach a per-view monocular depth prior to the train
        # cameras only (val/test are evaluated, not regularised). Missing
        # cache files are allowed -> Camera.mono_depth stays None and the
        # depth_consist loss simply skips that view.
        self.depth_prior_cfg = dict(depth_prior) if depth_prior else None
        if self.depth_prior_cfg is not None and self.depth_prior_cfg.get("enabled", False):
            from ..utils.depth_prior import attach_mono_depth_to_cameras
            attach_mono_depth_to_cameras(
                self.train,
                scene_root=scene_root,
                split="train",
                cfg=self.depth_prior_cfg,
            )

        # Heuristic scene scale: max distance between train cam centers.
        if len(self.train) >= 2:
            centers = np.stack(
                [np.linalg.inv(c.viewmat.numpy())[:3, 3] for c in self.train], axis=0
            )
            self.scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max())
            self.scene_scale = max(self.scene_scale, 1.0)
        else:
            self.scene_scale = 1.0

    def __repr__(self) -> str:
        return (
            f"NerfSyntheticDataset(scene={self.scene}, "
            f"train={len(self.train)}/{self.n_total_train} (ids={self.train_view_ids}), "
            f"val={len(self.val)}, test={len(self.test)}, "
            f"scene_scale={self.scene_scale:.3f})"
        )
