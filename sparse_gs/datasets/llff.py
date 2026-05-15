"""LLFF / forward-facing real-scene loader.

Reads the standard ``poses_bounds.npy`` produced by Mildenhall et al.'s
LLFF ``imgs2poses.py`` pipeline (and shipped with the canonical
``nerf_llff_data`` benchmark used by RegNeRF / FreeNeRF / SparseGS).

Conventions
-----------
LLFF stores per-frame poses as ``(3, 5)`` matrices where:

  * columns 0..2 are the camera *rotation* in **LLFF camera axes**
    ``[down, right, backwards]`` (i.e. ``-y, x, z`` of the OpenGL frame),
  * column 3 is the camera *translation* in world coordinates,
  * column 4 is ``[H, W, focal]``.

The ``poses_bounds.npy`` array has shape ``(N, 17)`` storing
``pose.flatten()`` (15) followed by ``[near, far]`` (2) per view.

We convert ``[down, right, back] -> [right, down, forward]`` (OpenCV)
in two steps to keep the math auditable:

    1) LLFF -> OpenGL:  re-order columns ``[-y, x, z] -> [x, y, z]``.
       Equivalent right-multiplication: ``R_gl = R_llff @ M_llff2gl``
       where  ``M_llff2gl = [[0,1,0],[-1,0,0],[0,0,1]]``.
    2) OpenGL -> OpenCV: flip the y and z camera axes,
       ``R_cv = R_gl @ diag(1, -1, -1)``.

Sparse-view protocol (RegNeRF / FreeNeRF / SparseGS standard)
-------------------------------------------------------------
* Sort views by file name (so the index axis is reproducible).
* Test split = every 8th view (``i % 8 == 0``).
* From the remaining 7/8, take ``n_train_views`` *uniformly* (default 3).
* No separate val split exists for LLFF in the literature; we expose
  ``val = test`` so the trainer's eval loop has something to render.

Scene normalisation
-------------------
We follow the recentering + scale convention shared by nerf-pytorch /
nerfstudio / 3DGS-vanilla LLFF readers:

    * recenter: rigidly transform world so that the *average* camera
      pose becomes identity (improves numerical conditioning of the
      optimiser without changing the rendered image at all);
    * rescale: divide all translations and the per-view ``bds`` by
      ``1.0 / (bds.min() * 0.75)`` so the closest scene depth is ~1.33.
      This brings the scene scale roughly in line with NeRF-Synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..utils.io import load_image_rgba
from .nerf_synthetic import Camera           # reuse the canonical Camera dataclass
from .sparse_sampler import sparse_view_indices


# OpenGL -> OpenCV: flip y and z of the camera axes.
_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


# ---------------------------------------------------------------------
# pose matrix utilities (kept in float64 for numerical stability)
# ---------------------------------------------------------------------
def _viewmatrix(z: np.ndarray, up: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Construct an OpenGL-style 3x4 c2w from a (z, up, position) triple."""
    z = z / (np.linalg.norm(z) + 1e-12)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z, pos], axis=1)


def _poses_avg(poses: np.ndarray) -> np.ndarray:
    """Average pose (3, 4) in OpenGL convention. ``poses`` is (N, 3, 4)."""
    center = poses[:, :3, 3].mean(axis=0)
    z = poses[:, :3, 2].sum(axis=0)
    up = poses[:, :3, 1].sum(axis=0)
    return _viewmatrix(z, up, center)


def _recenter_poses(poses: np.ndarray) -> np.ndarray:
    """Rigidly transform world so the mean pose is identity. ``poses`` (N,3,4)."""
    avg = _poses_avg(poses)                        # (3, 4)
    avg4 = np.concatenate([avg, np.array([[0, 0, 0, 1.0]])], axis=0)  # (4, 4)
    inv = np.linalg.inv(avg4)
    poses_h = np.concatenate(
        [poses, np.tile(np.array([[[0, 0, 0, 1.0]]]), (poses.shape[0], 1, 1))],
        axis=1,
    )                                              # (N, 4, 4)
    out = inv @ poses_h                            # broadcast on first dim
    return out[:, :3, :4]


# ---------------------------------------------------------------------
# image loading
# ---------------------------------------------------------------------
def _list_image_files(images_dir: Path) -> List[Path]:
    """Return image files sorted lexicographically (matches LLFF convention)."""
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    files = [p for p in images_dir.iterdir() if p.suffix in exts]
    if not files:
        raise FileNotFoundError(f"No images found under {images_dir}")
    return sorted(files, key=lambda p: p.name)


def _pick_images_dir(scene_root: Path, image_downsample: int) -> Tuple[Path, int]:
    """Pick LLFF's pre-downsampled folder if available, else use ``images/``.

    LLFF ships ``images_4`` / ``images_8`` next to ``images``. Using the
    pre-downsampled directory matches what every prior paper does and
    avoids wasting CPU on bilinear resizing in Python.

    Returns:
        (images_dir, effective_downsample)
    """
    if image_downsample <= 1:
        d = scene_root / "images"
        if not d.is_dir():
            raise FileNotFoundError(d)
        return d, 1

    # Try the LLFF pre-downsampled folder first.
    pre = scene_root / f"images_{image_downsample}"
    if pre.is_dir():
        return pre, image_downsample

    # Fall back to images/ + manual downsample.
    d = scene_root / "images"
    if not d.is_dir():
        raise FileNotFoundError(d)
    return d, image_downsample


def _maybe_downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean downsample by integer factor (only if folder wasn't already)."""
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    img = img[: h - h % factor, : w - w % factor]
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    if img.ndim == 2:
        img = img[..., None]
    img = img.reshape(h // factor, factor, w // factor, factor, c).mean(axis=(1, 3))
    if c == 1:
        img = img[..., 0]
    return img


# ---------------------------------------------------------------------
# main loader
# ---------------------------------------------------------------------
class LLFFDataset:
    """Forward-facing real-scene dataset (Mildenhall LLFF format).

    Args:
        root: directory containing the per-scene subdirectories
            (e.g. ``.../nerf_llff_data``).
        scene: scene name, e.g. ``"fern"``.
        n_train_views: how many training views to keep (default 3, the
            standard sparse-view protocol).
        train_view_ids: explicit override for the kept training indices
            (indices are into the *non-test* view list, after the every-8th
            test split has been removed).
        image_downsample: integer downsample factor. ``4`` and ``8`` use
            the pre-downsampled ``images_4`` / ``images_8`` folders if
            present; otherwise we block-mean downsample on the fly.
        white_background: ignored (real scenes have no alpha); accepted
            so the trainer config schema stays uniform.
        recenter: apply pose recentering (default True).
        rescale: rescale poses + bounds so the nearest depth is ~1.33
            (default True). This is the convention used by 3DGS official
            LLFF reader; turning it off keeps raw LLFF units.
        seed: RNG seed for ``sparse_mode='random'``.
        sparse_mode: ``"uniform"`` (default) or ``"random"``.
        depth_prior: same schema as ``NerfSyntheticDataset``.
    """

    def __init__(
        self,
        root: str | Path,
        scene: str,
        n_train_views: int = 3,
        train_view_ids: Optional[Sequence[int]] = None,
        image_downsample: int = 4,
        white_background: bool = False,             # noqa: ARG002 (kept for schema parity)
        recenter: bool = True,
        rescale: bool = True,
        seed: int = 42,
        sparse_mode: str = "uniform",
        depth_prior: Optional[dict] = None,
    ) -> None:
        scene_root = Path(root) / scene
        if not scene_root.is_dir():
            raise FileNotFoundError(
                f"LLFF scene directory not found: {scene_root}\n"
                f"Expected layout: <root>/<scene>/poses_bounds.npy + images/*.{{png,jpg}}"
            )
        self.scene_root = scene_root
        self.scene = scene

        # ---- 1) read poses_bounds.npy ----
        pb_path = scene_root / "poses_bounds.npy"
        if not pb_path.is_file():
            raise FileNotFoundError(pb_path)
        pb = np.load(pb_path).astype(np.float64)        # (N, 17)
        if pb.ndim != 2 or pb.shape[1] != 17:
            raise ValueError(f"unexpected poses_bounds shape: {pb.shape}")

        poses_llff = pb[:, :15].reshape(-1, 3, 5)       # (N, 3, 5)
        bds        = pb[:, 15:17]                       # (N, 2): near, far
        hwf        = poses_llff[0, :, 4].copy()         # [H, W, focal] (assumed shared)

        # ---- 2) LLFF axes ([down, right, back]) -> OpenGL ([right, up, back]) ----
        # Re-order columns: new = [pose[:,1], -pose[:,0], pose[:,2], pose[:,3]]
        poses = np.concatenate(
            [
                poses_llff[:, :, 1:2],            # right    (was col 1)
                -poses_llff[:, :, 0:1],           # up = -down
                poses_llff[:, :, 2:3],            # back
                poses_llff[:, :, 3:4],            # translation
            ],
            axis=2,
        )                                          # (N, 3, 4) in OpenGL convention

        # ---- 3) (optional) recenter so mean pose is identity ----
        if recenter:
            poses = _recenter_poses(poses)

        # ---- 4) (optional) rescale so nearest depth ~1.33 ----
        if rescale:
            sc = 1.0 / (bds.min() * 0.75)
            poses[:, :3, 3] *= sc
            bds *= sc
        self.bounds_near_far = (float(bds[:, 0].min()), float(bds[:, 1].max()))

        # ---- 5) match poses to image files ----
        images_dir, ds_eff = _pick_images_dir(scene_root, image_downsample)
        # Did we already downsample by selecting images_X? If yes, no on-the-fly resize.
        on_the_fly_ds = ds_eff if (images_dir.name == "images" and ds_eff > 1) else 1
        files = _list_image_files(images_dir)
        if len(files) != poses.shape[0]:
            raise RuntimeError(
                f"#images ({len(files)}) != #poses ({poses.shape[0]}) for scene {scene}. "
                f"Did poses_bounds.npy come from this exact image set?"
            )

        H_full, W_full, focal_full = float(hwf[0]), float(hwf[1]), float(hwf[2])

        # ---- 6) build Camera objects ----
        all_cams: List[Camera] = []
        for i, (P_gl, fpath) in enumerate(zip(poses, files)):
            img = load_image_rgba(fpath)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]                  # LLFF has no meaningful alpha
            img = _maybe_downsample(img, on_the_fly_ds)
            H, W = img.shape[:2]

            # Intrinsics: scale focal length proportionally to actual image size.
            #   ``hwf`` is for the *raw* (un-downsampled) image.
            #   If we used images_4, then H ~= H_full/4 -> focal scales by H/H_full.
            scale = H / H_full
            fx = fy = focal_full * scale
            cx, cy = 0.5 * W, 0.5 * H
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            # Pose: OpenGL c2w (3x4) -> OpenCV c2w (4x4) -> w2c viewmat.
            P_gl_h = np.concatenate([P_gl, np.array([[0, 0, 0, 1.0]])], axis=0)  # (4,4)
            c2w_cv = P_gl_h @ _GL2CV
            viewmat = np.linalg.inv(c2w_cv).astype(np.float32)

            all_cams.append(
                Camera(
                    image=torch.from_numpy(np.ascontiguousarray(img)).float(),
                    viewmat=torch.from_numpy(viewmat).float(),
                    K=torch.from_numpy(K).float(),
                    width=W,
                    height=H,
                    image_name=fpath.name,
                    cam_idx=i,
                    alpha=None,
                )
            )

        # ---- 7) RegNeRF/FreeNeRF/SparseGS sparse-view split ----
        # Test = every 8th view; train candidates = the rest.
        test_ids  = [i for i in range(len(all_cams)) if i % 8 == 0]
        train_pool = [i for i in range(len(all_cams)) if i % 8 != 0]

        # Pick n_train_views uniformly from the train pool.
        kept_local = sparse_view_indices(
            n_total=len(train_pool),
            n_train=n_train_views,
            explicit=train_view_ids,
            seed=seed,
            mode=sparse_mode,
        )
        kept_global = [train_pool[i] for i in kept_local]

        self.train: List[Camera] = [all_cams[i] for i in kept_global]
        self.test:  List[Camera] = [all_cams[i] for i in test_ids]
        # No separate val split in LLFF -> reuse test for the trainer's eval loop.
        self.val:   List[Camera] = list(self.test)

        self.train_view_ids: List[int] = list(kept_global)
        self.test_view_ids:  List[int] = list(test_ids)
        self.n_total_train = len(train_pool)
        self.n_total = len(all_cams)

        # ---- 8) optional monocular depth prior ----
        self.depth_prior_cfg = dict(depth_prior) if depth_prior else None
        if self.depth_prior_cfg is not None and self.depth_prior_cfg.get("enabled", False):
            from ..utils.depth_prior import attach_mono_depth_to_cameras
            attach_mono_depth_to_cameras(
                self.train,
                scene_root=scene_root,
                split="train",
                cfg=self.depth_prior_cfg,
            )

        # ---- 9) heuristic scene scale (same definition as NeRF-Synthetic) ----
        if len(self.train) >= 2:
            centers = np.stack(
                [np.linalg.inv(c.viewmat.numpy())[:3, 3] for c in self.train], axis=0
            )
            self.scene_scale = float(
                np.linalg.norm(centers - centers.mean(0), axis=1).max()
            )
            self.scene_scale = max(self.scene_scale, 1.0)
        else:
            self.scene_scale = 1.0

    def __repr__(self) -> str:
        return (
            f"LLFFDataset(scene={self.scene}, "
            f"train={len(self.train)}/{self.n_total_train} (ids={self.train_view_ids}), "
            f"test={len(self.test)} (ids={self.test_view_ids}), "
            f"bounds={self.bounds_near_far}, "
            f"scene_scale={self.scene_scale:.3f})"
        )
