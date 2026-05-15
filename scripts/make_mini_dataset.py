"""Generate a tiny synthetic ``nerf_synthetic``-format scene for plumbing tests.

Creates ``data/nerf_synthetic/_mini/`` with:

* 8 train + 4 val + 4 test cameras around a coloured cube
* 64x64 PNGs, white background
* matching ``transforms_*.json``

This is **only** for verifying the training loop end-to-end on machines
without the real NeRF-Synthetic dataset. PSNR will plateau low — that's
expected because the cube is purely procedural.

Run from ``d:/SSL/sparse_gs``::

    python scripts/make_mini_dataset.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))


def _look_at_opengl(eye, target, up):
    """Return a 4x4 OpenGL camera-to-world matrix (Blender convention)."""
    f = (target - eye); f = f / np.linalg.norm(f)
    up = up / np.linalg.norm(up)
    s = np.cross(f, up); s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    # In OpenGL: camera looks down -Z, +Y up, +X right.
    R = np.stack([s, u, -f], axis=1)              # cols = camera basis in world
    c2w = np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = eye
    return c2w


def _project_cube(c2w_gl, fov_x, W, H):
    """Render a coloured cube via 3D->2D projection (no z-buffer; just front face)."""
    # OpenGL c2w -> world->camera (OpenCV-ish for our drawing). We'll project
    # vertices in OpenGL (right-down inverted), then use them directly.
    img = np.full((H, W, 3), 1.0, dtype=np.float32)              # white bg
    fx = 0.5 * W / math.tan(0.5 * fov_x)
    cx, cy = 0.5 * W, 0.5 * H

    # Cube: 8 vertices at +-0.5 with 6 face colours (RGBYMC)
    s = 0.5
    verts = np.array([[x, y, z] for z in (-s, s) for y in (-s, s) for x in (-s, s)], dtype=np.float32)
    faces = [
        # (vert idx, colour). Index uses (z,y,x) order above:
        # 0: (-s,-s,-s) 1:( s,-s,-s) 2:(-s, s,-s) 3:( s, s,-s)
        # 4:(-s,-s, s) 5:( s,-s, s) 6:(-s, s, s) 7:( s, s, s)
        ([0, 1, 3, 2], (1.0, 0.2, 0.2)),  # -Z red
        ([4, 5, 7, 6], (0.2, 0.2, 1.0)),  # +Z blue
        ([0, 1, 5, 4], (0.2, 1.0, 0.2)),  # -Y green
        ([2, 3, 7, 6], (1.0, 1.0, 0.2)),  # +Y yellow
        ([0, 2, 6, 4], (1.0, 0.2, 1.0)),  # -X magenta
        ([1, 3, 7, 5], (0.2, 1.0, 1.0)),  # +X cyan
    ]

    # World -> camera (OpenGL): w2c = inv(c2w_gl). Camera looks down -Z.
    w2c = np.linalg.inv(c2w_gl)

    def project(p_world):
        p_h = np.append(p_world, 1.0)
        p_cam = w2c @ p_h
        z = -p_cam[2]                 # OpenGL: forward = -Z
        if z <= 1e-6:
            return None
        u = fx * p_cam[0] / z + cx
        v = -fx * p_cam[1] / z + cy   # flip y for image coords
        return np.array([u, v]), z

    # Painter's algorithm: sort faces by mean depth.
    face_records = []
    for idxs, col in faces:
        proj = [project(verts[i]) for i in idxs]
        if any(p is None for p in proj):
            continue
        pts = np.stack([p[0] for p in proj], axis=0)
        depth = float(np.mean([p[1] for p in proj]))
        face_records.append((depth, pts, col))
    face_records.sort(key=lambda r: -r[0])  # far first

    # Rasterize each face by polygon fill (using PIL).
    pil = Image.fromarray((img * 255).astype(np.uint8))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(pil)
    for _, pts, col in face_records:
        poly = [(float(p[0]), float(p[1])) for p in pts]
        draw.polygon(poly, fill=tuple(int(c * 255) for c in col),
                     outline=(0, 0, 0))
    return np.asarray(pil).astype(np.float32) / 255.0


def _ring_cameras(n: int, radius: float = 3.0, height: float = 1.0):
    poses = []
    for k in range(n):
        ang = 2 * math.pi * k / n
        eye = np.array([radius * math.cos(ang), height, radius * math.sin(ang)],
                       dtype=np.float32)
        target = np.zeros(3, dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)
        poses.append(_look_at_opengl(eye, target, up))
    return poses


def main():
    out = _ROOT / "data" / "nerf_synthetic" / "_mini"
    out.mkdir(parents=True, exist_ok=True)
    H = W = 64
    fov_x = math.radians(50)

    splits = {"train": 8, "val": 4, "test": 4}
    for split, n in splits.items():
        (out / split).mkdir(exist_ok=True)
        poses = _ring_cameras(n, radius=3.0, height=1.0 if split != "val" else 0.5)
        frames = []
        for i, c2w in enumerate(poses):
            img = _project_cube(c2w, fov_x, W, H)
            # Save as RGBA so the loader treats white as 'background'.
            rgba = np.concatenate([img, np.ones((H, W, 1), dtype=np.float32)], axis=-1)
            arr = (rgba * 255).clip(0, 255).astype(np.uint8)
            fname = f"r_{i}.png"
            Image.fromarray(arr, mode="RGBA").save(out / split / fname)
            frames.append({
                "file_path": f"./{split}/r_{i}",
                "transform_matrix": c2w.tolist(),
                "rotation": 0.0,
            })
        with open(out / f"transforms_{split}.json", "w", encoding="utf-8") as f:
            json.dump({"camera_angle_x": fov_x, "frames": frames}, f, indent=2)

    print(f"[mini-data] wrote synthetic dataset to {out}")
    print("[mini-data] quick test:")
    print("    python scripts/train.py --config configs/sparse_view_mini.yaml")


if __name__ == "__main__":
    main()
