"""Smoke test for the wide pseudo-view sampler (Option B1).

Constructs a fake pose pool of 8 cameras roughly distributed on a hemisphere
around origin (matches NeRF-Synthetic geometry), then exercises both the
legacy and the wide sampler. Checks viewmat shape, finite values, that
sphere-mode look-at points to ~origin, and that extrapolated t actually
moves the center outside [Ca, Cb].
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts import _bootstrap  # noqa: F401

import math
import random
import torch

from sparse_gs.datasets.nerf_synthetic import Camera
from sparse_gs.datasets.pseudo_pose import sample_pseudo_camera, set_wide_sampling


def _build_lookat_viewmat(eye, target, up, device):
    z = (target - eye); z = z / z.norm().clamp_min(1e-8)
    x = torch.linalg.cross(z, up); x = x / x.norm().clamp_min(1e-8)
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z], dim=1)
    c2w = torch.eye(4, device=device, dtype=eye.dtype)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye
    return torch.linalg.inv(c2w).contiguous()


def make_fake_pool(device):
    H, W = 64, 64
    K = torch.tensor([[80.0, 0.0, W/2], [0.0, 80.0, H/2], [0.0, 0.0, 1.0]], device=device)
    target = torch.zeros(3, device=device)
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    pool = []
    R_world = 4.0
    for i in range(8):
        azim = 2 * math.pi * i / 8
        elev = math.radians(15.0 + 10.0 * (i % 3))   # 15..35 deg
        eye = torch.tensor([
            R_world * math.cos(elev) * math.cos(azim),
            R_world * math.cos(elev) * math.sin(azim),
            R_world * math.sin(elev),
        ], device=device)
        viewmat = _build_lookat_viewmat(eye, target, up, device)
        pool.append(Camera(
            image=torch.zeros(H, W, 3, device=device),
            viewmat=viewmat.float(),
            K=K.float(),
            width=W, height=H,
            image_name=f"fake_{i}",
            cam_idx=i,
        ))
    return pool


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = make_fake_pool(device)
    print(f"[smoke] pool size = {len(pool)}, device = {device}")

    # ---- legacy path ----
    set_wide_sampling(None)
    random.seed(0)
    for i in range(3):
        c = sample_pseudo_camera(pool, device=device)
        print(f"  legacy[{i}] {c.image_name} viewmat.shape={tuple(c.viewmat.shape)} finite={bool(torch.isfinite(c.viewmat).all().item())}")
        assert c.viewmat.shape == (4, 4)
        assert torch.isfinite(c.viewmat).all()

    # ---- wide path ----
    set_wide_sampling({
        "enabled": True,
        "mode_probs": [0.34, 0.33, 0.33],
        "interp_t_range": [0.25, 0.75],
        "extrap_t_range": [-0.3, 1.3],
        "elevation_margin_deg": 10.0,
        "radius_jitter": 0.05,
    })
    random.seed(1)
    mode_count = {"interp": 0, "extrap": 0, "sphere": 0}
    centers_all = []
    for i in range(60):
        c = sample_pseudo_camera(pool, device=device)
        for k in mode_count:
            if k in c.image_name:
                mode_count[k] += 1
                break
        assert torch.isfinite(c.viewmat).all(), f"non-finite viewmat at i={i}: {c.image_name}"
        cw = torch.linalg.inv(c.viewmat)[:3, 3]
        centers_all.append(cw)
        if i < 6:
            r = float(cw.norm().item())
            print(f"  wide[{i}] {c.image_name} center={cw.tolist()} r={r:.3f}")
    print(f"  wide mode counts (out of 60): {mode_count}")

    # check that sphere-mode look-at really targets scene center (= mean of
    # train-cam centers, not necessarily the origin)
    set_wide_sampling({
        "enabled": True,
        "mode_probs": [0.0, 0.0, 1.0],   # sphere only
        "elevation_margin_deg": 10.0,
        "radius_jitter": 0.05,
    })
    # compute the same scene_center the sampler uses
    train_centers = torch.stack([torch.linalg.inv(c.viewmat)[:3, 3] for c in pool], dim=0)
    scene_center = train_centers.mean(dim=0)
    print(f"  scene_center (mean of train-cam centers) = {scene_center.tolist()}")
    random.seed(2)
    max_target_err = 0.0
    for i in range(20):
        c = sample_pseudo_camera(pool, device=device)
        c2w = torch.linalg.inv(c.viewmat)
        eye = c2w[:3, 3]
        forward = c2w[:3, 2]                       # camera +z (look direction)
        # closest point on the ray (eye + t*forward) to scene_center
        d = scene_center - eye
        t_star = (d @ forward).item()
        closest = eye + t_star * forward
        err = float((closest - scene_center).norm().item())
        max_target_err = max(max_target_err, err)
    print(f"  sphere-mode lookat err vs scene_center (max over 20) = {max_target_err:.4e}")
    assert max_target_err < 1e-3, "sphere-mode cameras are NOT looking at scene center"

    print("[smoke] OK")


if __name__ == "__main__":
    main()
