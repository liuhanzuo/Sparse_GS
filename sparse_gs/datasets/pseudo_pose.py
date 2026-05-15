"""Pseudo-view pose sampling.

Given the sparse set of training cameras, we synthesize *unseen* poses by
interpolating between them. Because all training views face the scene
center, linear interpolation between two of them generally stays inside
the valid viewing hemisphere — which is what we want for an SSL target.

Camera convention: **world-to-camera**, OpenCV.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..datasets.nerf_synthetic import Camera


# --------------------------------------------------------------------------
# Wide-sampling global config (Option B1 — wider pseudo-view distribution)
# --------------------------------------------------------------------------
# We keep the public API ``sample_pseudo_camera(...)`` unchanged so that no
# call-site needs to know about this. Trainers call ``set_wide_sampling(cfg)``
# once at init and the sampler picks the right strategy internally.
_WIDE_CFG: Optional[Dict[str, Any]] = None
# Cached geometry stats of the current pose pool (so we don't recompute
# every step). Keyed by ``id(pose_pool)``.
_POOL_STATS_CACHE: Dict[int, Dict[str, Any]] = {}


def set_wide_sampling(cfg: Optional[Dict[str, Any]]) -> None:
    """Enable / disable wide pseudo-view sampling globally.

    cfg keys (all optional, with defaults):
      enabled:              bool   (default False; if False this is a no-op)
      mode_probs:           [interp, extrap, sphere_uniform]  (default [0.4, 0.3, 0.3])
      interp_t_range:       [lo, hi]   (default [0.25, 0.75])
      extrap_t_range:       [lo, hi]   (default [-0.3, 1.3])
      elevation_margin_deg: float  (default 10.0)  — for sphere_uniform
      radius_jitter:        float  (default 0.05)  — ±fraction
    """
    global _WIDE_CFG, _POOL_STATS_CACHE
    if cfg is None or not bool(cfg.get("enabled", False)):
        _WIDE_CFG = None
    else:
        _WIDE_CFG = dict(cfg)
    # any cfg change invalidates the cached stats so re-derive next call
    _POOL_STATS_CACHE = {}


# --------------------------------------------------------------------------
# small rotation utils
# --------------------------------------------------------------------------
def _rot_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(3,3) rotation -> (4,) wxyz quaternion."""
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        r = torch.sqrt(1.0 + t)
        w = 0.5 * r
        x = (m[2, 1] - m[1, 2]) / (2.0 * r)
        y = (m[0, 2] - m[2, 0]) / (2.0 * r)
        z = (m[1, 0] - m[0, 1]) / (2.0 * r)
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        r = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / (2.0 * r)
        x = 0.5 * r
        y = (m[0, 1] + m[1, 0]) / (2.0 * r)
        z = (m[0, 2] + m[2, 0]) / (2.0 * r)
    elif m[1, 1] > m[2, 2]:
        r = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / (2.0 * r)
        x = (m[0, 1] + m[1, 0]) / (2.0 * r)
        y = 0.5 * r
        z = (m[1, 2] + m[2, 1]) / (2.0 * r)
    else:
        r = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / (2.0 * r)
        x = (m[0, 2] + m[2, 0]) / (2.0 * r)
        y = (m[1, 2] + m[2, 1]) / (2.0 * r)
        z = 0.5 * r
    return torch.stack([w, x, y, z])


def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """(4,) wxyz -> (3,3)."""
    q = q / q.norm().clamp_min(1e-12)
    w, x, y, z = q.unbind()
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)]),
        torch.stack([    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)]),
        torch.stack([    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
    ])
    return R


def _slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    q0 = q0 / q0.norm().clamp_min(1e-12)
    q1 = q1 / q1.norm().clamp_min(1e-12)
    dot = (q0 * q1).sum()
    # take shorter path
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / out.norm().clamp_min(1e-12)
    theta0 = torch.acos(dot.clamp(-1.0, 1.0))
    theta = theta0 * t
    s0 = torch.sin(theta0 - theta) / torch.sin(theta0)
    s1 = torch.sin(theta) / torch.sin(theta0)
    return s0 * q0 + s1 * q1


# --------------------------------------------------------------------------
# look-at helper: build c2w (OpenCV) so the camera at ``eye`` looks at
# ``target`` with given world up. Returns 4x4 c2w.
# --------------------------------------------------------------------------
def _lookat_c2w(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """OpenCV convention c2w: x=right, y=down, z=forward (eye->target)."""
    z = target - eye
    z = z / z.norm().clamp_min(1e-8)            # forward (camera +z)
    x = torch.linalg.cross(z, up)
    x = x / x.norm().clamp_min(1e-8)            # right (camera +x)
    y = torch.linalg.cross(z, x)                # down (camera +y)
    R = torch.stack([x, y, z], dim=1)           # columns = camera basis in world
    c2w = torch.eye(4, device=eye.device, dtype=eye.dtype)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye
    return c2w


# --------------------------------------------------------------------------
# pose-pool geometry stats (cached): center, mean radius, elevation range
# --------------------------------------------------------------------------
def _pool_stats(pose_pool: List[Camera], device: torch.device) -> Dict[str, Any]:
    key = id(pose_pool)
    cached = _POOL_STATS_CACHE.get(key)
    if cached is not None:
        return cached
    centers = []
    for cam in pose_pool:
        v = cam.viewmat.to(device).float()
        c = torch.linalg.inv(v)[:3, 3]
        centers.append(c)
    C = torch.stack(centers, dim=0)             # (N, 3)
    scene_center = C.mean(dim=0)                # (3,)
    rel = C - scene_center
    radii = rel.norm(dim=1)                     # (N,)
    mean_radius = float(radii.mean().item())
    # elevation = asin(z / r) treating world +z as up (NeRF-Synthetic standard)
    z_over_r = (rel[:, 2] / radii.clamp_min(1e-8)).clamp(-1.0, 1.0)
    elevs = torch.asin(z_over_r)                # rad, (N,)
    elev_min = float(elevs.min().item())
    elev_max = float(elevs.max().item())
    stats = {
        "scene_center": scene_center,           # (3,) tensor on device
        "mean_radius": mean_radius,
        "elev_min": elev_min,
        "elev_max": elev_max,
    }
    _POOL_STATS_CACHE[key] = stats
    return stats


def _sample_extrapolate(
    pose_pool: List[Camera],
    device: torch.device,
    t_range: Tuple[float, float],
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SLERP/LERP with t outside [0,1] to step beyond the train-cam pair."""
    a, b = rng.sample(range(len(pose_pool)), 2)
    t = float(rng.uniform(*t_range))
    cam_a, cam_b = pose_pool[a], pose_pool[b]
    va = cam_a.viewmat.to(device); vb = cam_b.viewmat.to(device)
    Ca = torch.linalg.inv(va); Cb = torch.linalg.inv(vb)
    qa = _rot_to_quat(Ca[:3, :3])
    qb = _rot_to_quat(Cb[:3, :3])
    # Note: _slerp clamps to shortest arc for t in [0,1]; for extrapolation
    # we explicitly compute via the same formula (sin(theta0-theta) etc.)
    q = _slerp(qa, qb, t)
    R = _quat_to_rot(q)
    center = (1.0 - t) * Ca[:3, 3] + t * Cb[:3, 3]
    return R, center


def _sample_sphere_uniform(
    pose_pool: List[Camera],
    device: torch.device,
    elevation_margin_deg: float,
    radius_jitter: float,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Uniform on a spherical shell around scene_center, look-at scene_center.

    Elevation is restricted to the train-cam elevation range expanded by
    ±elevation_margin_deg so we never sample views completely outside the
    region the model has any signal for.
    """
    stats = _pool_stats(pose_pool, device)
    scene_c = stats["scene_center"]
    margin = math.radians(elevation_margin_deg)
    elev_lo = max(stats["elev_min"] - margin, -math.pi / 2 + 1e-3)
    elev_hi = min(stats["elev_max"] + margin,  math.pi / 2 - 1e-3)
    # uniform over (sin(elev_lo), sin(elev_hi)) so it's area-uniform on the
    # spherical cap rather than angle-uniform (avoids pole bias)
    u = rng.uniform(math.sin(elev_lo), math.sin(elev_hi))
    elev = math.asin(u)
    azim = rng.uniform(-math.pi, math.pi)
    r_base = stats["mean_radius"]
    r = r_base * (1.0 + rng.uniform(-radius_jitter, radius_jitter))
    # eye position in world coords (z is up)
    eye = scene_c + torch.tensor([
        r * math.cos(elev) * math.cos(azim),
        r * math.cos(elev) * math.sin(azim),
        r * math.sin(elev),
    ], device=device, dtype=scene_c.dtype)
    up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=scene_c.dtype)
    c2w = _lookat_c2w(eye, scene_c, up)
    return c2w[:3, :3], c2w[:3, 3]


# --------------------------------------------------------------------------
# public: sample a pseudo camera
# --------------------------------------------------------------------------
def sample_pseudo_camera(
    pose_pool: List[Camera],
    device: torch.device,
    t_range: tuple = (0.25, 0.75),
    rng: Optional[random.Random] = None,
) -> Camera:
    """Synthesize an unseen pose.

    Default behavior (no wide cfg set): SLERP rotation + LERP center between
    two random training cameras with t in ``t_range`` — same as before.

    When ``set_wide_sampling({...enabled: true})`` was called earlier, this
    function probabilistically chooses one of three strategies:
      - interp:         same as default (t in interp_t_range)
      - extrap:         SLERP/LERP with t in extrap_t_range (outside [0,1])
      - sphere_uniform: look-at scene-center, uniform over the spherical
                        cap defined by train-cam elevation ± margin.

    Intrinsics are copied from a randomly chosen anchor cam; image size too.
    The returned Camera's ``image`` is a dummy zero tensor — pseudo-view
    losses must never read GT from it.
    """
    if rng is None:
        rng = random
    assert len(pose_pool) >= 2, "need at least 2 training cameras to sample a pseudo view"

    wide = _WIDE_CFG
    if wide is None:
        # ---- legacy path (unchanged) ----
        a, b = rng.sample(range(len(pose_pool)), 2)
        t = float(rng.uniform(*t_range))
        cam_a, cam_b = pose_pool[a], pose_pool[b]
        va = cam_a.viewmat.to(device); vb = cam_b.viewmat.to(device)
        Ca = torch.linalg.inv(va); Cb = torch.linalg.inv(vb)
        qa = _rot_to_quat(Ca[:3, :3])
        qb = _rot_to_quat(Cb[:3, :3])
        q = _slerp(qa, qb, t)
        R = _quat_to_rot(q)
        center = (1.0 - t) * Ca[:3, 3] + t * Cb[:3, 3]
        anchor = cam_a
        name = f"pseudo_{cam_a.cam_idx}_{cam_b.cam_idx}_t{t:.2f}"
    else:
        # ---- wide path: sample a mode, then dispatch ----
        probs = wide.get("mode_probs", [0.4, 0.3, 0.3])
        if not (isinstance(probs, (list, tuple)) and len(probs) == 3):
            probs = [0.4, 0.3, 0.3]
        s = float(sum(probs))
        if s <= 0:
            probs = [0.4, 0.3, 0.3]
            s = 1.0
        probs = [p / s for p in probs]
        u = rng.random()
        if u < probs[0]:
            mode = "interp"
        elif u < probs[0] + probs[1]:
            mode = "extrap"
        else:
            mode = "sphere"
        if mode == "interp":
            lo, hi = wide.get("interp_t_range", [0.25, 0.75])
            R, center = _sample_extrapolate(pose_pool, device, (lo, hi), rng)
        elif mode == "extrap":
            lo, hi = wide.get("extrap_t_range", [-0.3, 1.3])
            R, center = _sample_extrapolate(pose_pool, device, (lo, hi), rng)
        else:
            elev_margin = float(wide.get("elevation_margin_deg", 10.0))
            r_jit = float(wide.get("radius_jitter", 0.05))
            R, center = _sample_sphere_uniform(
                pose_pool, device, elev_margin, r_jit, rng,
            )
        anchor = pose_pool[rng.randrange(len(pose_pool))]
        name = f"pseudo_wide_{mode}"

    c2w = torch.eye(4, device=device)
    c2w[:3, :3] = R
    c2w[:3, 3] = center
    viewmat = torch.linalg.inv(c2w).contiguous()

    K = anchor.K.to(device)
    W, H = anchor.width, anchor.height
    dummy_img = torch.zeros(H, W, 3, device=device)

    return Camera(
        image=dummy_img,
        viewmat=viewmat.float(),
        K=K.float(),
        width=W,
        height=H,
        image_name=name,
        cam_idx=-1,
    )
