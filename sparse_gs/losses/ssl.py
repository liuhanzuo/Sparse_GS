"""Self-supervised loss bank.

All losses share a single ``ctx`` dict produced by the trainer:

    {
        "step": int,
        "rendered": {"rgb": (H,W,3), "depth": (H,W,1), "alpha": (H,W,1), "info": ...},
        "gt_rgb":    (H,W,3),
        "camera":    Camera              # current train view
        "pose_pool": List[Camera]        # training cameras (for pseudo sampling)
        "teacher":   Optional[EMATeacher],
        "gaussians": GaussianModel,
        "renderer":  GSplatRenderer,
        "background": torch.Tensor(3,),
    }

Each loss returns ``(scalar_tensor, log_dict)``. Losses whose prerequisites
are missing (e.g. ``teacher`` is None) return a zero tensor and log nothing.
Config lives under ``cfg.ssl.<name>`` and always has at least
``enabled: bool`` and ``weight: float``; other keys are loss-specific.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..datasets.pseudo_pose import sample_pseudo_camera
from ..utils.metrics import ssim, hwc_to_bchw


LossFn = Callable[[Dict[str, Any], Dict[str, Any]], Tuple[torch.Tensor, Dict[str, float]]]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _zero_like(ctx: Dict[str, Any]) -> torch.Tensor:
    rgb = ctx["rendered"]["rgb"]
    return torch.zeros((), device=rgb.device, dtype=rgb.dtype)


def _after(step: int, start: int) -> bool:
    return step >= int(start)


def pseudo_view_warmup_weight(step: int, start_sample_pseudo: int, warmup_iters: int = 500) -> float:
    """FSGS-style pseudo-view linear warmup weight."""
    return min(max((int(step) - int(start_sample_pseudo)) / float(warmup_iters), 0.0), 1.0)


def should_sample_pseudo_view(
    step: int,
    start_sample_pseudo: int,
    end_sample_pseudo: int,
    sample_pseudo_interval: int,
) -> bool:
    """FSGS pseudo-view trigger: start < iter < end and interval matched."""
    interval = max(1, int(sample_pseudo_interval))
    return int(start_sample_pseudo) < int(step) < int(end_sample_pseudo) and int(step) % interval == 0


def _l1_ssim(pred: torch.Tensor, target: torch.Tensor, ssim_lambda: float = 0.2) -> torch.Tensor:
    """(H,W,3) -> scalar. Same formulation as the photometric baseline loss."""
    l1 = (pred - target).abs().mean()
    if ssim_lambda <= 0.0:
        return l1
    s = ssim(hwc_to_bchw(pred), hwc_to_bchw(target))
    return (1.0 - ssim_lambda) * l1 + ssim_lambda * (1.0 - s)


# --------------------------------------------------------------------------
# (1) Pseudo-view distillation with the EMA teacher
# --------------------------------------------------------------------------
def _pseudo_view(ctx, sub_cfg):
    """Render a novel (interpolated) pose with both teacher & student, match.

    Config keys:
        enabled:     bool
        weight:      float
        start_iter:  int   (legacy alias for start_sample_pseudo)
        start_sample_pseudo: int (default: start_iter or 500)
        end_sample_pseudo:   int (default: train.iterations + 1)
        sample_pseudo_interval: int (default: 1)
        ssim_lambda: float (default 0.2)
        t_range:     [lo, hi] interpolation range between 2 train cameras
        alpha_thresh: float — pixels where teacher alpha < thresh are masked
    """
    teacher = ctx.get("teacher")
    if teacher is None:
        return _zero_like(ctx), {}
    start = int(sub_cfg.get("start_sample_pseudo", sub_cfg.get("start_iter", 500)))
    end = int(sub_cfg.get("end_sample_pseudo", ctx.get("total_steps", 10**12) + 1))
    interval = int(sub_cfg.get("sample_pseudo_interval", 1))
    if not should_sample_pseudo_view(ctx["step"], start, end, interval):
        return _zero_like(ctx), {}
    loss_scale = pseudo_view_warmup_weight(ctx["step"], start)

    pose_pool: List = ctx["pose_pool"]
    if len(pose_pool) < 2:
        return _zero_like(ctx), {}

    renderer = ctx["renderer"]
    student = ctx["gaussians"]
    bg = ctx.get("background")
    device = ctx["rendered"]["rgb"].device

    t_range = tuple(sub_cfg.get("t_range", [0.25, 0.75]))
    pseudo_cam = sample_pseudo_camera(pose_pool, device=device, t_range=t_range)

    # teacher forward (no grad). We temporarily swap the renderer's gaussians.
    with torch.no_grad():
        t_out = renderer.render(
            teacher.model,
            viewmat=pseudo_cam.viewmat, K=pseudo_cam.K,
            width=pseudo_cam.width, height=pseudo_cam.height,
            active_sh_degree=teacher.model.active_sh_degree,
            background=bg,
        )

    # student forward at the same pose. Default: detach_geometry=True so
    # this branch only updates colors / opacities and never feeds gsplat's
    # 2D-mean gradient that drives densification.
    detach_geo = bool(sub_cfg.get("detach_geometry", True))
    s_out = renderer.render(
        student,
        viewmat=pseudo_cam.viewmat, K=pseudo_cam.K,
        width=pseudo_cam.width, height=pseudo_cam.height,
        active_sh_degree=student.active_sh_degree,
        background=bg,
        detach_geometry=detach_geo,
    )

    # mask: only where teacher has reasonable coverage (alpha > thresh)
    alpha_thresh = float(sub_cfg.get("alpha_thresh", 0.5))
    mask = (t_out["alpha"] > alpha_thresh).float()                # (H,W,1)
    coverage = mask.mean().clamp_min(1e-6)

    # masked L1 + SSIM
    diff = (s_out["rgb"] - t_out["rgb"]).abs() * mask             # (H,W,3)
    l1 = diff.mean() / coverage
    ssim_lambda = float(sub_cfg.get("ssim_lambda", 0.2))
    if ssim_lambda > 0:
        s = ssim(hwc_to_bchw(s_out["rgb"] * mask),
                 hwc_to_bchw(t_out["rgb"] * mask))
        loss = (1.0 - ssim_lambda) * l1 + ssim_lambda * (1.0 - s)
    else:
        loss = l1

    return loss * loss_scale, {
        "coverage": float(coverage.detach().item()),
        "warmup": float(loss_scale),
    }


# --------------------------------------------------------------------------
# (2) EMA-teacher same-view consistency
# --------------------------------------------------------------------------
def _ema_teacher_consist(ctx, sub_cfg):
    """Match the student's render of the current camera to the teacher's.

    This is a *mild* regularizer — the photometric loss already pulls
    strongly toward GT at the current camera; the teacher target mostly
    helps tame the appearance (colors / opacity) without distorting the
    geometry that the photo loss has just shaped.

    Implementation: student is re-rendered with ``detach_geometry=True``
    so the loss back-props only into sh0/shN/opacities. This guarantees
    we do not corrupt gsplat's 2D-mean gradient that drives densification.

    Config:
        enabled, weight
        start_iter:      int
        ssim_lambda:     float
        depth_weight:    float (extra L1 on depth if >0)
    """
    teacher = ctx.get("teacher")
    if teacher is None:
        return _zero_like(ctx), {}
    start = int(sub_cfg.get("start_iter", 500))
    if not _after(ctx["step"], start):
        return _zero_like(ctx), {}

    renderer = ctx["renderer"]
    student = ctx["gaussians"]
    cam = ctx["camera"]
    bg = ctx.get("background")

    with torch.no_grad():
        t_out = renderer.render(
            teacher.model,
            viewmat=cam.viewmat, K=cam.K,
            width=cam.width, height=cam.height,
            active_sh_degree=teacher.model.active_sh_degree,
            background=bg,
        )

    # Re-render the student with detached geometry so this branch
    # contributes only to color/opacity gradients.
    s_out = renderer.render(
        student,
        viewmat=cam.viewmat, K=cam.K,
        width=cam.width, height=cam.height,
        active_sh_degree=student.active_sh_degree,
        background=bg,
        detach_geometry=True,
    )

    ssim_lambda = float(sub_cfg.get("ssim_lambda", 0.1))
    loss = _l1_ssim(s_out["rgb"], t_out["rgb"], ssim_lambda=ssim_lambda)

    dw = float(sub_cfg.get("depth_weight", 0.0))
    if dw > 0.0 and s_out.get("depth") is not None and t_out.get("depth") is not None:
        d_loss = (s_out["depth"] - t_out["depth"]).abs().mean()
        loss = loss + dw * d_loss

    return loss, {}


# --------------------------------------------------------------------------
# (3) Multi-view photometric consistency
# --------------------------------------------------------------------------
def _pick_neighbor(pose_pool: List, current_idx: int, rng: random.Random) -> int:
    """Pick a neighbor camera index from the pool, distinct from current."""
    n = len(pose_pool)
    if n < 2:
        return current_idx
    j = rng.randrange(n - 1)
    if j >= current_idx:
        j += 1
    return j


def _multiview_photo(ctx, sub_cfg):
    """Cross-view photometric consistency via depth-based reprojection.

    Procedure (per training step):
      1. Render the current camera (already done) to obtain student depth D_a.
      2. Pick a neighbor training camera b (with real GT image I_b).
      3. Un-project every pixel of camera a using D_a to a 3D point in world.
      4. Project that point into camera b -> uv_b in pixel space.
      5. ``grid_sample`` I_b at uv_b -> warped image I_{b->a}.
      6. Compute L1 between I_{b->a} and the current GT I_a, masked by:
           - in-bounds projection
           - student alpha > thresh (skip empty regions)
           - depth > 0 (skip pixels where student saw nothing)

    This does *not* use any external prior (no monocular depth, no DINO).
    It only uses the existing sparse training images, but couples them
    through the *student's own* geometry — so the loss directly penalizes
    geometry that is inconsistent across views, which is exactly the
    sparse-view failure mode (floaters / wrong depth in unconstrained
    regions).

    Config keys:
        enabled, weight
        start_iter:       int   (default 1500) — wait for depth to be sane
        n_neighbors:      int   (default 1)    — average over k neighbors
        alpha_thresh:     float (default 0.5)  — student alpha mask
        ssim_lambda:      float (default 0.0)  — SSIM is noisy on warped images
        depth_eps:        float (default 1e-3) — skip near-zero depth
        occlusion_check:  bool  (default False)— forward-backward depth check
        occlusion_tau:    float (default 0.05) — relative-depth tolerance
        occlusion_alpha_thresh: float (default 0.5) — cam_b student alpha mask
        occlusion_start_iter: int (default = start_iter) — delay occ. check
            until geometry is more reliable. Lets the early SSL signal run
            unfiltered; useful in very sparse (n ≤ 3) settings where the
            student's depth is too noisy to be a trustworthy occlusion
            oracle at iter = start_iter.
    """
    start = int(sub_cfg.get("start_iter", 1500))
    if not _after(ctx["step"], start):
        return _zero_like(ctx), {}

    pose_pool: List = ctx["pose_pool"]
    if len(pose_pool) < 2:
        return _zero_like(ctx), {}

    # We do NOT reuse ctx["rendered"] (= main forward) here, because gsplat's
    # DefaultStrategy retains and consumes the 2D-mean gradient of the main
    # forward's `info["means2d"]` to drive densification. Routing an extra
    # reprojection loss through that same forward inflates grad2d and causes
    # uncontrolled densify. Instead, we do a *separate* student forward for
    # this loss; its `info` is never handed to the strategy, so densification
    # remains driven purely by the photometric loss while geometry parameters
    # still receive the multi-view gradient via standard autograd.
    cam_a = ctx["camera"]
    gt_a = ctx["gt_rgb"]                  # (H,W,3)
    H, W = gt_a.shape[:2]
    device = gt_a.device

    renderer = ctx["renderer"]
    student = ctx["gaussians"]
    bg = ctx.get("background")
    out_a = renderer.render(
        student,
        viewmat=cam_a.viewmat, K=cam_a.K,
        width=cam_a.width, height=cam_a.height,
        active_sh_degree=student.active_sh_degree,
        background=bg,
        detach_geometry=False,
    )
    pred_a_local = out_a["rgb"]
    depth = out_a["depth"]                 # (H,W,1)
    alpha = out_a["alpha"]                 # (H,W,1)
    if depth is None or alpha is None:
        return _zero_like(ctx), {}

    # current view's pose-pool index (so we don't pick ourselves)
    cur_idx = -1
    for i, c in enumerate(pose_pool):
        if c.cam_idx == cam_a.cam_idx:
            cur_idx = i
            break

    n_neighbors = max(1, int(sub_cfg.get("n_neighbors", 1)))
    rng = random
    alpha_thresh = float(sub_cfg.get("alpha_thresh", 0.5))
    depth_eps = float(sub_cfg.get("depth_eps", 1e-3))
    occlusion_check = bool(sub_cfg.get("occlusion_check", False))
    occlusion_tau = float(sub_cfg.get("occlusion_tau", 0.05))
    occ_alpha_thresh = float(sub_cfg.get("occlusion_alpha_thresh", 0.5))
    # If not set, the check is active from the moment the loss itself is
    # active. Set higher than start_iter to delay only the occlusion gate.
    occlusion_start_iter = int(sub_cfg.get("occlusion_start_iter", start))
    occlusion_active = occlusion_check and _after(ctx["step"], occlusion_start_iter)

    # ---- build a pixel grid of camera A in homogeneous camera coords ----
    # We do it once per call; no need to cache across steps.
    K_a = cam_a.K.to(device)
    Ka_inv = torch.linalg.inv(K_a)

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pix = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs)], dim=-1)  # (H,W,3)
    rays_cam = pix @ Ka_inv.T                                              # (H,W,3)

    # ---- pixels in world: P = C2W * (rays_cam * depth) ----
    c2w_a = torch.linalg.inv(cam_a.viewmat.to(device))
    R_a, t_a = c2w_a[:3, :3], c2w_a[:3, 3]
    pts_cam_a = rays_cam * depth                                           # (H,W,3)
    pts_world = pts_cam_a @ R_a.T + t_a                                    # (H,W,3)

    # student mask of where its prediction is meaningful
    valid_a = ((alpha[..., 0] > alpha_thresh) & (depth[..., 0] > depth_eps))  # (H,W) bool

    total_loss = _zero_like(ctx)
    total_cov = 0.0
    total_kept_ratio = 0.0
    n_used = 0
    n_occ_logged = 0

    tried = set([cur_idx])
    for _ in range(n_neighbors):
        # pick a fresh neighbor
        b_idx = _pick_neighbor(pose_pool, cur_idx if cur_idx >= 0 else 0, rng)
        if b_idx in tried:
            continue
        tried.add(b_idx)

        cam_b = pose_pool[b_idx]
        viewmat_b = cam_b.viewmat.to(device)
        K_b = cam_b.K.to(device)
        gt_b = cam_b.image.to(device)                                      # (H,W,3)

        # world -> camera B
        Rb, tb = viewmat_b[:3, :3], viewmat_b[:3, 3]
        pts_cam_b = pts_world @ Rb.T + tb                                  # (H,W,3)
        z_b = pts_cam_b[..., 2:3]
        in_front = (z_b[..., 0] > depth_eps)

        # project
        uv_b = pts_cam_b @ K_b.T                                           # (H,W,3)
        uv = uv_b[..., :2] / uv_b[..., 2:3].clamp_min(depth_eps)           # (H,W,2)

        # in-bounds mask in pixel coords
        in_bounds = (
            (uv[..., 0] >= 0) & (uv[..., 0] < (W - 1))
            & (uv[..., 1] >= 0) & (uv[..., 1] < (H - 1))
        )
        mask = (valid_a & in_front & in_bounds).float()                    # (H,W)
        if mask.sum() < 64:
            continue

        # grid_sample expects uv in [-1, 1] and shape (1, H, W, 2)
        u_norm = 2.0 * uv[..., 0] / (W - 1) - 1.0
        v_norm = 2.0 * uv[..., 1] / (H - 1) - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0)          # (1,H,W,2)

        # ---- optional forward-backward depth occlusion check ----------
        # Render the student's depth from cam_b (no grad — this branch only
        # builds a mask) and compare it to the warped z_b of every pixel.
        # If z_b is much larger than D_b(uv), pixel A projects behind the
        # student's surface at B (i.e. is occluded) and should be dropped.
        # We also drop pixels where cam_b has no meaningful coverage.
        if occlusion_active:
            with torch.no_grad():
                out_b = renderer.render(
                    student,
                    viewmat=viewmat_b, K=K_b,
                    width=cam_b.width, height=cam_b.height,
                    active_sh_degree=student.active_sh_degree,
                    background=bg,
                    detach_geometry=True,
                )
                depth_b = out_b.get("depth")
                alpha_b = out_b.get("alpha")
            if depth_b is not None and alpha_b is not None:
                # sample depth_b at uv. depth_b is (H,W,1); reshape to (1,1,H,W).
                depth_b_chw = depth_b.permute(2, 0, 1).unsqueeze(0)         # (1,1,H,W)
                d_b_at_uv = F.grid_sample(
                    depth_b_chw, grid, mode="bilinear",
                    padding_mode="border", align_corners=True,
                ).squeeze(0).squeeze(0)                                     # (H,W)
                alpha_b_chw = alpha_b.permute(2, 0, 1).unsqueeze(0)
                a_b_at_uv = F.grid_sample(
                    alpha_b_chw, grid, mode="bilinear",
                    padding_mode="border", align_corners=True,
                ).squeeze(0).squeeze(0)                                     # (H,W)

                z = z_b[..., 0].clamp_min(depth_eps)
                rel = (z - d_b_at_uv).abs() / z
                # "not occluded" = projected depth agrees AND cam_b saw sth here
                not_occluded = (rel < occlusion_tau) & (a_b_at_uv > occ_alpha_thresh)
                # track: of pixels that were in-bounds & in-front, how many survived?
                pre_sum = mask.sum().clamp_min(1.0)
                mask = mask * not_occluded.float()
                total_kept_ratio += float((mask.sum() / pre_sum).detach().item())
                n_occ_logged += 1
                if mask.sum() < 64:
                    continue

        # gt_b is (H,W,3); F.grid_sample wants (N,C,H,W)
        gt_b_chw = gt_b.permute(2, 0, 1).unsqueeze(0)                      # (1,3,H,W)
        warped = F.grid_sample(gt_b_chw, grid, mode="bilinear",
                               padding_mode="border", align_corners=True)  # (1,3,H,W)
        warped = warped.squeeze(0).permute(1, 2, 0)                        # (H,W,3)

        # masked L1 between the student's render of camera A (separate forward,
        # see top of this function) and the cross-view reprojection of GT_b.
        # GT_a doesn't depend on the student, so comparing pred_a vs warped_b
        # is what makes the loss back-prop into geometry / colors.
        m3 = mask.unsqueeze(-1)
        cov = mask.mean().clamp_min(1e-6)
        diff = (pred_a_local - warped).abs() * m3                          # (H,W,3)
        l1 = diff.mean() / cov

        ssim_lambda = float(sub_cfg.get("ssim_lambda", 0.0))
        if ssim_lambda > 0:
            s = ssim(hwc_to_bchw(pred_a_local * m3), hwc_to_bchw(warped * m3))
            term = (1.0 - ssim_lambda) * l1 + ssim_lambda * (1.0 - s)
        else:
            term = l1

        total_loss = total_loss + term
        total_cov += float(cov.detach().item())
        n_used += 1

    if n_used == 0:
        return _zero_like(ctx), {}

    logs = {"coverage": total_cov / n_used, "n_used": float(n_used)}
    if n_occ_logged > 0:
        logs["kept_ratio"] = total_kept_ratio / n_occ_logged
    return total_loss / n_used, logs


# --------------------------------------------------------------------------
# (4) Depth / geometry consistency  —  scale-shift-invariant matching
#     between the rendered depth and a per-view monocular-depth prior
#     stored on ``Camera.mono_depth`` (set by the dataset loader).
# --------------------------------------------------------------------------
def _ssi_normalize(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
                   ) -> torch.Tensor:
    """Scale-shift-invariant normalization of ``x`` inside ``mask``.

    Returns ``(x - shift) / scale``, where ``shift`` and ``scale`` are
    computed on the masked pixels only. Unmasked pixels are still
    returned but the caller is expected to ignore them via the same mask.

    This is a fixed-shape, all-arithmetic approximation of the MiDaS
    median-MAD SSI transform. We deliberately avoid::

        flat = x[mask]        # boolean-indexing -> dynamic-shape gather
        med  = flat.median()  # sort over dynamic-shape input

    because that pattern (a) does a D2H sync via the implicit ``flat.numel()``
    Python check, and (b) launches a sort kernel whose input shape changes
    every step (since the alpha-derived ``mask`` jitters by a few pixels per
    iteration). On Windows + recent CUDA drivers we observed this to
    repeatedly trigger transient TDR resets after a few hundred iterations.

    The mean-std variant below is fully fixed-shape: only elementwise ops
    plus a single masked sum / sumsq. It is functionally close to median-MAD
    for our use case (matching depth ordering against a monocular prior),
    because both are equivariant under the same affine group ``y = a*x + b``.

    Shapes
    ------
    x:    (H, W, 1)
    mask: (H, W)   bool / float
    """
    if mask.dtype == torch.bool:
        m = mask.float()
    else:
        m = mask
    m1 = m.unsqueeze(-1)                                 # (H, W, 1)
    n = m.sum().clamp_min(1.0)                           # scalar tensor (no .item)
    shift = (x * m1).sum() / n
    var = ((x - shift) ** 2 * m1).sum() / n
    scale = var.clamp_min(eps).sqrt()
    return (x - shift) / scale


def _depth_grad_l1(pred_n: torch.Tensor, prior_n: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked first-order gradient consistency on normalized depth/disparity."""
    m = mask.float()
    dx_m = m[:, 1:] * m[:, :-1]
    dy_m = m[1:, :] * m[:-1, :]
    pred_x = pred_n[:, 1:, :] - pred_n[:, :-1, :]
    prior_x = prior_n[:, 1:, :] - prior_n[:, :-1, :]
    pred_y = pred_n[1:, :, :] - pred_n[:-1, :, :]
    prior_y = prior_n[1:, :, :] - prior_n[:-1, :, :]
    loss_x = ((pred_x - prior_x).abs() * dx_m.unsqueeze(-1)).mean()
    loss_y = ((pred_y - prior_y).abs() * dy_m.unsqueeze(-1)).mean()
    return loss_x + loss_y


def _pearson_corrcoef(src: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pearson correlation for flattened per-sample depth tensors."""
    if src.shape != target.shape:
        raise ValueError(f"Pearson inputs must have the same shape, got {tuple(src.shape)} and {tuple(target.shape)}")
    if src.ndim <= 2:
        src = src.reshape(1, -1)
        target = target.reshape(1, -1)
    else:
        src = src.reshape(src.shape[0], -1)
        target = target.reshape(target.shape[0], -1)

    src = src - src.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    src = src / (src.std(dim=1, keepdim=True, unbiased=False) + eps)
    target = target / (target.std(dim=1, keepdim=True, unbiased=False) + eps)
    return (src * target).mean(dim=1)


def pearson_depth_loss(
    rend_depth: torch.Tensor,
    prior_depth: torch.Tensor,
    target_transform: str = "identity",
    inverse_offset: float = 200.0,
) -> torch.Tensor:
    """Global Pearson depth/disparity loss, train-view only.

    ``rend_depth`` and ``prior_depth`` must already live in the same space
    (depth or disparity) unless ``target_transform`` explicitly requests the
    legacy FSGS transform. Invalid pixels should be masked by the caller.
    """
    if rend_depth.shape != prior_depth.shape:
        raise ValueError(
            f"rend_depth and prior_depth must have the same shape, got "
            f"{tuple(rend_depth.shape)} and {tuple(prior_depth.shape)}"
        )
    rend = rend_depth.float()
    prior = prior_depth.float()
    transform = str(target_transform).lower()

    if transform in ("identity", "same", "none"):
        return (1.0 - _pearson_corrcoef(prior, rend)).mean()
    if transform in ("negative", "neg", "flip"):
        return (1.0 - _pearson_corrcoef(-prior, rend)).mean()
    if transform in ("inverse", "inv"):
        return (1.0 - _pearson_corrcoef(1.0 / (prior + float(inverse_offset)), rend)).mean()
    if transform == "fsgs":
        loss_neg = 1.0 - _pearson_corrcoef(-prior, rend)
        loss_inv = 1.0 - _pearson_corrcoef(1.0 / (prior + float(inverse_offset)), rend)
        return torch.minimum(loss_neg, loss_inv).mean()
    raise ValueError(
        "pearson_depth_loss.target_transform must be one of "
        f"'identity', 'negative', 'inverse', or 'fsgs', got {target_transform!r}"
    )


def _local_depth_l1(
    pred_d: torch.Tensor,
    prior_d: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    n_patches: int,
) -> torch.Tensor:
    """DNGaussian-style local SSI depth loss over random same-view patches."""
    H, W = mask.shape
    p = max(4, min(int(patch_size), H, W))
    n = max(1, int(n_patches))
    total = pred_d.new_zeros(())
    for _ in range(n):
        top = random.randint(0, H - p) if H > p else 0
        left = random.randint(0, W - p) if W > p else 0
        sl_y = slice(top, top + p)
        sl_x = slice(left, left + p)
        m = mask[sl_y, sl_x]
        pn = _ssi_normalize(pred_d[sl_y, sl_x], m)
        qn = _ssi_normalize(prior_d[sl_y, sl_x], m)
        total = total + ((pn - qn).abs() * m.unsqueeze(-1).float()).mean()
    return total / n


def _depth_consist(ctx, sub_cfg):
    """Match rendered depth to monocular depth prior.

    ``mode`` controls which train-view depth prior loss is active:
      - ``"depthv2"``: existing global/local/grad SSI L1 branch.
      - ``"pearson"``: FSGS-style global Pearson branch.
      - ``"both"``: weighted sum of the two branches.

    Missing ``mode`` keeps the historical depthv2 behavior.
    """
    start = int(sub_cfg.get("start_iter", 0))
    if not _after(ctx["step"], start):
        return _zero_like(ctx), {}

    cam = ctx["camera"]
    prior = getattr(cam, "mono_depth", None)
    if prior is None:
        return _zero_like(ctx), {}

    rendered = ctx["rendered"]
    depth = rendered.get("depth")
    alpha = rendered.get("alpha")
    if depth is None or alpha is None:
        return _zero_like(ctx), {}

    prior = prior.to(depth.device)
    alpha_thresh = float(sub_cfg.get("alpha_thresh", 0.5))
    depth_eps = float(sub_cfg.get("depth_eps", 1e-3))
    space = str(sub_cfg.get("space", "disparity")).lower()
    mode = str(sub_cfg.get("mode", "depthv2")).lower()
    if mode not in ("depthv2", "pearson", "both"):
        raise ValueError(f"depth_consist.mode must be 'depthv2', 'pearson', or 'both', got {mode!r}")

    if space == "disparity":
        pred_d = 1.0 / depth.clamp_min(depth_eps)
        kind = getattr(cam, "depth_kind", None)
        prior_d = 1.0 / prior.clamp_min(depth_eps) if kind == "metric" else prior
    elif space == "depth":
        pred_d = depth
        kind = getattr(cam, "depth_kind", None)
        prior_d = 1.0 / prior.clamp_min(depth_eps) if kind == "disparity" else prior
    else:
        raise ValueError(f"depth_consist.space must be 'disparity' or 'depth', got {space!r}")

    if bool(sub_cfg.get("flip_prior", False)):
        prior_d = -prior_d

    mask = (alpha[..., 0] > alpha_thresh)
    logs = {"coverage": float(mask.float().mean().detach().item())}
    loss = depth.new_zeros(())

    if mode in ("depthv2", "both"):
        pred_n = _ssi_normalize(pred_d, mask)
        prior_n = _ssi_normalize(prior_d, mask)
        m1 = mask.unsqueeze(-1).float()

        global_term = ((pred_n - prior_n).abs() * m1).mean()
        global_w = float(sub_cfg.get("global_weight", 1.0))
        local_w = float(sub_cfg.get("local_weight", 0.0))
        grad_w = float(sub_cfg.get("grad_weight", 0.0))

        depthv2_term = global_w * global_term
        logs["global"] = float(global_term.detach().item())

        if local_w > 0.0:
            local_term = _local_depth_l1(
                pred_d, prior_d, mask,
                patch_size=int(sub_cfg.get("local_patch_size", sub_cfg.get("patch_size", 96))),
                n_patches=int(sub_cfg.get("local_n_patches", sub_cfg.get("n_patches", 8))),
            )
            depthv2_term = depthv2_term + local_w * local_term
            logs["local"] = float(local_term.detach().item())

        if grad_w > 0.0:
            grad_term = _depth_grad_l1(pred_n, prior_n, mask)
            depthv2_term = depthv2_term + grad_w * grad_term
            logs["grad"] = float(grad_term.detach().item())

        depthv2_w = float(sub_cfg.get("depthv2_weight", 1.0))
        loss = loss + depthv2_w * depthv2_term
        logs["depthv2"] = float(depthv2_term.detach().item())
        logs["depthv2_weighted"] = float((depthv2_w * depthv2_term).detach().item())

    if mode in ("pearson", "both"):
        if mask.sum() < 2:
            pearson_term = depth.new_zeros(())
        else:
            pearson_term = pearson_depth_loss(
                pred_d[..., 0][mask],
                prior_d[..., 0][mask],
                target_transform=str(sub_cfg.get("pearson_target_transform", "identity")),
                inverse_offset=float(sub_cfg.get("pearson_inverse_offset", 200.0)),
            )
        pearson_w = float(sub_cfg.get("pearson_weight", 0.05))
        loss = loss + pearson_w * pearson_term
        logs["pearson"] = float(pearson_term.detach().item())
        logs["pearson_weighted"] = float((pearson_w * pearson_term).detach().item())

    return loss, logs


# --------------------------------------------------------------------------
# (5) Feature-level (DINO/VGG) consistency   [TODO -- placeholder]
# --------------------------------------------------------------------------
def _feature_consist(ctx, sub_cfg):
    # TODO: run a frozen backbone on rendered + gt patches, match features.
    return _zero_like(ctx), {}


# --------------------------------------------------------------------------
# (6) Surface-flatten regularizer
#
#     Make every Gaussian *anisotropic in one direction* — push the smallest
#     scale toward zero so each Gaussian behaves like a 2D surfel
#     (disc-shaped). This is the geometric prior that turns 3DGS into a
#     surface-reconstruction model (SuGaR / 2DGS / PGSR / GaussianSurfels
#     all use a variant of this).
#
#     We do not need to call gsplat for this — it is a pure parameter-space
#     regularizer on Gaussian scales.
# --------------------------------------------------------------------------
def _surface_flatten(ctx, sub_cfg):
    """Penalize the smallest scale of each Gaussian.

    Two flavours, controlled by ``mode``:

      - ``"abs"``   :  loss = mean(min_scale)
            Cheapest. Pushes the thinnest axis toward 0 in absolute world
            units. Stable but couples flatness with overall size: tiny
            Gaussians satisfy this trivially with no flatness gain.

      - ``"ratio"`` :  loss = mean(min_scale / max_scale)
            Scale-invariant; only the *aspect ratio* matters. This is what
            2DGS / PGSR use. Recommended when you want flat Gaussians of
            any size; it does not bias densification toward giant
            Gaussians the way ``abs`` indirectly can.

    Optional masking keeps the loss off until a warmup is finished and
    only applies to opacity > thresh (mature Gaussians) so newborn
    Gaussians from densification are not yanked flat before they have
    found their place.

    Config keys
    -----------
        enabled, weight
        start_iter:    int   (default 1000) — warmup before geometry settles
        mode:          "abs" | "ratio"  (default "ratio")
        opacity_thresh:float (default 0.0) — only penalise Gaussians with
                       sigmoid(opacity) > this; 0.0 = penalise all.
        eps:           float (default 1e-8) — for ratio division
    """
    start = int(sub_cfg.get("start_iter", 1000))
    if not _after(ctx["step"], start):
        return _zero_like(ctx), {}

    g = ctx["gaussians"]
    raw_scales = g.params["scales"]                         # (N, 3) log-space
    scales = torch.exp(raw_scales)                          # (N, 3) real units

    op_thresh = float(sub_cfg.get("opacity_thresh", 0.0))
    if op_thresh > 0.0:
        opa = torch.sigmoid(g.params["opacities"])          # (N,)
        mask = (opa > op_thresh)
        if mask.sum() < 8:
            return _zero_like(ctx), {}
        scales = scales[mask]

    s_min = scales.min(dim=-1).values                       # (N,)
    mode = str(sub_cfg.get("mode", "ratio")).lower()
    if mode == "abs":
        loss = s_min.mean()
        log_extra = {"min_scale": float(s_min.detach().mean().item())}
    elif mode == "ratio":
        s_max = scales.max(dim=-1).values.clamp_min(float(sub_cfg.get("eps", 1e-8)))
        ratio = s_min / s_max                               # (N,) in (0, 1]
        loss = ratio.mean()
        log_extra = {
            "min_scale":   float(s_min.detach().mean().item()),
            "ratio_mean":  float(ratio.detach().mean().item()),
        }
    else:
        raise ValueError(f"surface_flatten.mode must be 'abs' or 'ratio', got {mode!r}")

    return loss, log_extra


# --------------------------------------------------------------------------
# (7) Normal smoothness from rendered depth
#
#     Use the rendered depth to derive a per-pixel surface normal
#     (cross product of finite-difference world-space tangents) and
#     penalize its TV / first-order discontinuities inside the foreground
#     mask.
#
#     Why it helps sparse-view: with very few views the photo loss alone
#     leaves the depth map noisy in unconstrained regions, which yields
#     scattered normals and a "splatted-noise" surface. A weak normal-TV
#     prior pulls the surface back to a piecewise-smooth manifold without
#     biasing photometry.
#
#     This is a *self-supervised* loss — it uses no external prior, only
#     the student's own rendered depth + camera intrinsics.
# --------------------------------------------------------------------------
def _backproject_depth_to_world(
    depth: torch.Tensor,         # (H, W, 1)
    K: torch.Tensor,             # (3, 3)
    viewmat: torch.Tensor,       # (4, 4) world->camera
) -> torch.Tensor:
    """Lift a depth map to a (H, W, 3) world-space point cloud."""
    H, W = depth.shape[:2]
    device = depth.device
    K_inv = torch.linalg.inv(K)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pix = torch.stack([xs + 0.5, ys + 0.5, torch.ones_like(xs)], dim=-1)  # (H,W,3)
    rays_cam = pix @ K_inv.T                                              # (H,W,3)
    pts_cam = rays_cam * depth                                            # (H,W,3)
    c2w = torch.linalg.inv(viewmat)
    R, t = c2w[:3, :3], c2w[:3, 3]
    return pts_cam @ R.T + t                                              # (H,W,3)


def _depth_to_normal(
    depth: torch.Tensor,         # (H, W, 1)
    K: torch.Tensor,
    viewmat: torch.Tensor,
) -> torch.Tensor:
    """Per-pixel world-space normal from finite differences of the depth."""
    pts = _backproject_depth_to_world(depth, K, viewmat)                  # (H,W,3)
    # tangent vectors via central differences (interior) -> (H-2, W-2, 3)
    dpdu = pts[1:-1, 2:, :] - pts[1:-1, :-2, :]                           # +x neighbours
    dpdv = pts[2:, 1:-1, :] - pts[:-2, 1:-1, :]                           # +y neighbours
    n = torch.cross(dpdu, dpdv, dim=-1)
    n = F.normalize(n, dim=-1, eps=1e-8)                                  # (H-2,W-2,3)
    # pad to (H,W,3) so callers can apply pixel-aligned masks. Border
    # pixels are zero, which is fine since we mask them out anyway.
    out = torch.zeros_like(pts)
    out[1:-1, 1:-1, :] = n
    return out


def _normal_smooth(ctx, sub_cfg):
    """Total-variation regularizer on the depth-derived normal map.

    Loss is computed as the masked mean of ``1 - cos(N(p), N(q))`` over
    all 4-connected neighbour pairs (p, q) inside the foreground mask.
    Equivalently: angular TV on the normal field.

    Config keys
    -----------
        enabled, weight
        start_iter:    int   (default 1500)  — wait for depth to be sane
        alpha_thresh:  float (default 0.5)   — foreground mask
        depth_eps:     float (default 1e-3)  — skip near-zero depth
        edge_aware:    bool  (default False) — down-weight TV at RGB edges
                       (classic edge-aware smoothness; useful in real
                       captures with texture seams).
        edge_lambda:   float (default 10.0)  — sharpness of the RGB edge gate
    """
    start = int(sub_cfg.get("start_iter", 1500))
    if not _after(ctx["step"], start):
        return _zero_like(ctx), {}

    rendered = ctx["rendered"]
    depth = rendered.get("depth")
    alpha = rendered.get("alpha")
    rgb = rendered.get("rgb")
    if depth is None or alpha is None:
        return _zero_like(ctx), {}

    cam = ctx["camera"]
    device = depth.device
    K = cam.K.to(device)
    viewmat = cam.viewmat.to(device)

    alpha_thresh = float(sub_cfg.get("alpha_thresh", 0.5))
    depth_eps = float(sub_cfg.get("depth_eps", 1e-3))

    fg = (alpha[..., 0] > alpha_thresh) & (depth[..., 0] > depth_eps)     # (H,W)
    if fg.sum() < 64:
        return _zero_like(ctx), {}

    normals = _depth_to_normal(depth, K, viewmat)                         # (H,W,3)

    # 4-connected angular TV. We compute (1 - dot) on x and y neighbour pairs
    # and mask by the AND of both endpoints' foreground masks.
    nx0 = normals[:, :-1, :]
    nx1 = normals[:, 1:,  :]
    dot_x = (nx0 * nx1).sum(dim=-1).clamp(-1.0, 1.0)                      # (H, W-1)
    mask_x = (fg[:, :-1] & fg[:, 1:]).float()

    ny0 = normals[:-1, :, :]
    ny1 = normals[1:,  :, :]
    dot_y = (ny0 * ny1).sum(dim=-1).clamp(-1.0, 1.0)                      # (H-1, W)
    mask_y = (fg[:-1, :] & fg[1:, :]).float()

    # optional RGB-edge gating: where the rendered RGB has a strong gradient
    # we believe a real geometry edge could exist there too, so we down-
    # weight smoothness. Classic Garg-style formulation.
    if bool(sub_cfg.get("edge_aware", False)) and rgb is not None:
        edge_lambda = float(sub_cfg.get("edge_lambda", 10.0))
        gx = (rgb[:, 1:, :] - rgb[:, :-1, :]).abs().mean(dim=-1)          # (H, W-1)
        gy = (rgb[1:, :, :] - rgb[:-1, :, :]).abs().mean(dim=-1)          # (H-1, W)
        mask_x = mask_x * torch.exp(-edge_lambda * gx)
        mask_y = mask_y * torch.exp(-edge_lambda * gy)

    tv_x = (1.0 - dot_x) * mask_x
    tv_y = (1.0 - dot_y) * mask_y

    mass = (mask_x.sum() + mask_y.sum()).clamp_min(1.0)
    loss = (tv_x.sum() + tv_y.sum()) / mass

    return loss, {
        "fg_ratio": float(fg.float().mean().detach().item()),
    }


# --------------------------------------------------------------------------
# aggregator
# --------------------------------------------------------------------------
class SSLLossBank:
    """Combines all enabled SSL losses into a single scalar + log dict."""

    LOSSES: Dict[str, LossFn] = {
        "pseudo_view":       _pseudo_view,
        "ema_teacher":       _ema_teacher_consist,
        "multiview_photo":   _multiview_photo,
        "depth_consist":     _depth_consist,
        "feature":           _feature_consist,
        "surface_flatten":   _surface_flatten,
        "normal_smooth":     _normal_smooth,
    }

    def __init__(self, ssl_cfg: Dict[str, Any]):
        self.cfg = ssl_cfg or {}
        self.enabled: List[str] = [
            name for name, sub in self.cfg.items()
            if name in self.LOSSES and isinstance(sub, dict) and bool(sub.get("enabled", False))
        ]

    def __call__(self, **ctx) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self.enabled:
            return _zero_like(ctx), {}

        step = int(ctx.get("step", 0))
        total = _zero_like(ctx)
        logs: Dict[str, float] = {}
        any_nonzero = False
        for name in self.enabled:
            fn = self.LOSSES.get(name)
            if fn is None:
                continue
            sub = self.cfg.get(name, {})
            w = float(sub.get("weight", 0.0))
            if w == 0.0:
                continue
            # Short-circuit: if the term is gated behind a start_iter, skip the
            # fn call entirely. This avoids the per-step ``.item()`` D2H sync
            # cost (otherwise paid even when fn returns ``_zero_like(ctx)``)
            # and, more importantly, avoids issuing any GPU work that would
            # otherwise interleave with rendering / backward on Windows.
            start = int(sub.get("start_iter", 0))
            if step < start:
                continue
            term, sublog = fn(ctx, sub)
            if term.requires_grad or term.detach().abs().item() > 0:
                total = total + w * term
                any_nonzero = True
            logs[f"ssl/{name}"] = float(term.detach().item()) if term.ndim == 0 else 0.0
            for k, v in sublog.items():
                logs[f"ssl/{name}/{k}"] = float(v)
        if any_nonzero:
            logs["ssl/total"] = float(total.detach().item())
        return total, logs

    def requires_teacher(self) -> bool:
        """Does any enabled loss consult the EMA teacher?"""
        for name in self.enabled:
            sub = self.cfg.get(name, {})
            if float(sub.get("weight", 0.0)) == 0.0:
                continue
            if name in ("pseudo_view", "ema_teacher"):
                return True
        # also honor a top-level override (cfg.ssl.ema_teacher.enabled alone
        # is enough to build a teacher even if its weight is 0 temporarily).
        ema = self.cfg.get("ema_teacher", {})
        if isinstance(ema, dict) and bool(ema.get("enabled", False)):
            return True
        return False

    def __repr__(self) -> str:
        return f"SSLLossBank(enabled={self.enabled})"
