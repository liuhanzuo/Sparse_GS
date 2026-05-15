"""Smoke test: instantiate GaussianModel + renderer + strategy and run one
forward+backward step with synthetic data. NO dataset required.

Run from ``d:/SSL/sparse_gs``:

    python -m scripts.smoke_test
"""

from __future__ import annotations

# Robust bootstrap: works for both ``python -m scripts.smoke_test`` and
# ``python d:/SSL/sparse_gs/scripts/smoke_test.py``.
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts import _bootstrap                                    # noqa: E402,F401

import time                                                       # noqa: E402

import torch                                                      # noqa: E402

from sparse_gs.losses.photometric import photometric_loss         # noqa: E402
from sparse_gs.losses.ssl import SSLLossBank
from sparse_gs.models.gaussians import GaussianModel
from sparse_gs.rendering.gsplat_renderer import GSplatRenderer
from sparse_gs.strategies.densify import build_strategy_and_optimizers


def _fake_cfg():
    return {
        "model": {"sh_degree": 1, "scale_init_factor": 0.05,
                  "init": {"type": "random_in_box", "num_points": 2000,
                           "extent": 1.0, "rgb_init": 0.5}},
        "strategy": {"type": "default", "absgrad": True,
                     "refine_start_iter": 100, "refine_stop_iter": 1000},
        "optim": {
            "means_lr": 1.6e-4, "scales_lr": 5.0e-3, "quats_lr": 1.0e-3,
            "opacities_lr": 5.0e-2, "sh0_lr": 2.5e-3, "shN_lr": 1.25e-4,
            "eps": 1e-15,
        },
        "ssl": {},
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device = {device}")

    cfg = _fake_cfg()
    g = GaussianModel(sh_degree=cfg["model"]["sh_degree"]).to(device)
    g.init_random(num_points=cfg["model"]["init"]["num_points"],
                  extent=cfg["model"]["init"]["extent"],
                  rgb_init=cfg["model"]["init"]["rgb_init"],
                  scale_init_factor=cfg["model"]["scale_init_factor"],
                  device=device)
    print(f"[smoke] #gaussians = {g.num_points}")

    renderer = GSplatRenderer(sh_degree=cfg["model"]["sh_degree"],
                              rasterize_mode="antialiased",
                              packed=False, absgrad=True,
                              render_mode="RGB+ED")

    strategy, opts, state = build_strategy_and_optimizers(g, cfg, scene_scale=1.0)

    # Fake camera: identity viewmat shifted so the cube is in front of camera.
    H = W = 128
    fx = fy = 110.0
    K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], device=device)
    viewmat = torch.eye(4, device=device); viewmat[2, 3] = 3.0   # cam at z=-3

    gt = torch.full((H, W, 3), 0.7, device=device)
    bg = torch.tensor([1.0, 1.0, 1.0], device=device)

    bank = SSLLossBank({})    # all disabled

    t0 = time.time()
    for step in range(1, 6):
        out = renderer.render(g, viewmat=viewmat, K=K, width=W, height=H,
                              active_sh_degree=0, background=bg)
        strategy.step_pre_backward(g.params, opts, state, step, out["info"])
        loss = photometric_loss(out["rgb"], gt, ssim_lambda=0.2)
        ssl_term, _ = bank(step=step, rendered=out, gt_rgb=gt,
                           camera=None, pose_pool=[], teacher=None,
                           gaussians=g, renderer=renderer)
        (loss + ssl_term).backward()
        for opt in opts.values():
            opt.step(); opt.zero_grad(set_to_none=True)
        strategy.step_post_backward(g.params, opts, state, step, out["info"], packed=False)
        print(f"[smoke] step {step}: loss={loss.item():.4f}  rgb_mean={out['rgb'].mean().item():.3f}  N={g.num_points}")

    dt = time.time() - t0
    print(f"[smoke] OK ({dt:.1f}s for 5 steps incl. first-call CUDA compile)")


if __name__ == "__main__":
    main()
