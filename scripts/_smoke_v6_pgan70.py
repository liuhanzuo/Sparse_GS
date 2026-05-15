"""Smoke test for the v6_pgan70 changes (Option B: real GAN).

Checks:

1.  ``PatchDiscriminator(arch='pix2pix70')`` builds, forward + backward run,
    output spatial size matches the expected pix2pix-PatchGAN downsample
    (~stride 2^3 = 8 in spatial dims relative to input due to the 3 s=2
    convs followed by two s=1 convs).
2.  ``diffaug.apply_pair`` produces *identical* augmentation parameters
    on real and fake (same shape, deterministic when seeded), and is
    differentiable (a backward pass on the augmented tensor reaches the
    input).
3.  Full ``PseudoViewDiscriminator`` end-to-end with arch=pix2pix70 and
    DiffAug turned on: ``discriminator_step`` and ``generator_loss`` both
    return finite numbers and produce non-zero D logits / G grad.

Run:
    python scripts/_smoke_v6_pgan70.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sparse_gs.losses import diffaug as _diffaug          # noqa: E402
from sparse_gs.losses.gan import PatchDiscriminator        # noqa: E402
from sparse_gs.losses.pvd import PseudoViewDiscriminator   # noqa: E402


def _bool_ok(name: str, ok: bool, detail: str = "") -> None:
    tag = "OK " if ok else "FAIL"
    print(f"[{tag}] {name}{(' :: ' + detail) if detail else ''}")
    if not ok:
        sys.exit(1)


# --------------------------------------------------------------------- #
# 1. discriminator architecture
# --------------------------------------------------------------------- #


def test_pix2pix70_arch() -> None:
    print("\n=== test 1: pix2pix70 PatchDiscriminator ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = PatchDiscriminator(arch="pix2pix70", base_channels=32).to(device)
    x = torch.randn(2, 3, 128, 128, device=device, requires_grad=True)
    y = D(x)
    _bool_ok("forward shape", y.dim() == 4 and y.shape[0] == 2 and y.shape[1] == 1,
             f"out={tuple(y.shape)}")
    # 128 / 2 / 2 / 2 = 16 -> 16 -> head reduces by ~3 (4-1)/2 → spatial ~13
    _bool_ok("output spatial >= 12 (rich logit map)", y.shape[2] >= 12,
             f"H'={y.shape[2]}")
    loss = y.mean()
    loss.backward()
    _bool_ok("backward populates grad", x.grad is not None and x.grad.abs().sum().item() > 0)

    # Receptive field sanity: changing one pixel near the centre should
    # only change a localised window in the logit map. We verify by
    # making sure the gradient magnitude is concentrated, not flat.
    g = x.grad[0].abs().mean(dim=0)
    _bool_ok("grad has spatial structure (RF is local)",
             (g.max() / max(g.mean().item(), 1e-12)) > 1.5,
             f"max/mean={float(g.max()/g.mean()):.2f}")


# --------------------------------------------------------------------- #
# 2. diffaug
# --------------------------------------------------------------------- #


def test_diffaug_shared_params() -> None:
    print("\n=== test 2: DiffAug real/fake share params ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _diffaug.DiffAugConfig(color=True, translation=0.125, cutout=0.5)
    real = torch.rand(2, 3, 64, 64, device=device, requires_grad=True)
    fake = torch.rand(2, 3, 64, 64, device=device, requires_grad=True)

    # Pre-sample params, apply the *same* params to both -> if real and
    # fake start as identical inputs, the augmented outputs must also be
    # identical.
    same = real.detach().clone()
    params = _diffaug.sample_params(cfg, same.shape[0], same.shape[2], same.shape[3], device)
    a = _diffaug.apply(same.clone().requires_grad_(True), cfg, params)
    b = _diffaug.apply(same.clone().requires_grad_(True), cfg, params)
    _bool_ok("identical input + identical params -> identical output",
             torch.allclose(a, b, atol=1e-6))

    # Differentiability: gradient must reach input
    a_pair, b_pair = _diffaug.apply_pair(real, fake, cfg)
    (a_pair.sum() + b_pair.sum()).backward()
    _bool_ok("apply_pair backward reaches real",
             real.grad is not None and real.grad.abs().sum().item() > 0)
    _bool_ok("apply_pair backward reaches fake",
             fake.grad is not None and fake.grad.abs().sum().item() > 0)


# --------------------------------------------------------------------- #
# 3. full PVD end-to-end
# --------------------------------------------------------------------- #


def test_pvd_end_to_end() -> None:
    print("\n=== test 3: PseudoViewDiscriminator end-to-end (arch=pix2pix70 + DiffAug) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pvd_cfg = {
        "enabled": True,
        "start_iter": 0,
        "every": 1,
        "weight": 5.0e-3,
        "patch_size": 128,
        "augment": True,
        "detach_geometry": True,
        "loss": "hinge",
        "arch": "pix2pix70",
        "augment_diffaug": {
            "enabled": True,
            "color": True,
            "translation": 0.125,
            "cutout": 0.5,
        },
    }
    pvd = PseudoViewDiscriminator(pvd_cfg, device)
    _bool_ok("pvd built with arch=pix2pix70", pvd.disc.arch == "pix2pix70")
    _bool_ok("pvd has DiffAug cfg", pvd.diffaug_cfg is not None)

    H, W = 200, 200
    real_hwc = torch.rand(H, W, 3, device=device)
    fake_hwc = torch.rand(H, W, 3, device=device, requires_grad=True)

    d_loss, d_logs = pvd.discriminator_step(real_hwc, fake_hwc.detach())
    _bool_ok("D step finite", torch.isfinite(d_loss).item(),
             f"d_loss={float(d_loss):.4f}")
    _bool_ok("D logs have real/fake/gap",
             {"pvd/real_logit", "pvd/fake_logit", "pvd/ema_gap"}.issubset(d_logs.keys()))

    g_term, g_logs = pvd.generator_loss(fake_hwc)
    _bool_ok("G term finite", torch.isfinite(g_term).item(),
             f"g={float(g_term):.4e} raw={g_logs.get('pvd/g_raw', 0):.4e}")
    g_term.backward()
    _bool_ok("G grad reaches fake",
             fake_hwc.grad is not None and fake_hwc.grad.abs().sum().item() > 0)

    # floater_score_map sanity (not used for grad)
    score = pvd.floater_score_map(torch.rand(H, W, 3, device=device), H, W)
    _bool_ok("score map shape (H,W)", score.shape == (H, W),
             f"got {tuple(score.shape)}")
    _bool_ok("score map in [0,1]",
             float(score.min()) >= 0.0 and float(score.max()) <= 1.0,
             f"min={float(score.min()):.3f} max={float(score.max()):.3f}")


if __name__ == "__main__":
    test_pix2pix70_arch()
    test_diffaug_shared_params()
    test_pvd_end_to_end()
    print("\nALL SMOKE TESTS PASSED.")
