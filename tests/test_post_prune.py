"""Unit tests for sparse_gs.strategies.post_prune.

Covers:
  - compute_unseen_mask:
      * all-seen case  -> all False
      * partially seen -> only unseen indices are True
  - compute_floater_mask:
      * picks ~thresh_bin fraction of pixels and back-projects via means2d
        on a synthetic 64x64 view (packed=True)
  - apply_safety_cap:
      * mask above max_ratio -> empty + over_aggressive=True
      * mask below max_ratio -> passthrough + over_aggressive=False

Run with: ``pytest tests/test_post_prune.py -q``

No GPU required; no gsplat dependency; runs on CPU.
"""

from __future__ import annotations

import torch

from sparse_gs.strategies.post_prune import (
    apply_safety_cap,
    compute_floater_mask,
    compute_unseen_mask,
)


# ---------------------------------------------------------------------------
# compute_unseen_mask
# ---------------------------------------------------------------------------
def test_unseen_mask_all_seen() -> None:
    n = 100
    visibility_count = torch.full((n,), 5, dtype=torch.long)  # every gauss seen 5 times
    mask = compute_unseen_mask(visibility_count)
    assert mask.dtype == torch.bool
    assert mask.shape == (n,)
    assert int(mask.sum().item()) == 0, "no Gaussian should be flagged unseen"


def test_unseen_mask_partial() -> None:
    n = 100
    visibility_count = torch.zeros(n, dtype=torch.long)
    # last 10 Gaussians ARE seen at least once; first 90 are unseen
    visibility_count[-10:] = 1
    mask = compute_unseen_mask(visibility_count)
    assert int(mask.sum().item()) == 90
    assert mask[:90].all().item()
    assert (~mask[-10:]).all().item()


# ---------------------------------------------------------------------------
# apply_safety_cap
# ---------------------------------------------------------------------------
def test_safety_cap_blocks_overaggressive() -> None:
    n = 1000
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:600] = True  # 60% prune -> WAY over 5%
    capped, over = apply_safety_cap(mask, max_ratio=0.05)
    assert over is True
    assert int(capped.sum().item()) == 0


def test_safety_cap_passes_through() -> None:
    n = 1000
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:30] = True  # 3% prune -> below 5% cap
    capped, over = apply_safety_cap(mask, max_ratio=0.05)
    assert over is False
    assert int(capped.sum().item()) == 30


# ---------------------------------------------------------------------------
# compute_floater_mask (packed=True path)
# ---------------------------------------------------------------------------
def test_floater_mask_basic_packed_true() -> None:
    """Synthetic test:
        - 64x64 depth ramp (top rows = small depth = "near"/floater zone)
        - alpha = 1 everywhere (no alpha gating)
        - 4096 Gaussians, one per pixel; means2d in packed order maps each
          gaussian to its own pixel
        - thresh_bin = 0.13 -> the bottom 13% of normalized depth values
          are flagged as floater pixels; back-projection picks exactly the
          Gaussians whose means2d sit in those pixels.
    """
    torch.manual_seed(0)
    H, W = 64, 64
    N = H * W

    # depth ramp: row 0 = 0.0 (smallest -> floater region), row 63 = 1.0 (far)
    depth = torch.linspace(0, 1, H).reshape(H, 1).repeat(1, W)
    alpha = torch.ones(H, W)

    # pixel coords (xy = (col, row)) -> means2d has shape (P, 2)
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    means2d = torch.stack([xs.flatten().float(), ys.flatten().float()], dim=1)

    # gaussian_ids: in packed mode it's a global index. Use a permutation to
    # make sure the function honors the index-mapping (not just identity).
    perm = torch.randperm(N)
    means2d_perm = means2d[perm]
    gaussian_ids = perm.clone()  # global index for each contribution

    mask = compute_floater_mask(
        depth=depth,
        alpha=alpha,
        n_gaussians=N,
        gaussian_ids=gaussian_ids,
        means2d=means2d_perm,
        alpha_thresh=0.5,
        thresh_bin=0.13,
        packed=True,
    )
    # Expected: ~13% of N flagged.
    n_flag = int(mask.sum().item())
    assert mask.dtype == torch.bool
    assert mask.shape == (N,)
    # We allow [10%, 16%] tolerance because torch.quantile interpolates.
    assert 0.10 * N <= n_flag <= 0.16 * N, f"got n_flag={n_flag}, expected ~{0.13 * N}"


def test_floater_mask_alpha_gating() -> None:
    """If alpha < threshold everywhere, no floater pixel is detected."""
    H, W = 32, 32
    N = H * W
    depth = torch.rand(H, W)
    alpha = torch.full((H, W), 0.1)  # below default 0.5 threshold

    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    means2d = torch.stack([xs.flatten().float(), ys.flatten().float()], dim=1)
    gaussian_ids = torch.arange(N)

    mask = compute_floater_mask(
        depth=depth,
        alpha=alpha,
        n_gaussians=N,
        gaussian_ids=gaussian_ids,
        means2d=means2d,
        alpha_thresh=0.5,
        thresh_bin=0.13,
        packed=True,
    )
    assert int(mask.sum().item()) == 0


def test_floater_mask_no_means2d_returns_empty() -> None:
    """Without means2d we cannot back-project, so we conservatively return empty."""
    H, W = 16, 16
    N = 1000
    depth = torch.rand(H, W)
    alpha = torch.ones(H, W)

    mask = compute_floater_mask(
        depth=depth,
        alpha=alpha,
        n_gaussians=N,
        gaussian_ids=torch.arange(min(N, H * W)),
        means2d=None,
        alpha_thresh=0.5,
        thresh_bin=0.13,
        packed=True,
    )
    assert mask.shape == (N,)
    assert int(mask.sum().item()) == 0


def test_floater_mask_packed_false_with_batched_radii() -> None:
    """gsplat 1.5.3 packed=False returns radii of shape (B, N, 2) and means2d
    of shape (B, N, 2). Verify our shape adapter handles this."""
    H, W = 32, 32
    N = 100
    # Make first 50 Gaussians visible (radii>0) and project them to row 0
    # (i.e. into the floater band when depth ramps from 0..1 over rows).
    radii = torch.zeros(1, N, 2, dtype=torch.int32)
    radii[0, :50, :] = 5  # visible
    means2d = torch.zeros(1, N, 2, dtype=torch.float32)
    # First 50 Gaussians sit in row 0 (floater band), columns 0..49.
    means2d[0, :50, 0] = torch.arange(50, dtype=torch.float32)  # x = col
    means2d[0, :50, 1] = 0                                       # y = row 0
    # depth ramp: row 0 = 0 (floater); row 31 = 1 (far). thresh_bin=0.13 picks
    # roughly the bottom 4 rows -> row 0 is definitely a floater pixel.
    depth = torch.linspace(0, 1, H).reshape(H, 1).repeat(1, W)
    alpha = torch.ones(H, W)

    mask = compute_floater_mask(
        depth=depth,
        alpha=alpha,
        n_gaussians=N,
        gaussian_ids=None,
        radii=radii,
        means2d=means2d,
        alpha_thresh=0.5,
        thresh_bin=0.13,
        packed=False,
    )
    # All 50 visible Gaussians sit on row 0 -> all flagged.
    assert mask.shape == (N,)
    assert int(mask[:50].sum().item()) == 50
    assert int(mask[50:].sum().item()) == 0