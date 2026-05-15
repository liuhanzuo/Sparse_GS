import torch

from sparse_gs.losses.ssl import (
    pearson_depth_loss,
    pseudo_view_warmup_weight,
    should_sample_pseudo_view,
)

def test_pearson_depth_loss_same_depth_is_near_zero():
    prior = torch.linspace(0.1, 5.0, steps=64, dtype=torch.float32).reshape(8, 8)
    rend = prior

    loss = pearson_depth_loss(rend, prior)

    assert loss.item() < 1e-3


def test_pearson_depth_loss_reversed_depth_is_near_two():
    prior = torch.linspace(10.0, 30.0, steps=64, dtype=torch.float32).reshape(8, 8)
    rend = -prior

    loss = pearson_depth_loss(rend, prior)

    assert loss.item() > 1.99


def test_pearson_depth_loss_supports_batch_input():
    prior_a = torch.linspace(0.1, 5.0, steps=64, dtype=torch.float32).reshape(8, 8)
    prior_b = torch.linspace(5.0, 0.1, steps=64, dtype=torch.float32).reshape(8, 8)
    prior = torch.stack([prior_a, prior_b], dim=0)
    rend = prior

    loss = pearson_depth_loss(rend, prior)

    assert loss.ndim == 0
    assert loss.item() < 1e-3


def test_pearson_depth_loss_supports_legacy_fsgs_transform():
    prior = torch.linspace(0.1, 5.0, steps=64, dtype=torch.float32).reshape(8, 8)
    rend = -prior

    loss = pearson_depth_loss(rend, prior, target_transform="fsgs")

    assert loss.item() < 1e-3


def test_pearson_depth_loss_backward():
    prior = torch.linspace(0.1, 5.0, steps=64, dtype=torch.float32).reshape(8, 8)
    rend = (prior + 0.01 * torch.randn_like(prior)).detach().requires_grad_(True)

    loss = pearson_depth_loss(rend, prior)
    loss.backward()

    assert rend.grad is not None
    assert torch.isfinite(rend.grad).all()


def test_pseudo_view_warmup_matches_fsgs_schedule():
    start = 2000

    assert pseudo_view_warmup_weight(start, start) == 0.0
    assert pseudo_view_warmup_weight(start + 250, start) == 0.5
    assert pseudo_view_warmup_weight(start + 500, start) == 1.0
    assert pseudo_view_warmup_weight(start + 1000, start) == 1.0


def test_should_sample_pseudo_view_respects_bounds_and_interval():
    start = 2000
    end = 9500
    interval = 10

    assert not should_sample_pseudo_view(start, start, end, interval)
    assert not should_sample_pseudo_view(end, start, end, interval)
    assert not should_sample_pseudo_view(start + 1, start, end, interval)
    assert should_sample_pseudo_view(start + 10, start, end, interval)
