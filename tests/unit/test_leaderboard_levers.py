"""CPU unit tests for ADR-0038 leaderboard levers."""

from __future__ import annotations

import math

import torch

from psalm.infrastructure.ml.leaderboard_levers import (
    Muon,
    build_submission_optimizers,
    make_freq_informed_mlm_mask,
    progressive_seq_len,
    scheduled_mask_prob,
    split_params_for_muon,
)


def test_scheduled_mask_prob_endpoints_and_monotone() -> None:
    total = 100
    assert math.isclose(scheduled_mask_prob(0, total, p_start=0.3, p_end=0.15), 0.3, abs_tol=1e-6)
    assert math.isclose(
        scheduled_mask_prob(total - 1, total, p_start=0.3, p_end=0.15), 0.15, abs_tol=1e-6
    )
    prev = 1.0
    for s in range(total):
        cur = scheduled_mask_prob(s, total, p_start=0.3, p_end=0.15)
        assert cur <= prev + 1e-9  # non-increasing
        prev = cur
    # constant kind ignores progress
    assert scheduled_mask_prob(50, total, p_start=0.3, kind="constant") == 0.3


def test_progressive_seq_len_steps_up() -> None:
    sched = [(0.0, 64), (0.5, 128), (0.8, 256)]
    total = 100
    assert progressive_seq_len(0, total, sched) == 64
    assert progressive_seq_len(60, total, sched) == 128
    assert progressive_seq_len(total - 1, total, sched) == 256


def test_freq_informed_mask_preserves_expected_rate() -> None:
    torch.manual_seed(0)
    vocab = 200
    # Zipf-ish log frequencies.
    log_freq = -torch.log(torch.arange(1, vocab + 1, dtype=torch.float32))
    idx = torch.randint(0, vocab, (64, 128))
    prob = 0.25
    rates = []
    for _ in range(20):
        _, loss_mask = make_freq_informed_mlm_mask(
            idx, mask_id=vocab - 1, probability=prob, token_log_freq=log_freq, alpha=0.7
        )
        rates.append(loss_mask.float().mean().item())
    mean_rate = sum(rates) / len(rates)
    # Expected masked fraction stays close to `prob` despite frequency weighting.
    assert abs(mean_rate - prob) < 0.03


def test_freq_informed_alpha_zero_matches_uniform_rate() -> None:
    torch.manual_seed(1)
    vocab = 100
    log_freq = torch.zeros(vocab)
    idx = torch.randint(0, vocab - 1, (32, 64))
    _, loss_mask = make_freq_informed_mlm_mask(
        idx, mask_id=vocab - 1, probability=0.3, token_log_freq=log_freq, alpha=0.0
    )
    assert abs(loss_mask.float().mean().item() - 0.3) < 0.05


def test_muon_reduces_quadratic_loss() -> None:
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 6))
    target = torch.randn(8, 6)
    opt = Muon([w], lr=0.05, momentum=0.9)
    first = None
    for _ in range(50):
        opt.zero_grad()
        loss = ((w - target) ** 2).mean()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first


def test_split_params_excludes_embeddings_and_head() -> None:
    model = torch.nn.Module()
    model.embed = torch.nn.Embedding(50, 16)
    model.linear = torch.nn.Linear(16, 16, bias=True)
    model.lm_head = torch.nn.Linear(16, 50, bias=False)
    muon, rest = split_params_for_muon(model)
    # The hidden linear weight is the only Muon matrix.
    assert any(p.shape == (16, 16) for p in muon)
    assert all(p.shape != (50, 16) for p in muon)  # embedding excluded
    assert all(p.shape != (50, 16) or p.ndim != 2 for p in muon)  # head excluded
    # Embedding weight, linear bias, and head weight all land in rest.
    assert len(rest) >= 3


def test_build_submission_optimizers_steps() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    opts = build_submission_optimizers(model, muon_lr=0.02, adamw_lr=1e-3)
    assert len(opts) >= 1
    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))
    before = torch.nn.functional.cross_entropy(model(x), y).item()
    for _ in range(30):
        for o in opts:
            o.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        for o in opts:
            o.step()
    assert loss.item() < before
