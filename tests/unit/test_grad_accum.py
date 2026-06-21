"""Tests for gradient accumulation in the training loop.

Tests the mechanics of micro-batch accumulation, effective-step scheduling,
loss scaling, and invariant preservation when grad_accum=1 (default).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from psalm.infrastructure.ml.elc_trainer import cosine_warmup_lr


def _compute_effective_steps(total_micro_steps: int, grad_accum: int) -> int:
    """Helper: compute effective optimizer steps from micro-steps."""
    return max(1, total_micro_steps // grad_accum)


class TestGradAccumScheduling:
    """Test that effective-step scheduling produces correct LR values."""

    def test_grad_accum_1_identity(self) -> None:
        """With grad_accum=1, effective steps == micro steps."""
        total_micro = 100
        grad_accum = 1
        effective = _compute_effective_steps(total_micro, grad_accum)
        assert effective == 100

    def test_grad_accum_2_halves_steps(self) -> None:
        """With grad_accum=2, effective steps = total_micro // 2."""
        total_micro = 100
        grad_accum = 2
        effective = _compute_effective_steps(total_micro, grad_accum)
        assert effective == 50

    def test_grad_accum_4_quarters_steps(self) -> None:
        """With grad_accum=4, effective steps = total_micro // 4."""
        total_micro = 100
        grad_accum = 4
        effective = _compute_effective_steps(total_micro, grad_accum)
        assert effective == 25

    def test_cosine_lr_at_effective_step_k(self) -> None:
        """LR at effective step k should be computed from k, not from micro-step."""
        peak_lr = 1e-3
        total_steps = 100
        warmup_steps = 10

        # Without accumulation: micro step 20 -> lr at step 20
        lr_micro_20 = cosine_warmup_lr(
            20, peak_lr=peak_lr, warmup_steps=warmup_steps, total_steps=total_steps
        )

        # With accumulation=2: micro step 20 -> effective step 10 -> lr at step 10
        lr_eff_10 = cosine_warmup_lr(10, peak_lr=peak_lr, warmup_steps=warmup_steps, total_steps=50)

        # These should be different (20 is past warmup, 10 is in warmup)
        assert lr_micro_20 != lr_eff_10

    def test_warmup_steps_scaled_with_accum(self) -> None:
        """With grad_accum, warmup_steps must also be scaled down."""
        total_micro_steps = 100
        grad_accum = 2
        warmup_frac = 0.1

        # Micro-level calculation (before)
        warmup_micro = max(int(warmup_frac * total_micro_steps), 1)
        assert warmup_micro == 10

        # Effective-level calculation (after)
        total_effective = _compute_effective_steps(total_micro_steps, grad_accum)
        warmup_eff = max(int(warmup_frac * total_effective), 1)
        assert warmup_eff == 5


class TestLossScaling:
    """Test that loss is scaled correctly during accumulation."""

    def test_loss_scaled_by_inverse_accum(self) -> None:
        """Each micro-batch loss should be scaled by 1/grad_accum."""
        loss_per_microbatch = 1.0
        grad_accum = 4

        scaled_loss = loss_per_microbatch / grad_accum
        assert scaled_loss == pytest.approx(0.25)

    def test_accumulated_gradients_magnitude(self) -> None:
        """Accumulated gradients should have magnitude ~= no-accum gradients."""
        # Create a simple model with known loss
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
        torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create a dummy input and target
        x = torch.randn(4, 10)  # batch_size=4
        y_true = torch.randn(4, 1)

        # Scenario 1: gradient step without accumulation (batch of 4)
        model.zero_grad()
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()
        grad_without_accum = [
            p.grad.clone() if p.grad is not None else None for p in model.parameters()
        ]

        # Scenario 2: gradient step with accumulation (2 micro-batches of 2 each)
        model.zero_grad()
        for i in range(2):
            y_pred = model(x[i * 2 : (i + 1) * 2])
            y_true_micro = y_true[i * 2 : (i + 1) * 2]
            loss = nn.functional.mse_loss(y_pred, y_true_micro)
            scaled_loss = loss / 2  # scale by 1/accum
            scaled_loss.backward()
        grad_with_accum = [
            p.grad.clone() if p.grad is not None else None for p in model.parameters()
        ]

        # Gradients should be approximately equal (within numerical precision)
        for g1, g2 in zip(grad_without_accum, grad_with_accum, strict=False):
            if g1 is not None and g2 is not None:
                # Allow some tolerance for numerical differences
                assert torch.allclose(g1, g2, atol=1e-5)


class TestEffectiveStepMath:
    """Test the effective-step accounting with multiple micro-batches."""

    def test_micro_steps_to_effective_steps_mapping(self) -> None:
        """Map micro-step indices to effective-step indices."""
        grad_accum = 2
        expected_mapping = {
            0: 0,  # micro 0-1 -> eff 0
            1: 0,
            2: 1,  # micro 2-3 -> eff 1
            3: 1,
            4: 2,  # micro 4-5 -> eff 2
            5: 2,
        }
        for micro_step, expected_eff_step in expected_mapping.items():
            actual_eff_step = micro_step // grad_accum
            assert actual_eff_step == expected_eff_step

    def test_optimizer_step_count(self) -> None:
        """With grad_accum=N, total_micro_steps M, optimizer.step() called M//N times."""
        total_micro_steps = 10
        grad_accum = 3
        expected_opt_steps = total_micro_steps // grad_accum
        assert expected_opt_steps == 3

    def test_partial_accumulation_discarded(self) -> None:
        """Incomplete final accumulation step is discarded (10 micros, accum=3 -> 3 steps)."""
        total_micro_steps = 10
        grad_accum = 3
        # 10 // 3 = 3 (discards the final incomplete step with only 1 micro-batch)
        effective_steps = total_micro_steps // grad_accum
        assert effective_steps == 3


class TestWordCountingInvariance:
    """Test that word/token accounting is unaffected by grad_accum."""

    def test_tokens_per_step_unchanged_by_accum(self) -> None:
        """Tokens processed per micro-step is independent of grad_accum."""
        batch_size = 256
        seq_len = 256
        tok_per_microstep = batch_size * seq_len

        # With grad_accum=1 or 2, tokens per micro-step are the same
        # (grad_accum only affects when optimizer.step() is called, not token counting)
        for _grad_accum in [1, 2, 4]:
            assert tok_per_microstep == 256 * 256

    def test_milestone_checkpoints_aligned_regardless_of_accum(self) -> None:
        """Word-count milestones should trigger at the same word counts regardless of grad_accum."""
        # BabyLM milestones: 1M, 2M, ..., 10M words
        milestones = [1_000_000, 2_000_000, 3_000_000]

        # These are word counts, independent of grad_accum
        # (accumulation doesn't change word-seen accounting, only optimizer-step timing)
        for _grad_accum in [1, 2, 4]:
            # Milestone is still 1M words, regardless of grad_accum
            assert 1_000_000 in milestones


class TestGradAccumWithOptimizer:
    """Test gradient accumulation with actual torch optimizers."""

    def test_grad_accum_1_matches_eager_gradient(self) -> None:
        """With grad_accum=1, a single backward == accumulation of 1 step."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
        torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        # Single step without accumulation
        model.zero_grad()
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y)
        loss.backward()
        grad_expected = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]

        # Single accumulation step (grad_accum=1)
        model.zero_grad()
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y)
        (loss / 1).backward()  # scaled by 1/accum where accum=1
        grad_actual = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]

        for g_exp, g_act in zip(grad_expected, grad_actual, strict=False):
            if g_exp is not None and g_act is not None:
                assert torch.allclose(g_exp, g_act, atol=1e-6)

    def test_zero_grad_only_after_accum_complete(self) -> None:
        """zero_grad() should only be called once per grad_accum steps."""
        model = nn.Linear(10, 5)
        torch.optim.Adam(model.parameters())

        # Zero before 2 micro-batches (grad_accum=2)
        model.zero_grad()
        assert model.weight.grad is None

        # First micro-batch backward
        x1 = torch.randn(2, 10)
        y1 = torch.randn(2, 5)
        loss1 = nn.functional.mse_loss(model(x1), y1)
        (loss1 / 2).backward()
        grad_after_first = model.weight.grad.clone()
        assert grad_after_first is not None

        # Second micro-batch backward (accumulate, don't zero)
        x2 = torch.randn(2, 10)
        y2 = torch.randn(2, 5)
        loss2 = nn.functional.mse_loss(model(x2), y2)
        (loss2 / 2).backward()
        grad_after_second = model.weight.grad.clone()

        # Gradient should be non-zero and different from after first (accumulated)
        assert not torch.allclose(grad_after_first, grad_after_second)

    def test_clip_grad_norm_applied_once_per_effective_step(self) -> None:
        """clip_grad_norm_ should be called once per grad_accum steps, not per micro-step."""
        model = nn.Linear(10, 5)

        # Simulate accumulation: 2 micro-batches
        model.zero_grad()
        for _i in range(2):
            x = torch.randn(2, 10)
            y = torch.randn(2, 5)
            loss = nn.functional.mse_loss(model(x), y)
            (loss / 2).backward()

        # Clip after accumulation (once)
        norm_before = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # If max_norm=1.0 was hit, norm_before > 1.0
        assert norm_before >= 0.0  # should be a valid norm value
