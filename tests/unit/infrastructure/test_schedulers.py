"""Tests for infrastructure schedulers."""
import pytest
import torch
import math
from unittest.mock import Mock

from src.infrastructure.schedulers.warmup_cosine import WarmupCosineScheduler


class TestWarmupCosineScheduler:
    """Test cases for WarmupCosineScheduler."""

    def test_scheduler_creation(self):
        """Test scheduler creation with valid parameters."""
        optimizer = Mock()
        scheduler = WarmupCosineScheduler(optimizer, warmup_ratio=0.1, max_steps=1000)

        assert scheduler.optimizer == optimizer
        assert scheduler.warmup_ratio == 0.1
        assert scheduler.max_steps == 1000
        assert scheduler.warmup_steps == 100  # 0.1 * 1000

    def test_warmup_phase_lr(self):
        """Test learning rate during warmup phase."""
        optimizer = Mock()
        optimizer.param_groups = [{'lr': 1e-3}]
        scheduler = WarmupCosineScheduler(optimizer, warmup_ratio=0.1, max_steps=1000)

        # At step 0 (start of warmup)
        lr_at_0 = scheduler.scheduler.get_last_lr()[0]
        assert lr_at_0 == 0.0  # Linear warmup starts at 0

        # At step 50 (middle of warmup)
        for _ in range(50):
            scheduler.step()
        lr_at_50 = scheduler.scheduler.get_last_lr()[0]
        assert abs(lr_at_50 - 5e-4) < 1e-6  # Should be 50% of base_lr

        # At step 100 (end of warmup)
        for _ in range(50):
            scheduler.step()
        lr_at_100 = scheduler.scheduler.get_last_lr()[0]
        assert abs(lr_at_100 - 1e-3) < 1e-6  # Should reach base_lr

    def test_cosine_decay_phase_lr(self):
        """Test learning rate during cosine decay phase."""
        optimizer = Mock()
        optimizer.param_groups = [{'lr': 1e-3}]
        scheduler = WarmupCosineScheduler(optimizer, warmup_ratio=0.1, max_steps=1000)

        # Skip warmup phase
        for _ in range(100):
            scheduler.step()

        # At step 550 (middle of cosine decay: warmup=100, decay half-way at 100 + 0.5*(1000-100) = 550)
        for _ in range(450):
            scheduler.step()
        lr_at_550 = scheduler.scheduler.get_last_lr()[0]

        # Cosine decay: 0.5 * (1 + cos(π * progress))
        # At 50% progress, cos(π * 0.5) = cos(π/2) = 0, so lr = 0.5 * base_lr
        expected_lr = 0.5 * 1e-3
        assert abs(lr_at_550 - expected_lr) < 1e-5

    def test_scheduler_step_method(self):
        """Test scheduler step method."""
        optimizer = Mock()
        scheduler = WarmupCosineScheduler(optimizer, warmup_ratio=0.1, max_steps=1000)

        # Step should call the underlying scheduler
        scheduler.step()
        assert scheduler.scheduler.last_epoch == 1

    def test_scheduler_state_dict(self):
        """Test scheduler state persistence."""
        optimizer = Mock()
        scheduler = WarmupCosineScheduler(optimizer, warmup_ratio=0.1, max_steps=1000)

        # Step a few times
        for _ in range(10):
            scheduler.step()

        # Get state
        state = scheduler.state_dict()
        assert 'last_epoch' in state

        # Create new scheduler and load state
        new_optimizer = Mock()
        new_scheduler = WarmupCosineScheduler(new_optimizer, warmup_ratio=0.1, max_steps=1000)
        new_scheduler.load_state_dict(state)

        # Should have same state
        assert new_scheduler.scheduler.last_epoch == scheduler.scheduler.last_epoch

    def test_scheduler_base_lrs(self):
        """Test base learning rates property."""
        optimizer = Mock()
        optimizer.param_groups = [{'lr': 1e-3}, {'lr': 2e-3}]
        scheduler = WarmupCosineScheduler(optimizer, warmup_ratio=0.1, max_steps=1000)

        base_lrs = scheduler.base_lrs
        assert len(base_lrs) == 2
        assert base_lrs[0] == 1e-3
        assert base_lrs[1] == 2e-3

    def test_scheduler_get_last_lr(self):
        """Test get_last_lr method."""
        optimizer = Mock()
        optimizer.param_groups = [{'lr': 1e-3}]
        scheduler = WarmupCosineScheduler(optimizer, warmup_ratio=0.1, max_steps=1000)

        lr_values = scheduler.get_last_lr()
        assert isinstance(lr_values, list)
        assert len(lr_values) == 1

    def test_edge_case_zero_warmup(self):
        """Test scheduler with zero warmup ratio."""
        optimizer = Mock()
        scheduler = WarmupCosineScheduler(optimizer, warmup_ratio=0.0, max_steps=1000)

        assert scheduler.warmup_steps == 1  # max(1, 0 * 1000)

    def test_edge_case_full_warmup(self):
        """Test scheduler with full warmup ratio."""
        optimizer = Mock()
        scheduler = WarmupCosineScheduler(optimizer, warmup_ratio=1.0, max_steps=1000)

        assert scheduler.warmup_steps == 1000  # 1.0 * 1000

    def test_cosine_lr_calculation(self):
        """Test cosine learning rate calculation manually."""
        optimizer = Mock()
        optimizer.param_groups = [{'lr': 1e-3}]
        scheduler = WarmupCosineScheduler(optimizer, warmup_ratio=0.1, max_steps=1000)

        # Test various points in cosine decay
        test_points = [
            (100, 0.0),    # End of warmup
            (550, 0.5),    # Halfway through cosine
            (1000, 0.0),   # End of cosine decay
        ]

        for step, expected_progress in test_points:
            # Reset scheduler
            scheduler = WarmupCosineScheduler(optimizer, warmup_ratio=0.1, max_steps=1000)

            # Step to target
            for _ in range(step):
                scheduler.step()

            lr = scheduler.get_last_lr()[0]

            if step <= 100:  # Warmup phase
                expected_lr = min(1.0, step / 100) * 1e-3
            else:  # Cosine decay phase
                progress = (step - 100) / 900  # (step - warmup_steps) / (max_steps - warmup_steps)
                expected_lr = 0.5 * (1 + math.cos(math.pi * progress)) * 1e-3

            assert abs(lr - expected_lr) < 1e-5, f"Step {step}: expected {expected_lr}, got {lr}"
