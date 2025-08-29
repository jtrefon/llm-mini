"""Tests for shared factory functions."""
import pytest
import torch
from unittest.mock import Mock

from src.shared.factories.optimizer_factory import create_adamw_optimizer
from src.shared.factories.scheduler_factory import create_warmup_cosine_scheduler
from src.domain.entities.training_config import TrainingConfiguration


class TestOptimizerFactory:
    """Test cases for optimizer factory."""

    def test_create_adamw_optimizer(self, sample_training_config):
        """Test AdamW optimizer creation."""
        mock_model = Mock()
        mock_parameters = [Mock()]
        mock_model.parameters.return_value = mock_parameters

        optimizer = create_adamw_optimizer(mock_model.parameters(), sample_training_config)

        # Verify optimizer was created
        assert optimizer is not None

        # Verify optimizer has expected attributes (if available)
        if hasattr(optimizer, 'param_groups'):
            assert len(optimizer.param_groups) > 0

    def test_create_adamw_optimizer_with_different_configs(self):
        """Test optimizer creation with different configurations."""
        # Create different training configs
        config1 = TrainingConfiguration(
            seq_len=512, micro_batch_size=4, grad_accum_steps=8,
            max_steps=1000, eval_every=100, save_every=200,
            lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
            eps=1e-8, warmup_ratio=0.1, precision="32",
            seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
            limit_val_batches=1.0
        )

        config2 = TrainingConfiguration(
            seq_len=512, micro_batch_size=4, grad_accum_steps=8,
            max_steps=1000, eval_every=100, save_every=200,
            lr=5e-4, weight_decay=0.01, betas=[0.8, 0.99],  # Different values
            eps=1e-7, warmup_ratio=0.1, precision="32",
            seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
            limit_val_batches=1.0
        )

        mock_model = Mock()
        mock_parameters = [Mock()]
        mock_model.parameters.return_value = mock_parameters

        optimizer1 = create_adamw_optimizer(mock_model.parameters(), config1)
        optimizer2 = create_adamw_optimizer(mock_model.parameters(), config2)

        # Both should be created successfully
        assert optimizer1 is not None
        assert optimizer2 is not None
        assert optimizer1 is not optimizer2  # Different instances

    def test_optimizer_factory_consistency(self, sample_training_config):
        """Test that factory produces consistent results."""
        mock_model = Mock()
        mock_parameters = [Mock()]
        mock_model.parameters.return_value = mock_parameters

        # Create multiple optimizers with same config
        optimizer1 = create_adamw_optimizer(mock_model.parameters(), sample_training_config)
        optimizer2 = create_adamw_optimizer(mock_model.parameters(), sample_training_config)

        # Should be different instances but configured the same
        assert optimizer1 is not optimizer2
        assert optimizer1 is not None
        assert optimizer2 is not None


class TestSchedulerFactory:
    """Test cases for scheduler factory."""

    def test_create_warmup_cosine_scheduler(self, sample_training_config):
        """Test warmup cosine scheduler creation."""
        mock_optimizer = Mock()

        scheduler = create_warmup_cosine_scheduler(mock_optimizer, sample_training_config)

        # Verify scheduler was created
        assert scheduler is not None

        # Verify scheduler has expected attributes
        assert hasattr(scheduler, 'step')
        assert hasattr(scheduler, 'get_last_lr')

    def test_scheduler_creation_with_different_configs(self):
        """Test scheduler creation with different configurations."""
        # Create different training configs
        config1 = TrainingConfiguration(
            seq_len=512, micro_batch_size=4, grad_accum_steps=8,
            max_steps=1000, eval_every=100, save_every=200,
            lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
            eps=1e-8, warmup_ratio=0.1, precision="32",
            seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
            limit_val_batches=1.0
        )

        config2 = TrainingConfiguration(
            seq_len=512, micro_batch_size=4, grad_accum_steps=8,
            max_steps=2000, eval_every=100, save_every=200,  # Different max_steps
            lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
            eps=1e-8, warmup_ratio=0.05, precision="32",  # Different warmup_ratio
            seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
            limit_val_batches=1.0
        )

        mock_optimizer = Mock()

        scheduler1 = create_warmup_cosine_scheduler(mock_optimizer, config1)
        scheduler2 = create_warmup_cosine_scheduler(mock_optimizer, config2)

        # Both should be created successfully
        assert scheduler1 is not None
        assert scheduler2 is not None
        assert scheduler1 is not scheduler2  # Different instances

    def test_scheduler_warmup_calculation(self, sample_training_config):
        """Test that scheduler correctly calculates warmup steps."""
        mock_optimizer = Mock()

        # Config with 10% warmup ratio and 1000 max steps
        config = TrainingConfiguration(
            seq_len=512, micro_batch_size=4, grad_accum_steps=8,
            max_steps=1000, eval_every=100, save_every=200,
            lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
            eps=1e-8, warmup_ratio=0.1, precision="32",
            seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
            limit_val_batches=1.0
        )

        scheduler = create_warmup_cosine_scheduler(mock_optimizer, config)

        # Should have 100 warmup steps (0.1 * 1000)
        assert scheduler.warmup_steps == 100
        assert scheduler.max_steps == 1000

    def test_scheduler_edge_cases(self):
        """Test scheduler creation with edge case configurations."""
        mock_optimizer = Mock()

        # Test with zero warmup
        config_zero_warmup = TrainingConfiguration(
            seq_len=512, micro_batch_size=4, grad_accum_steps=8,
            max_steps=1000, eval_every=100, save_every=200,
            lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
            eps=1e-8, warmup_ratio=0.0, precision="32",
            seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
            limit_val_batches=1.0
        )

        scheduler_zero = create_warmup_cosine_scheduler(mock_optimizer, config_zero_warmup)
        assert scheduler_zero.warmup_steps == 1  # max(1, 0 * 1000)

        # Test with full warmup
        config_full_warmup = TrainingConfiguration(
            seq_len=512, micro_batch_size=4, grad_accum_steps=8,
            max_steps=1000, eval_every=100, save_every=200,
            lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
            eps=1e-8, warmup_ratio=1.0, precision="32",
            seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
            limit_val_batches=1.0
        )

        scheduler_full = create_warmup_cosine_scheduler(mock_optimizer, config_full_warmup)
        assert scheduler_full.warmup_steps == 1000  # 1.0 * 1000

    def test_scheduler_lr_lambda_function(self, sample_training_config):
        """Test the LR lambda function behavior."""
        mock_optimizer = Mock()
        scheduler = create_warmup_cosine_scheduler(mock_optimizer, sample_training_config)

        # Test warmup phase (step < warmup_steps)
        lr_warmup = scheduler.scheduler.lr_lambdas[0](50)  # Middle of warmup
        expected_warmup = 50 / 100  # step / warmup_steps
        assert abs(lr_warmup - expected_warmup) < 1e-6

        # Test cosine decay phase
        lr_decay = scheduler.scheduler.lr_lambdas[0](550)  # Middle of cosine decay
        # Should be 0.5 * (1 + cos(π * progress))
        progress = (550 - 100) / (1000 - 100)  # (step - warmup_steps) / (max_steps - warmup_steps)
        expected_decay = 0.5 * (1 + math.cos(math.pi * progress))
        assert abs(lr_decay - expected_decay) < 1e-5

    def test_scheduler_consistency(self, sample_training_config):
        """Test that factory produces consistent schedulers."""
        mock_optimizer1 = Mock()
        mock_optimizer2 = Mock()

        # Create multiple schedulers with same config
        scheduler1 = create_warmup_cosine_scheduler(mock_optimizer1, sample_training_config)
        scheduler2 = create_warmup_cosine_scheduler(mock_optimizer2, sample_training_config)

        # Should be different instances but with same configuration
        assert scheduler1 is not scheduler2
        assert scheduler1.warmup_steps == scheduler2.warmup_steps
        assert scheduler1.max_steps == scheduler2.max_steps


# Import math for cosine calculations in tests
import math
