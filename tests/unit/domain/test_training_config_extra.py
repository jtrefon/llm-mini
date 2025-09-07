"""Extra tests to increase coverage for TrainingConfiguration validation."""
import pytest

from src.domain.entities.training_config import TrainingConfiguration


def _base_kwargs():
    return dict(
        seq_len=8, micro_batch_size=1, grad_accum_steps=1,
        max_steps=2, eval_every=1, save_every=0,
        lr=1e-3, weight_decay=0.0, betas=[0.9, 0.95], eps=1e-8,
        warmup_ratio=0.01, precision='32', seed=42,
        steps_per_epoch=None, gradient_clip_val=None, limit_val_batches=1.0
    )


def test_training_config_max_steps_positive_validation():
    with pytest.raises(ValueError):
        TrainingConfiguration(**{**_base_kwargs(), 'max_steps': 0})


def test_training_config_eps_positive_validation():
    with pytest.raises(ValueError):
        TrainingConfiguration(**{**_base_kwargs(), 'eps': 0})


def test_training_config_limit_val_batches_bounds():
    # Lower bound
    with pytest.raises(ValueError):
        TrainingConfiguration(**{**_base_kwargs(), 'limit_val_batches': -0.1})
    # Upper bound
    with pytest.raises(ValueError):
        TrainingConfiguration(**{**_base_kwargs(), 'limit_val_batches': 1.1})
