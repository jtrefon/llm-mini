"""Extra tests to bump coverage for WarmupCosineScheduler."""
from unittest.mock import Mock

from src.infrastructure.schedulers.warmup_cosine import WarmupCosineScheduler


def test_last_epoch_property_access():
    # Mock optimizer with a param_groups lr to seed base_lrs path
    opt = Mock()
    opt.param_groups = [{'lr': 1e-3}]
    sched = WarmupCosineScheduler(opt, warmup_ratio=0.1, max_steps=10)
    # Access last_epoch to cover property
    le = sched.last_epoch
    assert isinstance(le, int)
