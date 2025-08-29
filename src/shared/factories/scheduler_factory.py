"""Shared scheduler factory - eliminates duplication between training scripts."""
from typing import Any

from src.domain.entities.training_config import TrainingConfiguration
from src.infrastructure.schedulers.warmup_cosine import WarmupCosineScheduler


def create_warmup_cosine_scheduler(optimizer: Any, config: TrainingConfiguration) -> WarmupCosineScheduler:
    """Create warmup + cosine learning rate scheduler with standardized configuration.

    This factory function ensures consistent scheduler setup across all
    training scripts, eliminating code duplication and configuration drift.

    Args:
        optimizer: PyTorch optimizer to schedule
        config: Training configuration with scheduler hyperparameters

    Returns:
        Configured LambdaLR scheduler with warmup + cosine decay
    """
    return WarmupCosineScheduler(
        optimizer,
        warmup_ratio=config.warmup_ratio,
        max_steps=config.max_steps,
    )
