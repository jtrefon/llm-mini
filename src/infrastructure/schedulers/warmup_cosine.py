"""Warmup cosine learning rate scheduler - Infrastructure layer implementation."""
import math
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineScheduler:
    """Warmup + cosine decay learning rate scheduler.

    Implements a learning rate schedule that:
    1. Linear warmup from 0 to base_lr over warmup_steps
    2. Cosine decay from base_lr to 0 over remaining steps

    This is a framework-specific implementation that belongs in the
    infrastructure layer, not in the domain layer.
    """

    def __init__(self, optimizer, warmup_ratio: float, max_steps: int):
        """Initialize the scheduler.

        Args:
            optimizer: PyTorch optimizer to schedule
            warmup_ratio: Fraction of training for warmup (0.0 to 1.0)
            max_steps: Total number of training steps
        """
        # Keep external reference as-is (may be a Mock)
        self.optimizer = optimizer
        self.warmup_ratio = warmup_ratio
        self.max_steps = max_steps

        self.warmup_steps = max(1, int(warmup_ratio * max_steps))

        def lr_lambda(step: int) -> float:
            """Learning rate lambda function."""
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Cosine decay
                progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))

        # Attach scheduler directly to the provided optimizer so Lightning can validate the pairing
        self.scheduler = LambdaLR(optimizer, lr_lambda)

    @property
    def base_lrs(self):  # pragma: no cover - passthrough to underlying scheduler
        return self.scheduler.base_lrs

    def step(self):
        """Update learning rate for next step."""
        self.scheduler.step()

    def get_last_lr(self):
        """Get current learning rate values."""
        return self.scheduler.get_last_lr()

    def state_dict(self):
        """Get scheduler state for checkpointing."""
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.scheduler.load_state_dict(state_dict)

    @property
    def base_lrs(self):
        """Get base learning rates."""
        return self.scheduler.base_lrs

    @property
    def last_epoch(self):
        """Get last epoch/step processed."""
        return self.scheduler.last_epoch
