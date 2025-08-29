"""Training configuration entity - encapsulates all training hyperparameters."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class TrainingConfiguration:
    """Training hyperparameters with validation and type safety.

    This entity represents the complete configuration for training a transformer model,
    including optimization, learning rate scheduling, and hardware settings.

    Attributes:
        seq_len: Maximum sequence length for training examples
        micro_batch_size: Micro batch size (effective batch = micro * grad_accum * devices)
        grad_accum_steps: Gradient accumulation steps
        max_steps: Maximum number of training steps
        eval_every: Evaluate validation set every N steps (0 to disable)
        save_every: Save checkpoint every N steps (0 to disable)
        lr: Learning rate for optimization
        weight_decay: L2 regularization weight decay
        betas: Adam optimizer beta parameters [beta1, beta2]
        eps: Adam optimizer epsilon for numerical stability
        warmup_ratio: Fraction of training for learning rate warmup (0.0 to 1.0)
        precision: Mixed precision training precision ("16", "32", "bf16")
        seed: Random seed for reproducibility
        steps_per_epoch: Steps per epoch override (None for automatic)
        gradient_clip_val: Gradient clipping threshold (None to disable)
        limit_val_batches: Fraction of validation set to use (0.0 to 1.0)
    """
    seq_len: int
    micro_batch_size: int
    grad_accum_steps: int
    max_steps: int
    eval_every: int
    save_every: int
    lr: float
    weight_decay: float
    betas: List[float]
    eps: float
    warmup_ratio: float
    precision: str
    seed: int
    steps_per_epoch: Optional[int]
    gradient_clip_val: Optional[float]
    limit_val_batches: float

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.lr <= 0:
            raise ValueError("learning rate must be positive")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        if not 0 <= self.limit_val_batches <= 1:
            raise ValueError("limit_val_batches must be between 0 and 1")
        if len(self.betas) != 2:
            raise ValueError("betas must contain exactly 2 values")
        if not all(0 <= beta <= 1 for beta in self.betas):
            raise ValueError("beta values must be between 0 and 1")
        if self.seq_len <= 0 or self.micro_batch_size <= 0:
            raise ValueError("seq_len and micro_batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.precision not in ["16", "32", "bf16", "16-mixed", "bf16-mixed"]:
            raise ValueError("precision must be one of: 16, 32, bf16, 16-mixed, bf16-mixed")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_clip_val is not None and self.gradient_clip_val <= 0:
            raise ValueError("gradient_clip_val must be positive if specified")

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size across all devices."""
        return self.micro_batch_size * self.grad_accum_steps

    @property
    def warmup_steps(self) -> int:
        """Calculate number of warmup steps."""
        return max(1, int(self.warmup_ratio * self.max_steps))

    @property
    def is_mixed_precision(self) -> bool:
        """Check if using mixed precision training."""
        return "mixed" in self.precision or self.precision in ["16", "bf16"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'seq_len': self.seq_len,
            'micro_batch_size': self.micro_batch_size,
            'grad_accum_steps': self.grad_accum_steps,
            'max_steps': self.max_steps,
            'eval_every': self.eval_every,
            'save_every': self.save_every,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'betas': self.betas,
            'eps': self.eps,
            'warmup_ratio': self.warmup_ratio,
            'precision': self.precision,
            'seed': self.seed,
            'steps_per_epoch': self.steps_per_epoch,
            'gradient_clip_val': self.gradient_clip_val,
            'limit_val_batches': self.limit_val_batches
        }
