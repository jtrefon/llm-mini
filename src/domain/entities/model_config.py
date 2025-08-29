"""Model configuration entity - encapsulates all model hyperparameters."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfiguration:
    """Model hyperparameters with validation and type safety.

    This entity represents the complete configuration for a GPT-style transformer model,
    ensuring all hyperparameters are validated and properly typed.

    Attributes:
        n_layers: Number of transformer layers in the model architecture
        d_model: Model dimension (embedding size)
        n_heads: Number of attention heads
        n_kv_heads: Number of key-value heads for Grouped-Query Attention (GQA)
        d_ff: Feed-forward network dimension (typically 4x d_model)
        dropout: Dropout probability for regularization (0.0 to disable)
        vocab_size: Size of the tokenizer vocabulary (set from tokenizer at runtime)
        rope_theta: Base frequency for RoPE positional embeddings
        tie_embeddings: Whether to tie input and output embeddings
        swa_window: Sliding Window Attention window size (0 to disable)
    """
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    dropout: float
    vocab_size: Optional[int] = None
    rope_theta: float = 10000.0
    tie_embeddings: bool = False
    swa_window: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        # Check positivity before relational checks so error messages match tests
        if self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("n_heads and n_kv_heads must be positive")
        if self.n_kv_heads > self.n_heads:
            raise ValueError("n_kv_heads cannot exceed n_heads")
        if self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError("d_model and n_layers must be positive")
        if not (0 <= self.dropout <= 1):
            raise ValueError("dropout must be between 0 and 1")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

    @property
    def head_dim(self) -> int:
        """Calculate head dimension from model dimension and number of heads."""
        return self.d_model // self.n_heads

    @property
    def is_gqa_enabled(self) -> bool:
        """Check if Grouped-Query Attention is enabled."""
        return self.n_kv_heads != self.n_heads

    @property
    def gqa_group_size(self) -> int:
        """Calculate GQA group size (how many Q heads share KV heads)."""
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'n_layers': self.n_layers,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_kv_heads': self.n_kv_heads,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'vocab_size': self.vocab_size,
            'rope_theta': self.rope_theta,
            'tie_embeddings': self.tie_embeddings,
            'swa_window': self.swa_window
        }
