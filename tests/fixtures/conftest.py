"""Test fixtures for unit testing."""
import pytest
from pathlib import Path
from unittest.mock import Mock

from src.domain.entities.model_config import ModelConfiguration
from src.domain.entities.training_config import TrainingConfiguration


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return ModelConfiguration(
        n_layers=6,
        d_model=512,
        n_heads=8,
        n_kv_heads=4,
        d_ff=2048,
        dropout=0.1,
        vocab_size=30000,
        rope_theta=10000.0,
        tie_embeddings=True,
        swa_window=512
    )


@pytest.fixture
def sample_training_config():
    """Sample training configuration for testing."""
    return TrainingConfiguration(
        seq_len=512,
        micro_batch_size=4,
        grad_accum_steps=8,
        max_steps=1000,
        eval_every=100,
        save_every=200,
        lr=1e-4,
        weight_decay=0.1,
        betas=[0.9, 0.95],
        eps=1e-8,
        warmup_ratio=0.1,
        precision="32",
        seed=42,
        steps_per_epoch=50,
        gradient_clip_val=1.0,
        limit_val_batches=1.0
    )


@pytest.fixture
def mock_checkpoint_repo():
    """Mock checkpoint repository for testing."""
    repo = Mock()
    repo.find_latest_checkpoint.return_value = Mock(
        path=Path("/tmp/checkpoint.ckpt"),
        epoch=5,
        step=500,
        val_loss=0.123,
        name="epoch=5-step=500-val_loss=0.123.ckpt"
    )
    repo.find_best_checkpoint.return_value = Mock(
        path=Path("/tmp/best.ckpt"),
        epoch=3,
        step=300,
        val_loss=0.089,
        name="epoch=3-step=300-val_loss=0.089.ckpt"
    )
    repo.checkpoint_exists.return_value = True
    return repo


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for file-based tests."""
    return tmp_path


@pytest.fixture
def sample_checkpoint_path(temp_dir):
    """Sample checkpoint path for testing."""
    return temp_dir / "sample_checkpoint.ckpt"
