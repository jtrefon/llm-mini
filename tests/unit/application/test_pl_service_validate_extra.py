"""Extra coverage for PyTorchLightningTrainingService.validate_configuration."""
from unittest.mock import Mock
import pytest

from src.application.services.pytorch_lightning_training_service import PyTorchLightningTrainingService
from src.domain.entities.model_config import ModelConfiguration
from src.domain.entities.training_config import TrainingConfiguration


def _svc():
    return PyTorchLightningTrainingService(checkpoint_repo=Mock(), model_factory=Mock(), data_loader_factory=Mock())


def test_validate_configuration_effective_batch_non_positive():
    svc = _svc()
    model_cfg = ModelConfiguration(n_layers=1, d_model=8, n_heads=1, n_kv_heads=1, d_ff=32, dropout=0.0)
    # Use a duck-typed object to avoid dataclass validation and force effective_batch_size <= 0
    class DummyCfg:
        lr = 1e-3
        effective_batch_size = 0

    with pytest.raises(ValueError):
        svc.validate_configuration(model_cfg, DummyCfg())
