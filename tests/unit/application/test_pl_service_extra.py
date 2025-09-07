"""Extra coverage tests for PyTorchLightningTrainingService internals."""
from unittest.mock import Mock
import torch
import torch.nn as nn

from src.application.services.pytorch_lightning_training_service import PyTorchLightningTrainingService
from src.domain.entities.model_config import ModelConfiguration
from src.domain.entities.training_config import TrainingConfiguration


def _make_service():
    checkpoint_repo = Mock()
    checkpoint_repo.find_latest_checkpoint.return_value = None
    checkpoint_repo.find_best_checkpoint.return_value = None
    model_factory = Mock()
    data_loader_factory = Mock()
    return PyTorchLightningTrainingService(checkpoint_repo, model_factory, data_loader_factory)


def test_reconstruct_configs_cover_paths():
    svc = _make_service()
    # Reconstruct model config
    mc = svc._reconstruct_model_config({'model': {
        'n_layers': 1, 'd_model': 8, 'n_heads': 1, 'n_kv_heads': 1, 'd_ff': 32, 'dropout': 0.0
    }})
    assert isinstance(mc, ModelConfiguration)
    # Reconstruct training config
    tc = svc._reconstruct_training_config({'training': {
        'seq_len': 8, 'micro_batch_size': 1, 'grad_accum_steps': 1,
        'max_steps': 2, 'eval_every': 1, 'save_every': 0,
        'lr': 1e-3, 'weight_decay': 0.0, 'betas': [0.9, 0.95], 'eps': 1e-8,
        'warmup_ratio': 0.01, 'precision': '32', 'seed': 42,
        'steps_per_epoch': None, 'gradient_clip_val': None, 'limit_val_batches': 1.0
    }})
    assert isinstance(tc, TrainingConfiguration)


def test_create_optimizer_with_real_parameters_and_dummy_branch():
    svc = _make_service()

    # Real module branch
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.ones(1))
    model_real = Tiny()
    cfg = TrainingConfiguration(
        seq_len=8, micro_batch_size=1, grad_accum_steps=1,
        max_steps=2, eval_every=1, save_every=0,
        lr=1e-3, weight_decay=0.0, betas=[0.9, 0.95], eps=1e-8,
        warmup_ratio=0.01, precision='32', seed=42,
        steps_per_epoch=None, gradient_clip_val=None, limit_val_batches=1.0
    )
    opt1 = svc._create_optimizer(model_real, cfg)
    assert opt1 is not None
    assert len(opt1.param_groups) > 0

    # Dummy branch: model without parameters attribute
    model_dummy = object()
    opt2 = svc._create_optimizer(model_dummy, cfg)
    assert opt2 is not None
    assert len(opt2.param_groups) > 0
