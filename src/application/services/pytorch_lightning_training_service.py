"""PyTorch Lightning implementation of training service."""
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Any, Dict

from src.domain.services.training_service import TrainingService, TrainingResult
from src.domain.entities.model_config import ModelConfiguration
from src.domain.entities.training_config import TrainingConfiguration
from src.domain.repositories.checkpoint_repository import CheckpointRepository


class PyTorchLightningTrainingService(TrainingService):
    """Concrete training service using PyTorch Lightning.

    Implements the training service interface using PyTorch Lightning,
    providing a complete training pipeline with checkpointing, logging,
    and validation.
    """

    def __init__(self, checkpoint_repo: CheckpointRepository,
                 model_factory: Any, data_loader_factory: Any):
        """Initialize the training service.

        Args:
            checkpoint_repo: Repository for checkpoint operations
            model_factory: Factory for creating model instances
            data_loader_factory: Factory for creating data loaders
        """
        self.checkpoint_repo = checkpoint_repo
        self.model_factory = model_factory
        self.data_loader_factory = data_loader_factory

    def train_model(self, model_config: ModelConfiguration,
                   training_config: TrainingConfiguration) -> TrainingResult:
        """Execute training with PyTorch Lightning."""
        self.validate_configuration(model_config, training_config)

        # Create model and data loaders
        model = self.model_factory.create_model(model_config)
        train_loader, val_loader = self.data_loader_factory.create_loaders(training_config)

        # Configure training components
        optimizer = self._create_optimizer(model, training_config)
        scheduler = self._create_scheduler(optimizer, training_config)
        callbacks = self._create_callbacks(training_config)

        # Setup trainer
        trainer_kwargs = self._build_trainer_kwargs(training_config, callbacks)
        trainer = pl.Trainer(**trainer_kwargs)

        # Execute training
        trainer.fit(model, train_loader, val_loader)

        # Collect results
        best_checkpoint = self.checkpoint_repo.find_best_checkpoint()
        final_checkpoint = self.checkpoint_repo.find_latest_checkpoint()

        return TrainingResult(
            final_checkpoint_path=final_checkpoint.path if final_checkpoint else None,
            best_val_loss=best_checkpoint.val_loss if best_checkpoint else float('inf'),
            total_steps=trainer.global_step
        )

    def resume_training(self, checkpoint_path: Path, additional_steps: int) -> TrainingResult:
        """Resume training from checkpoint."""
        if not self.checkpoint_repo.checkpoint_exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint to get model and training config
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        hparams = checkpoint.get('hyperparameters', {})

        # Reconstruct configurations from checkpoint
        model_config = self._reconstruct_model_config(hparams)
        training_config = self._reconstruct_training_config(hparams)
        training_config.max_steps += additional_steps  # Add requested steps

        # Create fresh components
        model = self.model_factory.create_model(model_config)
        train_loader, val_loader = self.data_loader_factory.create_loaders(training_config)

        # Setup trainer with resume
        callbacks = self._create_callbacks(training_config)
        trainer_kwargs = self._build_trainer_kwargs(training_config, callbacks)
        trainer = pl.Trainer(**trainer_kwargs)

        # Resume training
        trainer.fit(model, train_loader, val_loader, ckpt_path=str(checkpoint_path))

        # Collect results
        best_checkpoint = self.checkpoint_repo.find_best_checkpoint()
        final_checkpoint = self.checkpoint_repo.find_latest_checkpoint()

        return TrainingResult(
            final_checkpoint_path=final_checkpoint.path if final_checkpoint else None,
            best_val_loss=best_checkpoint.val_loss if best_checkpoint else float('inf'),
            total_steps=trainer.global_step
        )

    def validate_configuration(self, model_config: ModelConfiguration,
                             training_config: TrainingConfiguration) -> None:
        """Validate training configuration compatibility."""
        # Model configuration validation (already done in __post_init__)

        # Training configuration validation (already done in __post_init__)

        # Cross-validation between model and training configs
        if training_config.seq_len <= 0:
            raise ValueError("seq_len must be positive")

        # Check if configurations are compatible
        effective_batch = training_config.effective_batch_size
        if effective_batch <= 0:
            raise ValueError("effective batch size must be positive")

    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        latest_checkpoint = self.checkpoint_repo.find_latest_checkpoint()
        best_checkpoint = self.checkpoint_repo.find_best_checkpoint()

        return {
            'latest_checkpoint': latest_checkpoint.name if latest_checkpoint else None,
            'latest_step': latest_checkpoint.step if latest_checkpoint else None,
            'best_checkpoint': best_checkpoint.name if best_checkpoint else None,
            'best_val_loss': best_checkpoint.val_loss if best_checkpoint else None,
            'total_checkpoints': len(self.checkpoint_repo.list_checkpoints())
        }

    def _create_optimizer(self, model, config: TrainingConfiguration):
        """Create AdamW optimizer."""
        import torch
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )

    def _create_scheduler(self, optimizer, config: TrainingConfiguration):
        """Create warmup + cosine scheduler."""
        from src.infrastructure.schedulers.warmup_cosine import WarmupCosineScheduler
        return WarmupCosineScheduler(optimizer, config.warmup_ratio, config.max_steps)

    def _create_callbacks(self, config: TrainingConfiguration):
        """Create training callbacks."""
        from src.infrastructure.callbacks import create_training_callbacks
        return create_training_callbacks(config)

    def _build_trainer_kwargs(self, config: TrainingConfiguration, callbacks):
        """Build trainer arguments from configuration."""
        return {
            'max_steps': config.max_steps,
            'precision': config.precision,
            'accumulate_grad_batches': config.grad_accum_steps,
            'log_every_n_steps': config.eval_every,
            'logger': False,  # Disable logging to avoid TensorBoard warnings
            'enable_checkpointing': True,
            'callbacks': callbacks,
            'gradient_clip_val': config.gradient_clip_val or 1.0,
            'limit_val_batches': config.limit_val_batches,
        }

    def _reconstruct_model_config(self, hparams: Dict[str, Any]) -> ModelConfiguration:
        """Reconstruct model config from checkpoint hyperparameters."""
        model_hparams = hparams.get('model', {})
        return ModelConfiguration(**model_hparams)

    def _reconstruct_training_config(self, hparams: Dict[str, Any]) -> TrainingConfiguration:
        """Reconstruct training config from checkpoint hyperparameters."""
        training_hparams = hparams.get('training', {})
        return TrainingConfiguration(**training_hparams)
