"""Training callbacks factory - Infrastructure layer implementations."""
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from typing import List

from src.domain.entities.training_config import TrainingConfiguration


def create_training_callbacks(config: TrainingConfiguration) -> List[Callback]:
    """Factory for creating training callbacks based on configuration.

    This factory creates and configures all PyTorch Lightning callbacks
    needed for training, encapsulating callback setup logic.

    Args:
        config: Training configuration with callback settings

    Returns:
        List of configured PyTorch Lightning callbacks
    """
    callbacks = []

    # Model checkpoint callback for saving best models
    checkpoint_cb = _create_model_checkpoint_callback(config)
    if checkpoint_cb:
        callbacks.append(checkpoint_cb)

    # Step-based checkpoint callback for regular saves
    step_checkpoint_cb = _create_step_checkpoint_callback(config)
    if step_checkpoint_cb:
        callbacks.append(step_checkpoint_cb)

    # Early stopping callback
    early_stop_cb = _create_early_stopping_callback(config)
    if early_stop_cb:
        callbacks.append(early_stop_cb)

    return callbacks


def _create_model_checkpoint_callback(config: TrainingConfiguration) -> ModelCheckpoint:
    """Create model checkpoint callback for best model saving."""
    from pathlib import Path

    checkpoints_dir = Path('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)

    return ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename='epoch={epoch}-step={step}-val_loss={val_loss:.3f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )


def _create_step_checkpoint_callback(config: TrainingConfiguration) -> ModelCheckpoint:
    """Create step-based checkpoint callback for regular saves."""
    from pathlib import Path

    checkpoints_dir = Path('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)

    # Only create if save_every is configured
    if not hasattr(config, 'save_every') or config.save_every <= 0:
        return None

    return ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename='global_step={step}',
        save_top_k=-1,  # Keep all step checkpoints
        every_n_train_steps=config.save_every,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )


def _create_early_stopping_callback(config: TrainingConfiguration) -> EarlyStopping:
    """Create early stopping callback."""
    # Default early stopping configuration
    patience = getattr(config, 'early_stopping_patience', 5)
    min_delta = getattr(config, 'early_stopping_min_delta', 0.0)

    return EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=patience,
        min_delta=min_delta,
        check_on_train_epoch_end=False,
        verbose=True,
    )
