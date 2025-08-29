"""Training orchestration service - Business logic for training operations."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from src.domain.entities.model_config import ModelConfiguration
from src.domain.entities.training_config import TrainingConfiguration


class TrainingResult:
    """Result of a training session.

    Encapsulates the outcomes of a training run, providing
    access to final model state and training metrics.
    """

    def __init__(self, final_checkpoint_path: Path, best_val_loss: float, total_steps: int):
        """Initialize training result.

        Args:
            final_checkpoint_path: Path to the final checkpoint
            best_val_loss: Best validation loss achieved during training
            total_steps: Total number of training steps completed
        """
        self.final_checkpoint_path = final_checkpoint_path
        self.best_val_loss = best_val_loss
        self.total_steps = total_steps

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"TrainingResult(final_checkpoint={self.final_checkpoint_path.name}, "
                f"best_val_loss={self.best_val_loss:.4f}, total_steps={self.total_steps})")


class TrainingService(ABC):
    """Abstract interface for training orchestration.

    Defines the contract for training operations, allowing different
    implementations (PyTorch Lightning, custom training loop, etc.)
    to provide the same business functionality.
    """

    @abstractmethod
    def train_model(self, model_config: ModelConfiguration,
                   training_config: TrainingConfiguration) -> TrainingResult:
        """Execute complete training pipeline.

        Args:
            model_config: Model architecture configuration
            training_config: Training hyperparameters and settings

        Returns:
            TrainingResult with final model state and metrics
        """
        pass

    @abstractmethod
    def resume_training(self, checkpoint_path: Path, additional_steps: int) -> TrainingResult:
        """Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint to resume from
            additional_steps: Number of additional steps to train

        Returns:
            TrainingResult with final model state and metrics
        """
        pass

    @abstractmethod
    def validate_configuration(self, model_config: ModelConfiguration,
                             training_config: TrainingConfiguration) -> None:
        """Validate training configuration compatibility.

        Args:
            model_config: Model configuration to validate
            training_config: Training configuration to validate

        Raises:
            ValueError: If configuration is invalid or incompatible
        """
        pass

    @abstractmethod
    def get_training_status(self) -> dict:
        """Get current training status and progress information.

        Returns:
            Dictionary with training status information
        """
        pass
