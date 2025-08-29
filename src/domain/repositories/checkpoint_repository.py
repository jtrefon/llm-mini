"""Abstract checkpoint management - Repository pattern for data access."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional


class CheckpointInfo:
    """Checkpoint metadata container.

    Encapsulates information about a training checkpoint,
    including path, training progress, and performance metrics.
    """

    def __init__(self, path: Path, epoch: Optional[int], step: Optional[int], val_loss: Optional[float]):
        """Initialize checkpoint information.

        Args:
            path: Path to the checkpoint file
            epoch: Training epoch number (if available)
            step: Global training step number (if available)
            val_loss: Validation loss at this checkpoint (if available)
        """
        self.path = path
        self.epoch = epoch
        self.step = step
        self.val_loss = val_loss
        # Be tolerant if the file doesn't exist (tests may use fake paths)
        try:
            self.mtime = path.stat().st_mtime
        except FileNotFoundError:
            self.mtime = 0.0

    @property
    def name(self) -> str:
        """Get the checkpoint filename."""
        return self.path.name

    def __repr__(self) -> str:
        """String representation for debugging."""
        val_str = f"{self.val_loss:.4f}" if self.val_loss is not None else "None"
        return (
            f"CheckpointInfo(path={self.path.name}, "
            f"epoch={self.epoch}, step={self.step}, val_loss={val_str})"
        )


class CheckpointRepository(ABC):
    """Abstract interface for checkpoint management operations.

    This repository defines the contract for checkpoint data access,
    allowing different storage backends (file system, cloud storage, etc.)
    to implement the same interface.
    """

    @abstractmethod
    def find_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Find the most recent checkpoint by global step count.

        Returns:
            CheckpointInfo for the checkpoint with highest step count,
            or None if no checkpoints exist.
        """
        pass

    @abstractmethod
    def find_best_checkpoint(self) -> Optional[CheckpointInfo]:
        """Find the checkpoint with lowest validation loss.

        Returns:
            CheckpointInfo for the checkpoint with lowest val_loss,
            or None if no checkpoints with validation loss exist.
        """
        pass

    @abstractmethod
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all available checkpoints.

        Returns:
            List of CheckpointInfo objects, sorted by modification time (newest first).
        """
        pass

    @abstractmethod
    def parse_checkpoint_metadata(self, path: Path) -> CheckpointInfo:
        """Extract metadata from checkpoint filename.

        Args:
            path: Path to the checkpoint file

        Returns:
            CheckpointInfo with parsed metadata from filename
        """
        pass

    @abstractmethod
    def checkpoint_exists(self, path: Path) -> bool:
        """Check if a checkpoint exists at the given path.

        Args:
            path: Path to check for checkpoint existence

        Returns:
            True if checkpoint exists, False otherwise
        """
        pass
