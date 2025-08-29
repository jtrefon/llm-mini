"""Tests for checkpoint repository interface."""
import pytest
from pathlib import Path
from unittest.mock import Mock

from src.domain.repositories.checkpoint_repository import CheckpointRepository, CheckpointInfo


class TestCheckpointInfo:
    """Test cases for CheckpointInfo data structure."""

    def test_checkpoint_info_creation(self):
        """Test creating a checkpoint info object."""
        path = Path("/tmp/checkpoint.ckpt")
        info = CheckpointInfo(path, epoch=5, step=500, val_loss=0.123)

        assert info.path == path
        assert info.epoch == 5
        assert info.step == 500
        assert info.val_loss == 0.123
        assert info.name == "checkpoint.ckpt"

    def test_checkpoint_info_with_none_values(self):
        """Test checkpoint info with None values."""
        path = Path("/tmp/checkpoint.ckpt")
        info = CheckpointInfo(path, epoch=None, step=None, val_loss=None)

        assert info.epoch is None
        assert info.step is None
        assert info.val_loss is None

    def test_checkpoint_info_repr(self):
        """Test string representation of checkpoint info."""
        path = Path("/tmp/checkpoint.ckpt")
        info = CheckpointInfo(path, epoch=5, step=500, val_loss=0.123)

        repr_str = repr(info)
        assert "CheckpointInfo" in repr_str
        assert "checkpoint.ckpt" in repr_str
        assert "epoch=5" in repr_str
        assert "step=500" in repr_str
        assert "val_loss=0.123" in repr_str


class TestCheckpointRepositoryInterface:
    """Test cases for CheckpointRepository abstract interface."""

    def test_repository_is_abstract(self):
        """Test that CheckpointRepository cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CheckpointRepository()

    def test_abstract_methods_exist(self):
        """Test that all abstract methods are defined."""
        # Create a concrete implementation to test method signatures
        class ConcreteRepository(CheckpointRepository):
            def find_latest_checkpoint(self):
                return None

            def find_best_checkpoint(self):
                return None

            def list_checkpoints(self):
                return []

            def parse_checkpoint_metadata(self, path):
                return CheckpointInfo(path, None, None, None)

            def checkpoint_exists(self, path):
                return path.exists()

        repo = ConcreteRepository()

        # Test method signatures
        assert callable(repo.find_latest_checkpoint)
        assert callable(repo.find_best_checkpoint)
        assert callable(repo.list_checkpoints)
        assert callable(repo.parse_checkpoint_metadata)
        assert callable(repo.checkpoint_exists)

        # Test return types
        assert repo.find_latest_checkpoint() is None
        assert repo.find_best_checkpoint() is None
        assert repo.list_checkpoints() == []
        assert isinstance(repo.parse_checkpoint_metadata(Path("/tmp/test.ckpt")), CheckpointInfo)
        assert isinstance(repo.checkpoint_exists(Path("/tmp/test.ckpt")), bool)
