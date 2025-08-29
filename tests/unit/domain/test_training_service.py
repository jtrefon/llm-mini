"""Tests for training service interface."""
import pytest
from pathlib import Path
from unittest.mock import Mock

from src.domain.services.training_service import TrainingService, TrainingResult
from src.domain.entities.model_config import ModelConfiguration
from src.domain.entities.training_config import TrainingConfiguration


class TestTrainingResult:
    """Test cases for TrainingResult data structure."""

    def test_training_result_creation(self):
        """Test creating a training result object."""
        checkpoint_path = Path("/tmp/final.ckpt")
        result = TrainingResult(
            final_checkpoint_path=checkpoint_path,
            best_val_loss=0.123,
            total_steps=1000
        )

        assert result.final_checkpoint_path == checkpoint_path
        assert result.best_val_loss == 0.123
        assert result.total_steps == 1000

    def test_training_result_repr(self):
        """Test string representation of training result."""
        checkpoint_path = Path("/tmp/final.ckpt")
        result = TrainingResult(
            final_checkpoint_path=checkpoint_path,
            best_val_loss=0.123,
            total_steps=1000
        )

        repr_str = repr(result)
        assert "TrainingResult" in repr_str
        assert "final.ckpt" in repr_str
        assert "best_val_loss=0.123" in repr_str
        assert "total_steps=1000" in repr_str


class TestTrainingServiceInterface:
    """Test cases for TrainingService abstract interface."""

    def test_service_is_abstract(self):
        """Test that TrainingService cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TrainingService()

    def test_abstract_methods_exist(self):
        """Test that all abstract methods are defined."""
        # Create a concrete implementation to test method signatures
        class ConcreteTrainingService(TrainingService):
            def train_model(self, model_config, training_config):
                return TrainingResult(
                    Path("/tmp/final.ckpt"), 0.123, 1000
                )

            def resume_training(self, checkpoint_path, additional_steps):
                return TrainingResult(
                    Path("/tmp/final.ckpt"), 0.123, 1000
                )

            def validate_configuration(self, model_config, training_config):
                pass

            def get_training_status(self):
                return {"status": "running"}

        service = ConcreteTrainingService()

        # Test method signatures
        assert callable(service.train_model)
        assert callable(service.resume_training)
        assert callable(service.validate_configuration)
        assert callable(service.get_training_status)

        # Test return types
        result = service.train_model(Mock(), Mock())
        assert isinstance(result, TrainingResult)

        status = service.get_training_status()
        assert isinstance(status, dict)

    def test_mock_service_behavior(self, sample_model_config, sample_training_config):
        """Test service behavior with mock implementation."""
        class MockTrainingService(TrainingService):
            def train_model(self, model_config, training_config):
                return TrainingResult(
                    Path("/tmp/final.ckpt"), 0.089, 1500
                )

            def resume_training(self, checkpoint_path, additional_steps):
                return TrainingResult(
                    Path("/tmp/resumed.ckpt"), 0.078, 2000
                )

            def validate_configuration(self, model_config, training_config):
                if model_config.n_layers <= 0:
                    raise ValueError("Invalid model config")

            def get_training_status(self):
                return {
                    "latest_checkpoint": "epoch=10-step=1000-val_loss=0.089.ckpt",
                    "best_val_loss": 0.078,
                    "total_steps": 2000
                }

        service = MockTrainingService()

        # Test training
        result = service.train_model(sample_model_config, sample_training_config)
        assert result.total_steps == 1500
        assert result.best_val_loss == 0.089

        # Test resume
        resume_result = service.resume_training(Path("/tmp/checkpoint.ckpt"), 500)
        assert resume_result.total_steps == 2000

        # Test validation
        service.validate_configuration(sample_model_config, sample_training_config)

        # Test invalid config
        invalid_config = sample_model_config
        invalid_config.n_layers = 0
        with pytest.raises(ValueError):
            service.validate_configuration(invalid_config, sample_training_config)

        # Test status
        status = service.get_training_status()
        assert status["total_steps"] == 2000
        assert "latest_checkpoint" in status
