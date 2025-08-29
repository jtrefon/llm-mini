"""Tests for application layer services."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from src.application.services.pytorch_lightning_training_service import PyTorchLightningTrainingService
from src.domain.services.training_service import TrainingResult
from src.domain.entities.model_config import ModelConfiguration
from src.domain.entities.training_config import TrainingConfiguration


class TestPyTorchLightningTrainingService:
    """Test cases for PyTorch Lightning Training Service."""

    @pytest.fixture
    def mock_checkpoint_repo(self):
        """Mock checkpoint repository."""
        repo = Mock()
        repo.find_latest_checkpoint.return_value = Mock(
            path=Path("/tmp/checkpoint.ckpt"),
            epoch=10,
            step=1000,
            val_loss=0.123
        )
        repo.find_best_checkpoint.return_value = Mock(
            path=Path("/tmp/best.ckpt"),
            epoch=8,
            step=800,
            val_loss=0.089
        )
        return repo

    @pytest.fixture
    def mock_model_factory(self):
        """Mock model factory."""
        factory = Mock()
        mock_model = Mock()
        mock_model.configure_optimizers.return_value = {
            "optimizer": Mock(),
            "lr_scheduler": {"scheduler": Mock(), "interval": "step", "frequency": 1}
        }
        factory.create_model.return_value = mock_model
        return factory

    @pytest.fixture
    def mock_data_loader_factory(self):
        """Mock data loader factory."""
        factory = Mock()
        mock_train_loader = Mock()
        mock_train_loader.dataset = Mock()
        mock_train_loader.dataset.__len__ = Mock(return_value=1000)

        mock_val_loader = Mock()
        mock_val_loader.dataset = Mock()
        mock_val_loader.dataset.__len__ = Mock(return_value=100)

        factory.create_loaders.return_value = (mock_train_loader, mock_val_loader)
        return factory

    def test_service_creation(self, mock_checkpoint_repo, mock_model_factory, mock_data_loader_factory):
        """Test service creation with dependencies."""
        service = PyTorchLightningTrainingService(
            checkpoint_repo=mock_checkpoint_repo,
            model_factory=mock_model_factory,
            data_loader_factory=mock_data_loader_factory
        )

        assert service.checkpoint_repo == mock_checkpoint_repo
        assert service.model_factory == mock_model_factory
        assert service.data_loader_factory == mock_data_loader_factory

    @patch('pytorch_lightning.Trainer')
    def test_train_model_success(self, mock_trainer_class, mock_checkpoint_repo,
                                mock_model_factory, mock_data_loader_factory,
                                sample_model_config, sample_training_config):
        """Test successful model training."""
        # Setup mocks
        mock_trainer = Mock()
        mock_trainer.fit.return_value = None
        mock_trainer.global_step = 1000
        mock_trainer_class.return_value = mock_trainer

        service = PyTorchLightningTrainingService(
            checkpoint_repo=mock_checkpoint_repo,
            model_factory=mock_model_factory,
            data_loader_factory=mock_data_loader_factory
        )

        # Execute training
        result = service.train_model(sample_model_config, sample_training_config)

        # Verify result
        assert isinstance(result, TrainingResult)
        assert result.final_checkpoint_path == Path("/tmp/checkpoint.ckpt")
        assert result.best_val_loss == 0.089
        assert result.total_steps == 1000

        # Verify interactions
        mock_model_factory.create_model.assert_called_once_with(sample_model_config)
        mock_data_loader_factory.create_loaders.assert_called_once_with(sample_training_config)
        mock_trainer_class.assert_called_once()
        mock_trainer.fit.assert_called_once()

    @patch('pytorch_lightning.Trainer')
    def test_resume_training_success(self, mock_trainer_class, mock_checkpoint_repo,
                                    mock_model_factory, mock_data_loader_factory,
                                    sample_model_config, sample_training_config):
        """Test successful training resume."""
        # Setup mocks
        mock_trainer = Mock()
        mock_trainer.fit.return_value = None
        mock_trainer.global_step = 1200
        mock_trainer_class.return_value = mock_trainer

        mock_checkpoint_repo.checkpoint_exists.return_value = True

        service = PyTorchLightningTrainingService(
            checkpoint_repo=mock_checkpoint_repo,
            model_factory=mock_model_factory,
            data_loader_factory=mock_data_loader_factory
        )

        checkpoint_path = Path("/tmp/resume.ckpt")

        # Execute resume
        result = service.resume_training(checkpoint_path, additional_steps=200)

        # Verify result
        assert isinstance(result, TrainingResult)
        assert result.total_steps == 1200

        # Verify checkpoint existence was checked
        mock_checkpoint_repo.checkpoint_exists.assert_called_once_with(checkpoint_path)

        # Verify trainer was called with checkpoint
        mock_trainer.fit.assert_called_once()
        call_args = mock_trainer.fit.call_args
        assert call_args[1]['ckpt_path'] == str(checkpoint_path)

    def test_resume_training_checkpoint_not_found(self, mock_checkpoint_repo,
                                                 mock_model_factory, mock_data_loader_factory):
        """Test resume training with missing checkpoint."""
        mock_checkpoint_repo.checkpoint_exists.return_value = False

        service = PyTorchLightningTrainingService(
            checkpoint_repo=mock_checkpoint_repo,
            model_factory=mock_model_factory,
            data_loader_factory=mock_data_loader_factory
        )

        checkpoint_path = Path("/tmp/missing.ckpt")

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            service.resume_training(checkpoint_path, additional_steps=100)

        mock_checkpoint_repo.checkpoint_exists.assert_called_once_with(checkpoint_path)

    def test_validate_configuration_success(self, mock_checkpoint_repo,
                                          mock_model_factory, mock_data_loader_factory,
                                          sample_model_config, sample_training_config):
        """Test successful configuration validation."""
        service = PyTorchLightningTrainingService(
            checkpoint_repo=mock_checkpoint_repo,
            model_factory=mock_model_factory,
            data_loader_factory=mock_data_loader_factory
        )

        # Should not raise any exception
        service.validate_configuration(sample_model_config, sample_training_config)

    def test_validate_configuration_invalid_model(self, mock_checkpoint_repo,
                                                mock_model_factory, mock_data_loader_factory,
                                                sample_training_config):
        """Test configuration validation with invalid model config."""
        service = PyTorchLightningTrainingService(
            checkpoint_repo=mock_checkpoint_repo,
            model_factory=mock_model_factory,
            data_loader_factory=mock_data_loader_factory
        )

        # Create invalid model config
        invalid_config = sample_training_config
        invalid_config.n_layers = 0  # This should fail validation

        with pytest.raises(ValueError, match="n_layers must be positive"):
            service.validate_configuration(invalid_config, sample_training_config)

    def test_validate_configuration_invalid_training(self, mock_checkpoint_repo,
                                                   mock_model_factory, mock_data_loader_factory,
                                                   sample_model_config):
        """Test configuration validation with invalid training config."""
        service = PyTorchLightningTrainingService(
            checkpoint_repo=mock_checkpoint_repo,
            model_factory=mock_model_factory,
            data_loader_factory=mock_data_loader_factory
        )

        # Create invalid training config
        invalid_config = sample_model_config
        invalid_config.lr = -0.001  # Negative learning rate

        with pytest.raises(ValueError, match="learning rate must be positive"):
            service.validate_configuration(sample_model_config, invalid_config)

    def test_get_training_status(self, mock_checkpoint_repo, mock_model_factory,
                                mock_data_loader_factory):
        """Test getting training status."""
        service = PyTorchLightningTrainingService(
            checkpoint_repo=mock_checkpoint_repo,
            model_factory=mock_model_factory,
            data_loader_factory=mock_data_loader_factory
        )

        status = service.get_training_status()

        assert isinstance(status, dict)
        assert "latest_checkpoint" in status
        assert "best_val_loss" in status
        assert "total_steps" in status

        mock_checkpoint_repo.find_latest_checkpoint.assert_called_once()
        mock_checkpoint_repo.find_best_checkpoint.assert_called_once()
        mock_checkpoint_repo.list_checkpoints.assert_called_once()

    def test_create_optimizer(self, mock_checkpoint_repo, mock_model_factory,
                             mock_data_loader_factory, sample_training_config):
        """Test optimizer creation."""
        service = PyTorchLightningTrainingService(
            checkpoint_repo=mock_checkpoint_repo,
            model_factory=mock_model_factory,
            data_loader_factory=mock_data_loader_factory
        )

        mock_model = Mock()
        optimizer = service._create_optimizer(mock_model, sample_training_config)

        # Verify optimizer was created (we can't easily test the exact type without torch)
        assert optimizer is not None

    def test_create_scheduler(self, mock_checkpoint_repo, mock_model_factory,
                             mock_data_loader_factory, sample_training_config):
        """Test scheduler creation."""
        service = PyTorchLightningTrainingService(
            checkpoint_repo=mock_checkpoint_repo,
            model_factory=mock_model_factory,
            data_loader_factory=mock_data_loader_factory
        )

        mock_optimizer = Mock()
        scheduler = service._create_scheduler(mock_optimizer, sample_training_config)

        # Verify scheduler was created
        assert scheduler is not None
        assert hasattr(scheduler, 'step')

    def test_create_callbacks(self, mock_checkpoint_repo, mock_model_factory,
                             mock_data_loader_factory, sample_training_config):
        """Test callback creation."""
        service = PyTorchLightningTrainingService(
            checkpoint_repo=mock_checkpoint_repo,
            model_factory=mock_model_factory,
            data_loader_factory=mock_data_loader_factory
        )

        callbacks = service._create_callbacks(sample_training_config)

        # Verify callbacks were created
        assert isinstance(callbacks, list)
        assert len(callbacks) > 0  # Should have at least model checkpoint callback

    def test_build_trainer_kwargs(self, mock_checkpoint_repo, mock_model_factory,
                                 mock_data_loader_factory, sample_training_config):
        """Test trainer kwargs building."""
        service = PyTorchLightningTrainingService(
            checkpoint_repo=mock_checkpoint_repo,
            model_factory=mock_model_factory,
            data_loader_factory=mock_data_loader_factory
        )

        mock_callbacks = [Mock()]
        kwargs = service._build_trainer_kwargs(sample_training_config, mock_callbacks)

        assert isinstance(kwargs, dict)
        assert 'max_steps' in kwargs
        assert 'precision' in kwargs
        assert 'callbacks' in kwargs
        assert kwargs['max_steps'] == sample_training_config.max_steps
        assert kwargs['precision'] == sample_training_config.precision
        assert kwargs['callbacks'] == mock_callbacks
