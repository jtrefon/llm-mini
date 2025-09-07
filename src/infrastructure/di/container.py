"""Dependency injection container for clean component wiring."""
from pathlib import Path
from typing import Any, Dict, Optional

from src.domain.entities.model_config import ModelConfiguration
from src.domain.entities.training_config import TrainingConfiguration
from src.infrastructure.config.config_loader import HardwareConfig, DataConfig
from src.domain.repositories.checkpoint_repository import CheckpointRepository
from src.infrastructure.repositories.file_checkpoint_repository import FileCheckpointRepository
from src.domain.services.training_service import TrainingService
from src.application.services.pytorch_lightning_training_service import PyTorchLightningTrainingService


class Container:
    """Simple dependency injection container.

    Provides centralized component wiring and dependency management,
    following the Dependency Inversion Principle.
    """

    def __init__(self):
        """Initialize container with empty service registry."""
        self._services: Dict[str, Any] = {}
        self._configs: Dict[str, Any] = {}

    def register_configs(self, model_config: ModelConfiguration,
                        training_config: TrainingConfiguration,
                        hardware_config: HardwareConfig,
                        data_config: DataConfig) -> None:
        """Register configuration objects.

        Args:
            model_config: Model architecture configuration
            training_config: Training hyperparameters
            hardware_config: Hardware settings
            data_config: Dataset configuration
        """
        # Populate vocab_size from tokenizer if not provided
        if model_config.vocab_size is None and data_config and getattr(data_config, 'tokenizer_name', None):
            try:
                from transformers import AutoTokenizer  # import locally to avoid hard dep at import time
                tok = AutoTokenizer.from_pretrained(data_config.tokenizer_name, use_fast=True)
                model_config.vocab_size = int(tok.vocab_size)
            except Exception:
                # Leave as None; model factory may handle or will raise a clearer error later
                pass

        self._configs.update({
            'model': model_config,
            'training': training_config,
            'hardware': hardware_config,
            'data': data_config
        })

    def get_checkpoint_repository(self) -> CheckpointRepository:
        """Get checkpoint repository instance (singleton pattern)."""
        if 'checkpoint_repo' not in self._services:
            checkpoints_dir = Path('checkpoints')
            legacy_logs_dir = Path('lightning_logs')

            self._services['checkpoint_repo'] = FileCheckpointRepository(
                checkpoints_dir=checkpoints_dir,
                legacy_logs_dir=legacy_logs_dir
            )
        return self._services['checkpoint_repo']

    def get_model_factory(self) -> Any:
        """Get model factory instance."""
        if 'model_factory' not in self._services:
            # Import here to avoid circular dependencies
            from src.infrastructure.models.pytorch_model_factory import PyTorchModelFactory
            self._services['model_factory'] = PyTorchModelFactory()
        return self._services['model_factory']

    def get_data_loader_factory(self) -> Any:
        """Get data loader factory instance."""
        if 'data_loader_factory' not in self._services:
            # Import here to avoid circular dependencies
            from src.infrastructure.data.pytorch_data_loader_factory import PyTorchDataLoaderFactory
            data_config = self._configs.get('data')
            hardware_config = self._configs.get('hardware')
            if data_config:
                num_workers = getattr(hardware_config, 'num_workers', 0) if hardware_config else 0
                self._services['data_loader_factory'] = PyTorchDataLoaderFactory(data_config, num_workers=num_workers)
            else:
                raise ValueError("Data configuration not registered")
        return self._services['data_loader_factory']

    def get_training_service(self) -> TrainingService:
        """Get training service instance (singleton pattern)."""
        if 'training_service' not in self._services:
            checkpoint_repo = self.get_checkpoint_repository()
            model_factory = self.get_model_factory()
            data_loader_factory = self.get_data_loader_factory()

            self._services['training_service'] = PyTorchLightningTrainingService(
                checkpoint_repo=checkpoint_repo,
                model_factory=model_factory,
                data_loader_factory=data_loader_factory,
                hardware_config=self._configs.get('hardware')
            )
        return self._services['training_service']

    def get_config(self, config_name: str) -> Any:
        """Get registered configuration by name.

        Args:
            config_name: Name of configuration ('model', 'training', 'hardware', 'data')

        Returns:
            Configuration object

        Raises:
            KeyError: If configuration not found
        """
        if config_name not in self._configs:
            raise KeyError(f"Configuration '{config_name}' not registered")
        return self._configs[config_name]

    def clear_services(self) -> None:
        """Clear service registry (useful for testing)."""
        self._services.clear()
