"""Configuration management infrastructure - Type-safe YAML configuration loading."""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, ValidationError

from src.domain.entities.model_config import ModelConfiguration
from src.domain.entities.training_config import TrainingConfiguration


class HardwareConfig(BaseModel):
    """Hardware configuration with validation."""
    accelerator: str
    devices: int
    num_workers: int

    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class DataConfig(BaseModel):
    """Data configuration with validation."""
    dataset: str
    split: str
    text_field: str
    streaming: bool
    tokenizer_name: str
    max_shards: Optional[int]
    pack_sequences: bool
    train_docs: int
    val_docs: int

    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class ConfigLoader:
    """YAML configuration loader with validation and type safety.

    This class replaces scattered configuration access throughout the codebase
    with a centralized, validated configuration management system.
    """

    def __init__(self, config_path: Path):
        """Initialize configuration loader.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path

    def load_model_config(self) -> ModelConfiguration:
        """Load and validate model configuration."""
        config_data = self._load_yaml()
        model_data = config_data.get('model', {})

        # Convert vocab_size from None to Optional[int] for domain entity
        if 'vocab_size' in model_data and model_data['vocab_size'] is None:
            model_data['vocab_size'] = None

        return ModelConfiguration(**model_data)

    def load_training_config(self) -> TrainingConfiguration:
        """Load and validate training configuration."""
        config_data = self._load_yaml()
        training_data = config_data.get('training', {})

        # Handle optional fields with defaults
        training_data.setdefault('steps_per_epoch', None)
        training_data.setdefault('gradient_clip_val', None)

        # Coerce numeric fields that might come as strings in some YAML scenarios
        def _to_int(v):
            return None if v is None else int(v)

        def _to_float(v):
            return None if v is None else float(v)

        numeric_int_fields = [
            'seq_len', 'micro_batch_size', 'grad_accum_steps', 'max_steps',
            'eval_every', 'save_every', 'seed', 'steps_per_epoch'
        ]
        numeric_float_fields = [
            'lr', 'weight_decay', 'eps', 'warmup_ratio', 'gradient_clip_val',
            'limit_val_batches'
        ]

        for k in numeric_int_fields:
            if k in training_data:
                training_data[k] = _to_int(training_data[k])
        for k in numeric_float_fields:
            if k in training_data:
                training_data[k] = _to_float(training_data[k])
        if 'betas' in training_data and training_data['betas'] is not None:
            training_data['betas'] = [float(b) for b in training_data['betas']]

        return TrainingConfiguration(**training_data)

    def load_hardware_config(self) -> HardwareConfig:
        """Load and validate hardware configuration."""
        config_data = self._load_yaml()
        hardware_data = config_data.get('hardware', {})
        return HardwareConfig(**hardware_data)

    def load_data_config(self) -> DataConfig:
        """Load and validate data configuration."""
        config_data = self._load_yaml()
        data_data = config_data.get('data', {})
        return DataConfig(**data_data)

    def load_all_configs(self) -> Dict[str, Any]:
        """Load all configurations at once."""
        return {
            'model': self.load_model_config(),
            'training': self.load_training_config(),
            'hardware': self.load_hardware_config(),
            'data': self.load_data_config()
        }

    def _load_yaml(self) -> Dict[str, Any]:
        """Load raw YAML data with error handling."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")

    def validate_config_file(self) -> bool:
        """Validate that configuration file can be loaded and parsed."""
        try:
            self._load_yaml()
            return True
        except Exception:
            return False

    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema for documentation."""
        return {
            'model': ModelConfiguration.__annotations__,
            'training': TrainingConfiguration.__annotations__,
            'hardware': HardwareConfig.__annotations__,
            'data': DataConfig.__annotations__
        }
