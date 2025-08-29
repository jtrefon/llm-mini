"""Tests for infrastructure configuration management."""
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml
from io import StringIO

from src.infrastructure.config.config_loader import ConfigLoader, HardwareConfig, DataConfig
from src.infrastructure.config import ConfigLoader as ConfigLoaderImport


class TestHardwareConfig:
    """Test cases for HardwareConfig."""

    def test_hardware_config_creation(self):
        """Test creating hardware configuration."""
        config = HardwareConfig(
            accelerator="cuda",
            devices=2,
            num_workers=4
        )

        assert config.accelerator == "cuda"
        assert config.devices == 2
        assert config.num_workers == 4

    def test_hardware_config_validation(self):
        """Test hardware config validation via Pydantic."""
        # Valid config should work
        config = HardwareConfig(accelerator="mps", devices=1, num_workers=2)
        assert config.accelerator == "mps"

        # Should allow assignment after creation
        config.accelerator = "cuda"
        assert config.accelerator == "cuda"


class TestDataConfig:
    """Test cases for DataConfig."""

    def test_data_config_creation(self):
        """Test creating data configuration."""
        config = DataConfig(
            dataset="wikipedia",
            split="train",
            text_field="text",
            streaming=True,
            tokenizer_name="gpt2",
            max_shards=1000,
            pack_sequences=True,
            train_docs=50000,
            val_docs=5000
        )

        assert config.dataset == "wikipedia"
        assert config.split == "train"
        assert config.streaming is True
        assert config.pack_sequences is True

    def test_data_config_optional_fields(self):
        """Test data config with None optional fields."""
        config = DataConfig(
            dataset="test",
            split="train",
            text_field="text",
            streaming=False,
            tokenizer_name="gpt2",
            max_shards=None,
            pack_sequences=False,
            train_docs=1000,
            val_docs=100
        )

        assert config.max_shards is None


class TestConfigLoader:
    """Test cases for ConfigLoader."""

    @pytest.fixture
    def sample_config_yaml(self):
        """Sample YAML configuration content."""
        return """
model:
  n_layers: 12
  d_model: 768
  n_heads: 12
  n_kv_heads: 2
  d_ff: 3072
  dropout: 0.1
  rope_theta: 10000.0
  tie_embeddings: true
  swa_window: 512

training:
  seq_len: 1024
  micro_batch_size: 8
  grad_accum_steps: 4
  max_steps: 1000
  eval_every: 100
  save_every: 200
  lr: 0.0001
  weight_decay: 0.1
  betas: [0.9, 0.95]
  eps: 1e-8
  warmup_ratio: 0.1
  precision: "32"
  seed: 42
  steps_per_epoch: 50
  gradient_clip_val: 1.0
  limit_val_batches: 1.0

hardware:
  accelerator: "cuda"
  devices: 2
  num_workers: 4

data:
  dataset: "wikipedia"
  split: "train"
  text_field: "text"
  streaming: true
  tokenizer_name: "gpt2"
  max_shards: 1000
  pack_sequences: true
  train_docs: 50000
  val_docs: 5000
"""

    def test_config_loader_creation(self, tmp_path):
        """Test ConfigLoader creation."""
        config_file = tmp_path / "config.yaml"
        loader = ConfigLoader(config_file)

        assert loader.config_path == config_file

    def test_load_yaml_file_not_found(self, tmp_path):
        """Test handling of missing config file."""
        config_file = tmp_path / "nonexistent.yaml"
        loader = ConfigLoader(config_file)

        with pytest.raises(FileNotFoundError):
            loader._load_yaml()

    def test_load_yaml_invalid_yaml(self, tmp_path, sample_config_yaml):
        """Test handling of invalid YAML."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")

        loader = ConfigLoader(config_file)

        with pytest.raises(ValueError):
            loader._load_yaml()

    def test_load_yaml_valid_config(self, tmp_path, sample_config_yaml):
        """Test loading valid YAML configuration."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(sample_config_yaml)

        loader = ConfigLoader(config_file)
        config = loader._load_yaml()

        assert config["model"]["n_layers"] == 12
        assert config["training"]["lr"] == 0.0001
        assert config["hardware"]["accelerator"] == "cuda"
        assert config["data"]["dataset"] == "wikipedia"

    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    def test_load_yaml_io_error(self, mock_yaml_load, mock_file, tmp_path):
        """Test handling of file I/O errors."""
        mock_file.side_effect = IOError("File read error")
        config_file = tmp_path / "config.yaml"
        loader = ConfigLoader(config_file)

        with pytest.raises(IOError):
            loader._load_yaml()

    def test_validate_config_file_exists(self, tmp_path, sample_config_yaml):
        """Test config file validation when file exists."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(sample_config_yaml)

        loader = ConfigLoader(config_file)
        assert loader.validate_config_file() is True

    def test_validate_config_file_not_exists(self, tmp_path):
        """Test config file validation when file doesn't exist."""
        config_file = tmp_path / "nonexistent.yaml"
        loader = ConfigLoader(config_file)

        assert loader.validate_config_file() is False

    def test_get_config_schema(self, tmp_path):
        """Test configuration schema retrieval."""
        config_file = tmp_path / "config.yaml"
        loader = ConfigLoader(config_file)

        schema = loader.get_config_schema()

        assert "model" in schema
        assert "training" in schema
        assert "hardware" in schema
        assert "data" in schema

        # Check that schema contains expected fields
        assert "n_layers" in schema["model"]
        assert "lr" in schema["training"]
        assert "accelerator" in schema["hardware"]
        assert "dataset" in schema["data"]

    def test_load_all_configs(self, tmp_path, sample_config_yaml):
        """Test loading all configurations at once."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(sample_config_yaml)

        loader = ConfigLoader(config_file)
        configs = loader.load_all_configs()

        assert "model" in configs
        assert "training" in configs
        assert "hardware" in configs
        assert "data" in configs

        # Check types
        from src.domain.entities.model_config import ModelConfiguration
        from src.domain.entities.training_config import TrainingConfiguration

        assert isinstance(configs["model"], ModelConfiguration)
        assert isinstance(configs["training"], TrainingConfiguration)
        assert isinstance(configs["hardware"], HardwareConfig)
        assert isinstance(configs["data"], DataConfig)

    def test_load_model_config(self, tmp_path, sample_config_yaml):
        """Test loading model configuration."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(sample_config_yaml)

        loader = ConfigLoader(config_file)
        model_config = loader.load_model_config()

        assert model_config.n_layers == 12
        assert model_config.d_model == 768
        assert model_config.tie_embeddings is True

    def test_load_training_config(self, tmp_path, sample_config_yaml):
        """Test loading training configuration."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(sample_config_yaml)

        loader = ConfigLoader(config_file)
        training_config = loader.load_training_config()

        assert training_config.seq_len == 1024
        assert training_config.lr == 0.0001
        assert training_config.precision == "32"

    def test_load_hardware_config(self, tmp_path, sample_config_yaml):
        """Test loading hardware configuration."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(sample_config_yaml)

        loader = ConfigLoader(config_file)
        hardware_config = loader.load_hardware_config()

        assert hardware_config.accelerator == "cuda"
        assert hardware_config.devices == 2
        assert hardware_config.num_workers == 4

    def test_load_data_config(self, tmp_path, sample_config_yaml):
        """Test loading data configuration."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(sample_config_yaml)

        loader = ConfigLoader(config_file)
        data_config = loader.load_data_config()

        assert data_config.dataset == "wikipedia"
        assert data_config.streaming is True
        assert data_config.train_docs == 50000
