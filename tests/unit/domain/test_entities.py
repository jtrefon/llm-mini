"""Tests for domain entities."""
import pytest
from src.domain.entities.model_config import ModelConfiguration
from src.domain.entities.training_config import TrainingConfiguration


class TestModelConfiguration:
    """Test cases for ModelConfiguration entity."""

    def test_valid_configuration_creation(self):
        """Test creating a valid model configuration."""
        config = ModelConfiguration(
            n_layers=12,
            d_model=768,
            n_heads=12,
            n_kv_heads=2,
            d_ff=3072,
            dropout=0.1,
            vocab_size=30000,
            rope_theta=10000.0,
            tie_embeddings=True,
            swa_window=512
        )

        assert config.n_layers == 12
        assert config.d_model == 768
        assert config.n_heads == 12
        assert config.n_kv_heads == 2
        assert config.d_ff == 3072
        assert config.dropout == 0.1
        assert config.vocab_size == 30000
        assert config.rope_theta == 10000.0
        assert config.tie_embeddings is True
        assert config.swa_window == 512

    def test_head_dimension_calculation(self):
        """Test head dimension property."""
        config = ModelConfiguration(
            n_layers=6,
            d_model=768,
            n_heads=12,
            n_kv_heads=2,
            d_ff=3072,
            dropout=0.0,
            vocab_size=30000,
            rope_theta=10000.0,
            tie_embeddings=True,
            swa_window=0
        )

        assert config.head_dim == 64  # 768 / 12

    def test_gqa_detection(self):
        """Test GQA enabled detection."""
        # GQA enabled (n_kv_heads != n_heads)
        gqa_config = ModelConfiguration(
            n_layers=6, d_model=512, n_heads=8, n_kv_heads=4,
            d_ff=2048, dropout=0.0, vocab_size=30000,
            rope_theta=10000.0, tie_embeddings=True, swa_window=0
        )
        assert gqa_config.is_gqa_enabled is True
        assert gqa_config.gqa_group_size == 2  # 8 / 4

        # GQA disabled (n_kv_heads == n_heads)
        standard_config = ModelConfiguration(
            n_layers=6, d_model=512, n_heads=8, n_kv_heads=8,
            d_ff=2048, dropout=0.0, vocab_size=30000,
            rope_theta=10000.0, tie_embeddings=True, swa_window=0
        )
        assert standard_config.is_gqa_enabled is False
        assert standard_config.gqa_group_size == 1

    def test_none_vocab_size(self):
        """Test handling of None vocab_size."""
        config = ModelConfiguration(
            n_layers=6, d_model=512, n_heads=8, n_kv_heads=8,
            d_ff=2048, dropout=0.0, vocab_size=None,
            rope_theta=10000.0, tie_embeddings=True, swa_window=0
        )

        assert config.vocab_size is None

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        config = ModelConfiguration(
            n_layers=6, d_model=512, n_heads=8, n_kv_heads=4,
            d_ff=2048, dropout=0.1, vocab_size=30000,
            rope_theta=10000.0, tie_embeddings=True, swa_window=512
        )

        config_dict = config.to_dict()

        assert config_dict['n_layers'] == 6
        assert config_dict['d_model'] == 512
        assert config_dict['n_heads'] == 8
        assert config_dict['n_kv_heads'] == 4
        assert config_dict['vocab_size'] == 30000
        assert config_dict['dropout'] == 0.1
        assert config_dict['rope_theta'] == 10000.0
        assert config_dict['tie_embeddings'] is True
        assert config_dict['swa_window'] == 512

    @pytest.mark.parametrize("invalid_vocab_size", [-1, 0])
    def test_invalid_vocab_size(self, invalid_vocab_size):
        """Test validation of vocab_size."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            ModelConfiguration(
                n_layers=6, d_model=512, n_heads=8, n_kv_heads=8,
                d_ff=2048, dropout=0.0, vocab_size=invalid_vocab_size,
                rope_theta=10000.0, tie_embeddings=True, swa_window=0
            )

    @pytest.mark.parametrize("invalid_kv_heads,invalid_heads", [(4, 2), (3, 2)])
    def test_invalid_gqa_configuration(self, invalid_kv_heads, invalid_heads):
        """Test validation of GQA configuration."""
        with pytest.raises(ValueError, match="n_kv_heads cannot exceed n_heads"):
            ModelConfiguration(
                n_layers=6, d_model=512, n_heads=invalid_heads, n_kv_heads=invalid_kv_heads,
                d_ff=2048, dropout=0.0, vocab_size=30000,
                rope_theta=10000.0, tie_embeddings=True, swa_window=0
            )

    @pytest.mark.parametrize("invalid_param,invalid_value", [
        ("n_layers", 0), ("n_layers", -1),
        ("d_model", 0), ("d_model", -1),
        ("n_heads", 0), ("n_heads", -1),
        ("rope_theta", 0), ("rope_theta", -1)
    ])
    def test_invalid_positive_parameters(self, invalid_param, invalid_value):
        """Test validation of positive parameters."""
        config_data = {
            'n_layers': 6, 'd_model': 512, 'n_heads': 8, 'n_kv_heads': 8,
            'd_ff': 2048, 'dropout': 0.0, 'vocab_size': 30000,
            'rope_theta': 10000.0, 'tie_embeddings': True, 'swa_window': 0
        }
        config_data[invalid_param] = invalid_value

        with pytest.raises(ValueError, match="must be positive"):
            ModelConfiguration(**config_data)

    @pytest.mark.parametrize("invalid_dropout", [-0.1, 1.1])
    def test_invalid_dropout_range(self, invalid_dropout):
        """Test validation of dropout range."""
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            ModelConfiguration(
                n_layers=6, d_model=512, n_heads=8, n_kv_heads=8,
                d_ff=2048, dropout=invalid_dropout, vocab_size=30000,
                rope_theta=10000.0, tie_embeddings=True, swa_window=0
            )
