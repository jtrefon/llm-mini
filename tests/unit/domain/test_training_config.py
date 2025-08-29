"""Tests for training configuration entity."""
import pytest
from src.domain.entities.training_config import TrainingConfiguration


class TestTrainingConfiguration:
    """Test cases for TrainingConfiguration entity."""

    def test_valid_configuration_creation(self):
        """Test creating a valid training configuration."""
        config = TrainingConfiguration(
            seq_len=1024,
            micro_batch_size=8,
            grad_accum_steps=4,
            max_steps=1000,
            eval_every=100,
            save_every=200,
            lr=1e-4,
            weight_decay=0.1,
            betas=[0.9, 0.95],
            eps=1e-8,
            warmup_ratio=0.1,
            precision="32",
            seed=42,
            steps_per_epoch=50,
            gradient_clip_val=1.0,
            limit_val_batches=0.5
        )

        assert config.seq_len == 1024
        assert config.micro_batch_size == 8
        assert config.grad_accum_steps == 4
        assert config.max_steps == 1000
        assert config.lr == 1e-4
        assert config.weight_decay == 0.1
        assert config.betas == [0.9, 0.95]
        assert config.warmup_ratio == 0.1
        assert config.precision == "32"
        assert config.seed == 42

    def test_effective_batch_size_calculation(self):
        """Test effective batch size calculation."""
        config = TrainingConfiguration(
            seq_len=512, micro_batch_size=4, grad_accum_steps=8,
            max_steps=1000, eval_every=100, save_every=200,
            lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
            eps=1e-8, warmup_ratio=0.1, precision="32",
            seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
            limit_val_batches=1.0
        )

        assert config.effective_batch_size == 32  # 4 * 8

    def test_warmup_steps_calculation(self):
        """Test warmup steps calculation."""
        config = TrainingConfiguration(
            seq_len=512, micro_batch_size=4, grad_accum_steps=8,
            max_steps=1000, eval_every=100, save_every=200,
            lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
            eps=1e-8, warmup_ratio=0.1, precision="32",
            seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
            limit_val_batches=1.0
        )

        assert config.warmup_steps == 100  # 0.1 * 1000

    def test_mixed_precision_detection(self):
        """Test mixed precision detection."""
        # Standard precision
        config_32 = TrainingConfiguration(
            seq_len=512, micro_batch_size=4, grad_accum_steps=8,
            max_steps=1000, eval_every=100, save_every=200,
            lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
            eps=1e-8, warmup_ratio=0.1, precision="32",
            seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
            limit_val_batches=1.0
        )
        assert config_32.is_mixed_precision is False

        # Mixed precision
        config_16 = TrainingConfiguration(
            seq_len=512, micro_batch_size=4, grad_accum_steps=8,
            max_steps=1000, eval_every=100, save_every=200,
            lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
            eps=1e-8, warmup_ratio=0.1, precision="16-mixed",
            seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
            limit_val_batches=1.0
        )
        assert config_16.is_mixed_precision is True

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        config = TrainingConfiguration(
            seq_len=512, micro_batch_size=4, grad_accum_steps=8,
            max_steps=1000, eval_every=100, save_every=200,
            lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
            eps=1e-8, warmup_ratio=0.1, precision="32",
            seed=42, steps_per_epoch=50, gradient_clip_val=1.0,
            limit_val_batches=0.5
        )

        config_dict = config.to_dict()

        assert config_dict['seq_len'] == 512
        assert config_dict['micro_batch_size'] == 4
        assert config_dict['lr'] == 1e-4
        assert config_dict['betas'] == [0.9, 0.95]
        assert config_dict['precision'] == "32"
        assert config_dict['steps_per_epoch'] == 50

    @pytest.mark.parametrize("invalid_lr", [-1e-4, 0])
    def test_invalid_learning_rate(self, invalid_lr):
        """Test validation of learning rate."""
        with pytest.raises(ValueError, match="learning rate must be positive"):
            TrainingConfiguration(
                seq_len=512, micro_batch_size=4, grad_accum_steps=8,
                max_steps=1000, eval_every=100, save_every=200,
                lr=invalid_lr, weight_decay=0.1, betas=[0.9, 0.95],
                eps=1e-8, warmup_ratio=0.1, precision="32",
                seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
                limit_val_batches=1.0
            )

    @pytest.mark.parametrize("invalid_warmup_ratio", [-0.1, 1.1])
    def test_invalid_warmup_ratio_range(self, invalid_warmup_ratio):
        """Test validation of warmup ratio range."""
        with pytest.raises(ValueError, match="warmup_ratio must be between 0 and 1"):
            TrainingConfiguration(
                seq_len=512, micro_batch_size=4, grad_accum_steps=8,
                max_steps=1000, eval_every=100, save_every=200,
                lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
                eps=1e-8, warmup_ratio=invalid_warmup_ratio, precision="32",
                seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
                limit_val_batches=1.0
            )

    @pytest.mark.parametrize("invalid_limit_batches", [-0.1, 1.1])
    def test_invalid_limit_val_batches_range(self, invalid_limit_batches):
        """Test validation of limit_val_batches range."""
        with pytest.raises(ValueError, match="limit_val_batches must be between 0 and 1"):
            TrainingConfiguration(
                seq_len=512, micro_batch_size=4, grad_accum_steps=8,
                max_steps=1000, eval_every=100, save_every=200,
                lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
                eps=1e-8, warmup_ratio=0.1, precision="32",
                seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
                limit_val_batches=invalid_limit_batches
            )

    @pytest.mark.parametrize("invalid_betas", [[0.9], [0.9, 0.95, 0.99], []])
    def test_invalid_betas_length(self, invalid_betas):
        """Test validation of betas length."""
        with pytest.raises(ValueError, match="betas must contain exactly 2 values"):
            TrainingConfiguration(
                seq_len=512, micro_batch_size=4, grad_accum_steps=8,
                max_steps=1000, eval_every=100, save_every=200,
                lr=1e-4, weight_decay=0.1, betas=invalid_betas,
                eps=1e-8, warmup_ratio=0.1, precision="32",
                seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
                limit_val_batches=1.0
            )

    @pytest.mark.parametrize("invalid_beta", [-0.1, 1.1])
    def test_invalid_beta_range(self, invalid_beta):
        """Test validation of beta range."""
        with pytest.raises(ValueError, match="beta values must be between 0 and 1"):
            TrainingConfiguration(
                seq_len=512, micro_batch_size=4, grad_accum_steps=8,
                max_steps=1000, eval_every=100, save_every=200,
                lr=1e-4, weight_decay=0.1, betas=[0.9, invalid_beta],
                eps=1e-8, warmup_ratio=0.1, precision="32",
                seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
                limit_val_batches=1.0
            )

    @pytest.mark.parametrize("invalid_precision", ["8", "64", "mixed"])
    def test_invalid_precision(self, invalid_precision):
        """Test validation of precision values."""
        with pytest.raises(ValueError, match="precision must be one of"):
            TrainingConfiguration(
                seq_len=512, micro_batch_size=4, grad_accum_steps=8,
                max_steps=1000, eval_every=100, save_every=200,
                lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
                eps=1e-8, warmup_ratio=0.1, precision=invalid_precision,
                seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
                limit_val_batches=1.0
            )

    @pytest.mark.parametrize("invalid_weight_decay", [-0.1])
    def test_invalid_weight_decay(self, invalid_weight_decay):
        """Test validation of weight decay."""
        with pytest.raises(ValueError, match="weight_decay must be non-negative"):
            TrainingConfiguration(
                seq_len=512, micro_batch_size=4, grad_accum_steps=8,
                max_steps=1000, eval_every=100, save_every=200,
                lr=1e-4, weight_decay=invalid_weight_decay, betas=[0.9, 0.95],
                eps=1e-8, warmup_ratio=0.1, precision="32",
                seed=42, steps_per_epoch=None, gradient_clip_val=1.0,
                limit_val_batches=1.0
            )

    @pytest.mark.parametrize("invalid_grad_clip", [-1.0, 0])
    def test_invalid_gradient_clip_val(self, invalid_grad_clip):
        """Test validation of gradient clip value."""
        with pytest.raises(ValueError, match="gradient_clip_val must be positive"):
            TrainingConfiguration(
                seq_len=512, micro_batch_size=4, grad_accum_steps=8,
                max_steps=1000, eval_every=100, save_every=200,
                lr=1e-4, weight_decay=0.1, betas=[0.9, 0.95],
                eps=1e-8, warmup_ratio=0.1, precision="32",
                seed=42, steps_per_epoch=None, gradient_clip_val=invalid_grad_clip,
                limit_val_batches=1.0
            )
