"""Tests for domain models."""
import pytest
import torch
import math
from unittest.mock import patch

from src.domain.models.gpt_mini import GPTMini, RMSNorm, SwiGLU, GQAMultiheadAttention
from src.domain.entities.model_config import ModelConfiguration


class TestRMSNorm:
    """Test cases for RMSNorm implementation."""

    def test_rms_norm_basic(self):
        """Test basic RMSNorm functionality."""
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 10, 64)  # [B, T, C]

        output = norm(x)

        assert output.shape == x.shape
        assert torch.allclose(output.mean(dim=-1), torch.zeros_like(output.mean(dim=-1)), atol=1e-6)

    def test_rms_norm_weight_application(self):
        """Test that weight is properly applied."""
        norm = RMSNorm(d_model=64)
        x = torch.randn(2, 10, 64)

        # Set weight to 2.0
        norm.weight.data.fill_(2.0)
        output = norm(x)

        # Without weight, RMS would be 1, with weight 2.0 it should be 2.0
        expected_rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True))
        expected_output = 2.0 * x / expected_rms

        assert torch.allclose(output, expected_output, atol=1e-5)


class TestSwiGLU:
    """Test cases for SwiGLU MLP implementation."""

    def test_swiglu_forward(self):
        """Test SwiGLU forward pass."""
        swiglu = SwiGLU(d_model=64, d_ff=256)
        x = torch.randn(2, 10, 64)

        output = swiglu(x)

        assert output.shape == (2, 10, 64)  # Should maintain input dimensions

    def test_swiglu_components(self):
        """Test that SwiGLU uses correct components."""
        swiglu = SwiGLU(d_model=64, d_ff=128)

        # Check layer dimensions
        assert swiglu.w1.in_features == 64
        assert swiglu.w1.out_features == 128
        assert swiglu.w2.in_features == 64
        assert swiglu.w2.out_features == 128
        assert swiglu.w3.in_features == 128
        assert swiglu.w3.out_features == 64


class TestGQAMultiheadAttention:
    """Test cases for GQA Multi-Head Attention."""

    def test_gqa_creation(self):
        """Test GQA attention layer creation."""
        attn = GQAMultiheadAttention(
            d_model=64, n_heads=8, n_kv_heads=4, dropout=0.1
        )

        assert attn.d_model == 64
        assert attn.n_heads == 8
        assert attn.n_kv_heads == 4
        assert attn.head_dim == 8  # 64 / 8
        assert attn.group_size == 2  # 8 / 4

    def test_gqa_standard_attention(self):
        """Test standard attention (n_kv_heads == n_heads)."""
        attn = GQAMultiheadAttention(
            d_model=64, n_heads=8, n_kv_heads=8, dropout=0.0
        )

        x = torch.randn(2, 10, 64)
        output = attn(x)

        assert output.shape == x.shape
        assert attn.group_size == 1  # No grouping

    def test_gqa_grouped_attention(self):
        """Test grouped attention (n_kv_heads < n_heads)."""
        attn = GQAMultiheadAttention(
            d_model=64, n_heads=8, n_kv_heads=4, dropout=0.0
        )

        x = torch.randn(2, 10, 64)
        output = attn(x)

        assert output.shape == x.shape
        assert attn.group_size == 2  # 8 / 4 groups

    @patch('src.domain.models.gpt_mini.apply_rope')
    def test_rope_integration(self, mock_apply_rope):
        """Test RoPE integration in attention."""
        attn = GQAMultiheadAttention(
            d_model=64, n_heads=8, n_kv_heads=8, dropout=0.0
        )

        x = torch.randn(2, 10, 64)
        sin = torch.randn(10, 4)  # [T, D/2]
        cos = torch.randn(10, 4)

        mock_apply_rope.return_value = x  # Return input unchanged

        output = attn(x, rope_sin=sin, rope_cos=cos)

        # Verify RoPE was called
        assert mock_apply_rope.call_count == 2  # Once for q, once for k


class TestGPTMini:
    """Test cases for GPT Mini model."""

    def test_model_creation(self, sample_model_config):
        """Test GPT model creation with configuration."""
        model = GPTMini(sample_model_config)

        assert model.config == sample_model_config
        assert len(model.blocks) == sample_model_config.n_layers
        assert model.tok_emb.num_embeddings == sample_model_config.vocab_size
        assert model.lm_head.out_features == sample_model_config.vocab_size

    def test_model_forward(self, sample_model_config):
        """Test model forward pass."""
        model = GPTMini(sample_model_config)
        input_ids = torch.randint(0, sample_model_config.vocab_size, (2, 20))

        logits = model(input_ids)

        expected_shape = (2, 20, sample_model_config.vocab_size)
        assert logits.shape == expected_shape

    def test_model_tie_embeddings(self, sample_model_config):
        """Test embedding tying."""
        # Test with tie_embeddings=True
        model_tied = GPTMini(sample_model_config)
        assert model_tied.lm_head.weight is model_tied.tok_emb.weight

        # Test with tie_embeddings=False
        config_no_tie = sample_model_config
        config_no_tie.tie_embeddings = False
        model_no_tie = GPTMini(config_no_tie)
        assert model_no_tie.lm_head.weight is not model_no_tie.tok_emb.weight

    def test_rope_cache_building(self, sample_model_config):
        """Test RoPE cache building and reuse."""
        model = GPTMini(sample_model_config)
        seq_len = 50

        # First call should build cache
        input_ids = torch.randint(0, sample_model_config.vocab_size, (1, seq_len))
        _ = model(input_ids)

        # Check cache was built
        assert model._rope_sin is not None
        assert model._rope_cos is not None
        assert model._rope_cache_params[0] == seq_len

        # Second call with same seq_len should reuse cache
        input_ids2 = torch.randint(0, sample_model_config.vocab_size, (1, seq_len))
        _ = model(input_ids2)

        # Cache should still be the same
        assert model._rope_cache_params[0] == seq_len

    def test_sliding_window_attention(self, sample_model_config):
        """Test sliding window attention."""
        config_swa = sample_model_config
        config_swa.swa_window = 512
        model = GPTMini(config_swa)

        input_ids = torch.randint(0, config_swa.vocab_size, (1, 100))
        logits = model(input_ids)

        assert logits.shape[1] == 100  # Sequence length preserved

    def test_model_generate(self, sample_model_config):
        """Test text generation."""
        model = GPTMini(sample_model_config)
        input_ids = torch.randint(0, sample_model_config.vocab_size, (1, 5))

        generated = model.generate(input_ids, max_new_tokens=10)

        assert generated.shape[0] == 1  # Same batch size
        assert generated.shape[1] == 15  # Original 5 + 10 new tokens

    def test_generation_temperature(self, sample_model_config):
        """Test generation with different temperatures."""
        model = GPTMini(sample_model_config)
        input_ids = torch.randint(0, sample_model_config.vocab_size, (1, 5))

        # High temperature (more random)
        generated_hot = model.generate(input_ids, max_new_tokens=5, temperature=2.0, seed=42)

        # Low temperature (more deterministic)
        generated_cold = model.generate(input_ids, max_new_tokens=5, temperature=0.1, seed=42)

        # Results should be different due to temperature
        assert not torch.equal(generated_hot, generated_cold)

    def test_generation_nucleus_sampling(self, sample_model_config):
        """Test nucleus sampling in generation."""
        model = GPTMini(sample_model_config)
        input_ids = torch.randint(0, sample_model_config.vocab_size, (1, 5))

        # Generate with different top_p values
        generated_narrow = model.generate(input_ids, max_new_tokens=5, top_p=0.1)
        generated_wide = model.generate(input_ids, max_new_tokens=5, top_p=0.9)

        assert generated_narrow.shape == generated_wide.shape

    def test_generation_early_stopping(self, sample_model_config):
        """Test early stopping on EOS token."""
        model = GPTMini(sample_model_config)
        eos_token_id = 2  # Some EOS token
        input_ids = torch.tensor([[1, 1, 1, eos_token_id, 1]])  # EOS in middle

        generated = model.generate(input_ids, max_new_tokens=10, eos_token_id=eos_token_id)

        # Should stop at EOS token
        eos_positions = (generated == eos_token_id).nonzero()
        if eos_positions.numel() > 0:
            # If EOS was found, check we didn't generate beyond it
            first_eos_pos = eos_positions[0, 1].item()
            assert generated.shape[1] <= first_eos_pos + 1
