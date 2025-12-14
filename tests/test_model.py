import torch
import pytest
import torch.nn.functional as F
from model import GPTMini, GPTConfig, build_rope_cache, apply_rope

@pytest.fixture
def tiny_config():
    return GPTConfig(
        vocab_size=100,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        dropout=0.0,
        rope_theta=100.0,
        tie_embeddings=True
    )

def test_rope_cache_construction():
    seq_len = 10
    head_dim = 16
    theta = 100.0
    device = torch.device('cpu')
    dtype = torch.float32
    
    sin, cos = build_rope_cache(seq_len, head_dim, theta, device, dtype)
    
    assert sin.shape == (seq_len, head_dim // 2)
    assert cos.shape == (seq_len, head_dim // 2)
    assert sin.dtype == dtype
    assert cos.dtype == dtype

def test_rope_application():
    B, T, H, D = 2, 5, 4, 16 
    x = torch.randn(B, T, H, D)
    sin, cos = build_rope_cache(T, D, 100.0, x.device, torch.float32)
    
    out = apply_rope(x, sin, cos)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

def test_gpt_mini_initialization(tiny_config):
    model = GPTMini(tiny_config)
    assert isinstance(model, torch.nn.Module)
    assert len(model.blocks) == tiny_config.n_layers

def test_gpt_mini_forward(tiny_config):
    model = GPTMini(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 8)) # B=2, T=8
    logits = model(input_ids)
    
    assert logits.shape == (2, 8, tiny_config.vocab_size)

def test_gpt_mini_generate(tiny_config):
    model = GPTMini(tiny_config)
    input_ids = torch.tensor([[1, 2, 3]]) # B=1, T=3
    
    # Generate 5 new tokens
    out = model.generate(input_ids, max_new_tokens=5, temperature=0.0)
    assert out.shape == (1, 8) # 3 + 5
    
    # Check that input prefix is preserved
    assert torch.equal(out[:, :3], input_ids)

def test_kv_cache_logic(tiny_config):
    # Test that cached generation produces same result as non-cached forward for single steps (conceptually)
    # This is a basic smoke test, correctness of numerical outputs usually requires stricter tolerances
    model = GPTMini(tiny_config)
    model.eval()
    
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 5))
    
    # Run pure forward
    with torch.no_grad():
        logits_full = model(input_ids)
        last_logit_full = logits_full[:, -1, :]
    
    # Run generation step-by-step (which uses cache internally)
    # We can't easily expose the cache state from outside without hacking, 
    # but we can check if generate runs without error and produces valid output.
    # A true equivalence test requires instrumenting the forward pass.
    out = model.generate(input_ids[:, :-1], max_new_tokens=1, temperature=0.0)
    assert out.shape == (1, 5)


def test_gradient_checkpointing_matches_no_checkpointing():
    torch.manual_seed(0)
    vocab_size = 97
    base_cfg = GPTConfig(
        vocab_size=vocab_size,
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        dropout=0.0,
        rope_theta=100.0,
        tie_embeddings=True,
        gradient_checkpointing=False,
    )
    ckpt_cfg = GPTConfig(**{**base_cfg.__dict__, "gradient_checkpointing": True})

    model_no = GPTMini(base_cfg)
    model_ckpt = GPTMini(ckpt_cfg)
    model_ckpt.load_state_dict(model_no.state_dict())

    model_no.train()
    model_ckpt.train()

    input_ids = torch.randint(0, vocab_size, (2, 8))

    # Causal next-token loss (shifted).
    logits_no = model_no(input_ids)
    loss_no = F.cross_entropy(
        logits_no[:, :-1, :].reshape(-1, vocab_size),
        input_ids[:, 1:].reshape(-1),
    )
    loss_no.backward()

    logits_ckpt = model_ckpt(input_ids)
    loss_ckpt = F.cross_entropy(
        logits_ckpt[:, :-1, :].reshape(-1, vocab_size),
        input_ids[:, 1:].reshape(-1),
    )
    loss_ckpt.backward()

    # Spot-check a few gradients for exact match.
    pairs = [
        ("tok_emb.weight", model_no.tok_emb.weight.grad, model_ckpt.tok_emb.weight.grad),
        ("blocks.0.attn.out.weight", model_no.blocks[0].attn.out.weight.grad, model_ckpt.blocks[0].attn.out.weight.grad),
        ("blocks.1.mlp.w3.weight", model_no.blocks[1].mlp.w3.weight.grad, model_ckpt.blocks[1].mlp.w3.weight.grad),
    ]
    for name, g0, g1 in pairs:
        assert g0 is not None, f"missing grad for {name} (no-ckpt)"
        assert g1 is not None, f"missing grad for {name} (ckpt)"
        assert torch.allclose(g0, g1, atol=0.0, rtol=0.0), f"grad mismatch for {name}"
