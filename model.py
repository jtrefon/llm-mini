import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- RMSNorm (lighter than LayerNorm; used by Llama/Mistral)
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


# ---- RoPE utilities (rotary positional embeddings)
# Applies RoPE to q, k in-place-ish (returns new tensors).
# Reference idea: rotate pairs of dims with sin/cos frequencies.

def build_rope_cache(seq_len: int, head_dim: int, base_theta: float, device, dtype):
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    # Frequencies as in Llama: base^(2i/d)
    inv_freq = 1.0 / (base_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum('i,j->ij', t, inv_freq)  # [T, head_dim/2]
    sin = freqs.sin()
    cos = freqs.cos()
    return sin, cos


def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """
    x: [B, T, H, D]
    sin, cos: [T, D/2]
    """
    B, T, H, D = x.shape
    x = x.view(B, T, H, D // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    # Broadcast sin/cos: [T, D/2] -> [1, T, 1, D/2]
    sin_ = sin.view(1, T, 1, -1)
    cos_ = cos.view(1, T, 1, -1)
    # (x1, x2) rotated
    out1 = x1 * cos_ - x2 * sin_
    out2 = x1 * sin_ + x2 * cos_
    return torch.stack((out1, out2), dim=-1).view(B, T, H, D)


# ---- SwiGLU MLP (PaLM-style)
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ---- Multi-Head Attention with Grouped-Query Attention (GQA)
class GQAMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads for GQA"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.group_size = n_heads // n_kv_heads

        self.Wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal=True, attn_mask=None, rope_sin=None, rope_cos=None, swa_window: int = 0):
        # x: [B, T, C]
        B, T, C = x.size()
        q = self.Wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.Wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.Wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        if rope_sin is not None and rope_cos is not None:
            q = apply_rope(q, rope_sin, rope_cos)
            k = apply_rope(k, rope_sin, rope_cos)

        # Expand k,v from KV heads to Q heads (repeat per group)
        if self.n_kv_heads != self.n_heads:
            # [B,T,H_kv,D] -> repeat heads
            k = k.repeat_interleave(self.group_size, dim=2)
            v = v.repeat_interleave(self.group_size, dim=2)

        # SDPA expects [B*H, T, D]
        q = q.permute(0, 2, 1, 3).reshape(B * self.n_heads, T, self.head_dim)
        k = k.permute(0, 2, 1, 3).reshape(B * self.n_heads, T, self.head_dim)
        v = v.permute(0, 2, 1, 3).reshape(B * self.n_heads, T, self.head_dim)

        # Build causal/sliding window mask if needed. PyTorch SDPA can use attn_mask of shape [T, T] or [B*H, T, T]
        attn_mask_t = None
        if causal or swa_window > 0:
            # Causal base mask
            causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
            if swa_window > 0:
                # Allow attention only to the last `swa_window` tokens (including self)
                sw = torch.ones(T, T, device=x.device, dtype=torch.bool)
                idx = torch.arange(T, device=x.device)
                sw &= (idx.view(-1,1) - idx.view(1,-1)) > 0  # start with strictly future
                # Now, block positions older than window
                older_than_w = (idx.view(-1,1) - idx.view(1,-1)) > swa_window
                # Final mask = causal OR too-old
                attn_mask_t = causal_mask | older_than_w
            else:
                attn_mask_t = causal_mask

        try:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask_t, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False)
        except Exception:
            # Fallback manual attention (should rarely trigger)
            d = q.size(-1)
            scores = q @ k.transpose(-2, -1) / math.sqrt(d)
            if attn_mask_t is not None:
                scores = scores.masked_fill(attn_mask_t, float('-inf'))
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn = attn @ v

        attn = attn.reshape(B, self.n_heads, T, self.head_dim).permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return self.out(attn)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GQAMultiheadAttention(d_model, n_heads, n_kv_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope_sin, rope_cos, swa_window: int = 0):
        a = self.attn(self.norm1(x), causal=True, rope_sin=rope_sin, rope_cos=rope_cos, swa_window=swa_window)
        x = x + self.dropout(a)
        m = self.mlp(self.norm2(x))
        x = x + self.dropout(m)
        return x


@dataclass
class GPTConfig:
    vocab_size: int
    n_layers: int = 16
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 2
    d_ff: int = 3072
    dropout: float = 0.0
    rope_theta: float = 1000000.0
    tie_embeddings: bool = True
    swa_window: int = 0


class GPTMini(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.n_kv_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.register_buffer("_rope_sin", None, persistent=False)
        self.register_buffer("_rope_cos", None, persistent=False)
        self._rope_cache_params = (0, 0, 0.0)  # (seq_len, head_dim, theta)

    def _maybe_build_rope(self, seq_len: int, head_dim: int, theta: float, device, dtype):
        if self._rope_sin is None:
            sin, cos = build_rope_cache(seq_len, head_dim, theta, device, dtype)
            self._rope_sin = sin
            self._rope_cos = cos
            self._rope_cache_params = (seq_len, head_dim, theta)
        else:
            cached = self._rope_cache_params
            if cached != (seq_len, head_dim, theta):
                sin, cos = build_rope_cache(seq_len, head_dim, theta, device, dtype)
                self._rope_sin = sin
                self._rope_cos = cos
                self._rope_cache_params = (seq_len, head_dim, theta)

    def forward(self, input_ids: torch.Tensor):
        # input_ids: [B, T]
        B, T = input_ids.shape
        h = self.tok_emb(input_ids)  # [B,T,C]
        head_dim = self.config.d_model // self.config.n_heads
        self._maybe_build_rope(T, head_dim, self.config.rope_theta, h.device, h.dtype)
        for blk in self.blocks:
            h = blk(h, self._rope_sin, self._rope_cos, swa_window=self.config.swa_window)
        h = self.norm_f(h)
        logits = self.lm_head(h)
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens=128, temperature=0.8, top_p=0.9, eos_token_id: Optional[int]=None):
        self.eval()
        B = input_ids.size(0)
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)[:, -1, :]  # [B, vocab]
            logits = logits / max(1e-8, temperature)
            # nucleus sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum - sorted_probs > top_p
            sorted_probs[mask] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_idx = torch.multinomial(sorted_probs, num_samples=1)  # [B,1]
            next_token = sorted_idx.gather(-1, next_idx)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break
        return input_ids
