import math
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---- RMSNorm (lighter than LayerNorm)
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Stabilizes training by normalizing the input using the root mean square
    of its values. Lighter than LayerNorm as it does not re-center the mean.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMSNorm to the input tensor.

        Args:
            x: Input tensor of shape [B, T, C].

        Returns:
            Normalized tensor of the same shape.
        """
        # x: [B, T, C]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


# ---- RoPE utilities (rotary positional embeddings)
def build_rope_cache(
    seq_len: int, head_dim: int, base_theta: float, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-computes sine and cosine frequencies for RoPE.

    Args:
        seq_len: Maximum sequence length.
        head_dim: Dimension of each attention head.
        base_theta: Base period for the frequencies.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        Tuple of (sin, cos) tensors of shape [seq_len, head_dim/2].
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    # Frequencies as in Llama: base^(2i/d)
    inv_freq = 1.0 / (
        base_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, head_dim/2]
    sin = freqs.sin()
    cos = freqs.cos()
    return sin, cos


def apply_rope(
    x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, pos_offset: int = 0
) -> torch.Tensor:
    """Applies Rotary Positional Embeddings to input tensor.

    Args:
        x: Input tensor [B, T, H, D].
        sin: Pre-computed sine tensor [T, D/2].
        cos: Pre-computed cosine tensor [T, D/2].
        pos_offset: Offset for incremental decoding (default: 0).

    Returns:
        Rotated tensor of shape [B, T, H, D].
    """
    # x: [B, T, H, D]
    # sin, cos: [T, D/2]
    B, T, H, D = x.shape
    # Compute RoPE in float32 for numerical stability, then cast back
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x.view(B, T, H, D // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]

    # Broadcast sin/cos: [T, D/2] -> [1, T, 1, D/2]
    # Support positional offset for incremental decoding
    sin_slice = sin[pos_offset : pos_offset + T]
    cos_slice = cos[pos_offset : pos_offset + T]
    sin_ = sin_slice.to(torch.float32).view(1, T, 1, -1)
    cos_ = cos_slice.to(torch.float32).view(1, T, 1, -1)

    # (x1, x2) rotated
    out1 = x1 * cos_ - x2 * sin_
    out2 = x1 * sin_ + x2 * cos_
    out = torch.stack((out1, out2), dim=-1).view(B, T, H, D)
    return out.to(orig_dtype)


# ---- SwiGLU MLP (PaLM-style)
class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.

    Implements the gated linear unit with Swish activation, popular in PaLM/Llama.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ---- Multi-Head Attention with Grouped-Query Attention (GQA)
class GQAMultiheadAttention(nn.Module):
    """Grouped-Query Attention (GQA).

    Optimizes inference by sharing Key/Value heads across multiple Query heads.
    Standard MHA is a special case where n_kv_heads == n_heads.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert (
            n_heads % n_kv_heads == 0
        ), "n_heads must be divisible by n_kv_heads for GQA"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.group_size = n_heads // n_kv_heads

        self.Wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        swa_window: int = 0,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos_offset: int = 0,
        use_cache: bool = False,
    ) -> Any:
        # x: [B, Tq, C]
        B, Tq, C = x.size()
        q = self.Wq(x).view(B, Tq, self.n_heads, self.head_dim)
        k = self.Wk(x).view(B, Tq, self.n_kv_heads, self.head_dim)
        v = self.Wv(x).view(B, Tq, self.n_kv_heads, self.head_dim)

        if rope_sin is not None and rope_cos is not None:
            q = apply_rope(q, rope_sin, rope_cos, pos_offset=pos_offset)
            k = apply_rope(k, rope_sin, rope_cos, pos_offset=pos_offset)

        # Append past KV if provided (cache stores KV-heads, unexpanded)
        if past_kv is not None:
            pk, pv = past_kv
            if pk is not None and pv is not None:
                # pk/pv: [B, Tk, H_kv, D]
                k = torch.cat([pk, k], dim=1)
                v = torch.cat([pv, v], dim=1)

        # Optionally enforce sliding-window by trimming KV to the last W tokens
        if swa_window and swa_window > 0:
            Tk_total = k.size(1)
            if Tk_total > swa_window:
                k = k[:, -swa_window:, :, :]
                v = v[:, -swa_window:, :, :]

        # Expand k,v from KV heads to Q heads (repeat per group)
        if self.n_kv_heads != self.n_heads:
            k_expand = k.repeat_interleave(self.group_size, dim=2)
            v_expand = v.repeat_interleave(self.group_size, dim=2)
        else:
            k_expand = k
            v_expand = v

        # SDPA expects [B*H, T, D]
        q_3d = q.permute(0, 2, 1, 3).reshape(B * self.n_heads, Tq, self.head_dim)
        k_3d = (
            k_expand.permute(0, 2, 1, 3)
            .reshape(B * self.n_heads, k_expand.size(1), self.head_dim)
        )
        v_3d = (
            v_expand.permute(0, 2, 1, 3)
            .reshape(B * self.n_heads, v_expand.size(1), self.head_dim)
        )

        use_sdpa_is_causal = (
            (x.device.type != "mps")
            and (not use_cache)
            and bool(causal)
            and (swa_window <= 0)
            and (attn_mask is None)
        )

        # Build mask for training path (Tq==Tk) only when needed.
        # If we can use SDPA's built-in causal masking, avoid passing an explicit mask
        # to enable fused kernels (FlashAttention on CUDA).
        attn_mask_t = None
        if (
            (not use_cache)
            and (not use_sdpa_is_causal)
            and (bool(causal) or (swa_window > 0) or (attn_mask is not None))
        ):
            T = Tq  # equals Tk in training path
            causal_mask = (
                torch.ones(T, T, device=x.device, dtype=torch.bool).triu(1)
                if causal
                else torch.zeros(T, T, device=x.device, dtype=torch.bool)
            )
            if swa_window and swa_window > 0:
                idx = torch.arange(T, device=x.device)
                older_than_w = (idx.view(-1, 1) - idx.view(1, -1)) > swa_window
                attn_mask_t = causal_mask | older_than_w
            else:
                attn_mask_t = causal_mask

        attn_mask_sdpa = None
        if attn_mask_t is not None:
            attn_mask_sdpa = torch.zeros(
                attn_mask_t.shape,
                device=x.device,
                dtype=q_3d.dtype,
            ).masked_fill(attn_mask_t, float("-inf"))

        # Attention: prefer manual math on MPS for stability; use SDPA elsewhere
        if x.device.type == "mps":
            qf = q_3d.to(torch.float32)
            kf = k_3d.to(torch.float32)
            vf = v_3d.to(torch.float32)
            scores = torch.bmm(qf, kf.transpose(1, 2)) * (
                1.0 / math.sqrt(self.head_dim)
            )  # [B*H, Tq, Tk]
            if attn_mask_t is not None:
                mask = attn_mask_t
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).expand(scores.size(0), -1, -1)
                scores = scores.masked_fill(mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            if self.training and self.dropout.p > 0:
                weights = self.dropout(weights)
            attn = torch.bmm(weights, vf).to(q_3d.dtype)
        else:
            attn = F.scaled_dot_product_attention(
                q_3d,
                k_3d,
                v_3d,
                attn_mask=None if use_sdpa_is_causal else attn_mask_sdpa,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=use_sdpa_is_causal,
            )

        attn = (
            attn.reshape(B, self.n_heads, Tq, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, Tq, C)
        )
        out = self.out(attn)

        if use_cache:
            # Return updated cache in KV-head space
            return out, (k, v)
        return out


class TransformerBlock(nn.Module):
    """Single Transformer Block with RMSNorm, GQA, and SwiGLU."""

    def __init__(
        self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout: float = 0.0
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GQAMultiheadAttention(d_model, n_heads, n_kv_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_sin: torch.Tensor,
        rope_cos: torch.Tensor,
        swa_window: int = 0,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos_offset: int = 0,
        use_cache: bool = False,
    ) -> Any:
        if use_cache:
            a, present_kv = self.attn(
                self.norm1(x),
                causal=True,
                rope_sin=rope_sin,
                rope_cos=rope_cos,
                swa_window=swa_window,
                past_kv=past_kv,
                pos_offset=pos_offset,
                use_cache=True,
            )
        else:
            a = self.attn(
                self.norm1(x),
                causal=True,
                rope_sin=rope_sin,
                rope_cos=rope_cos,
                swa_window=swa_window,
            )

        # No dropout on attention residual to match common LLM practice
        x = x + a
        m = self.mlp(self.norm2(x))
        x = x + self.dropout(m)
        if use_cache:
            return x, present_kv
        return x


@dataclass
class GPTConfig:
    """Configuration class for GPTMini."""
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
    gradient_checkpointing: bool = False


class GPTMini(nn.Module):
    """The main GPT Mini model class."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.n_kv_heads,
                    config.d_ff,
                    config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.register_buffer("_rope_sin", None, persistent=False)
        self.register_buffer("_rope_cos", None, persistent=False)
        self._rope_cache_params = (0, 0, 0.0)  # (seq_len, head_dim, theta)
        self._init_weights()

    def _init_weights(self):
        # GPT-style init; residual projections get smaller std for stability.
        base_std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=base_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=base_std)

        # Scale residual projections (GPT-2 style) to stabilize deep stacks.
        if self.config.n_layers > 0:
            resid_std = base_std / math.sqrt(2.0 * float(self.config.n_layers))
            for block in self.blocks:
                nn.init.normal_(block.attn.out.weight, mean=0.0, std=resid_std)
                nn.init.normal_(block.mlp.w3.weight, mean=0.0, std=resid_std)

    def _maybe_build_rope(
        self, seq_len: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype
    ):
        """Builds or extends the RoPE cache if necessary."""
        # Build RoPE cache in float32 to avoid numerical issues when model tensors are float16
        rope_dtype = torch.float32
        if self._rope_sin is None:
            sin, cos = build_rope_cache(seq_len, head_dim, theta, device, rope_dtype)
            self._rope_sin = sin
            self._rope_cos = cos
            self._rope_cache_params = (seq_len, head_dim, theta)
        else:
            cached = self._rope_cache_params
            # Grow-only cache: reuse if requested len <= cached len and settings match
            cached_len, cached_hd, cached_theta = cached
            if (
                (head_dim != cached_hd)
                or (theta != cached_theta)
                or (seq_len > cached_len)
            ):
                new_len = max(seq_len, cached_len)
                sin, cos = build_rope_cache(
                    new_len, head_dim, theta, device, rope_dtype
                )
                self._rope_sin = sin
                self._rope_cos = cos
                self._rope_cache_params = (new_len, head_dim, theta)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for training/inference.

        Args:
            input_ids: Tensor of token IDs [B, T].

        Returns:
             Logits tensor [B, T, vocab_size].
        """
        # input_ids: [B, T]
        B, T = input_ids.shape
        h = self.tok_emb(input_ids)  # [B,T,C]
        head_dim = self.config.d_model // self.config.n_heads
        self._maybe_build_rope(
            T, head_dim, self.config.rope_theta, h.device, h.dtype
        )
        use_ckpt = (
            bool(getattr(self.config, "gradient_checkpointing", False))
            and self.training
        )
        for i, blk in enumerate(self.blocks):
            if use_ckpt:
                # IMPORTANT: bind `blk` per-iteration; otherwise checkpoint recompute can
                # close over the loop variable and use the wrong block.
                def fn(x, block=blk):
                    return block(
                        x,
                        self._rope_sin,
                        self._rope_cos,
                        swa_window=self.config.swa_window,
                    )

                try:
                    h = checkpoint(fn, h, use_reentrant=False)
                except TypeError:
                    # Fallback for older torch versions without use_reentrant
                    h = checkpoint(fn, h)
            else:
                h = blk(
                    h,
                    self._rope_sin,
                    self._rope_cos,
                    swa_window=self.config.swa_window,
                )
        h = self.norm_f(h)
        logits = self.lm_head(h)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        no_repeat_ngram_size: int = 0,
        max_repeat_token_run: int = 3,
    ) -> torch.Tensor:
        """Autoregressive generation with KV caching.

        Args:
            input_ids: Starter tokens [1, T].
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling (0 to disable).
            repetition_penalty: Penalty for repeated tokens.
            eos_token_id: Token ID to stop generation.
            no_repeat_ngram_size: Size of n-grams to block.
            max_repeat_token_run: Max consecutive runs of a single token.

        Returns:
            Tensor of completed sequence including input [1, T + new].
        """
        self.eval()
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        B = input_ids.size(0)
        assert B == 1, "generate with KV cache currently supports batch size 1"

        # Build KV caches by stepping through the prompt one token at a time
        caches = [None] * len(self.blocks)
        head_dim = self.config.d_model // self.config.n_heads
        total_len = int(input_ids.size(1))
        last_logits = None
        for t in range(total_len):
            tok = input_ids[:, t : t + 1]
            h = self.tok_emb(tok)
            self._maybe_build_rope(
                t + 1, head_dim, self.config.rope_theta, h.device, h.dtype
            )
            for i, blk in enumerate(self.blocks):
                past = caches[i]
                h, present = blk(
                    h,
                    self._rope_sin,
                    self._rope_cos,
                    swa_window=self.config.swa_window,
                    past_kv=past,
                    pos_offset=t,
                    use_cache=True,
                )
                caches[i] = present
            h = self.norm_f(h)
            last_logits = self.lm_head(h)[:, -1, :]

        generated = []
        cur_len = total_len
        for _ in range(max_new_tokens):
            temp_eps = 1e-8
            scale = max(temp_eps, float(temperature))
            logits = last_logits / scale

            # Simple repetition penalty: downweight tokens already present in the sequence
            if repetition_penalty != 1.0 and repetition_penalty > 0.0:
                uniq = torch.unique(input_ids[0], sorted=False)
                logits[0, uniq] = logits[0, uniq] / repetition_penalty

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Enforce simple no-repeat-ngram constraint (batch size = 1)
            if no_repeat_ngram_size and no_repeat_ngram_size > 1:
                seq = input_ids[0].tolist()
                n = int(no_repeat_ngram_size)
                if len(seq) >= n - 1:
                    prefix = tuple(seq[-(n - 1) :])
                    banned = set()
                    for i in range(len(seq) - n + 1):
                        if tuple(seq[i : i + (n - 1)]) == prefix:
                            banned.add(seq[i + (n - 1)])
                    if banned:
                        banned_idx = torch.tensor(
                            list(banned), device=probs.device, dtype=torch.long
                        )
                        banned_idx = banned_idx[
                            (banned_idx >= 0) & (banned_idx < probs.size(-1))
                        ]
                        if banned_idx.numel() > 0:
                            probs[0, banned_idx] = 0.0

            # Prevent degenerate runs of the same token (e.g., "computer: computer:")
            if (
                max_repeat_token_run
                and max_repeat_token_run > 0
                and input_ids.size(1) >= max_repeat_token_run
            ):
                last_tokens = input_ids[0, -max_repeat_token_run:]
                if torch.all(last_tokens == last_tokens[-1]):
                    probs[0, last_tokens[-1].item()] = 0.0

            # Optional top-k filtering
            if top_k and top_k > 0:
                topk_vals, topk_idx = torch.topk(
                    probs, k=min(top_k, probs.size(-1)), dim=-1
                )
                mask = torch.zeros_like(probs)
                mask.scatter_(1, topk_idx, topk_vals)
                probs = mask

            # Nucleus (top-p) filtering
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cum > top_p
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0
            probs = torch.zeros_like(probs).scatter(1, sorted_idx, sorted_probs)
            probs_sum = probs.sum(dim=-1, keepdim=True)
            probs = probs / probs_sum.clamp_min(1e-12)

            # If temperature==0 (greedy) or probabilities collapsed to zeros, fallback to argmax
            if (
                float(temperature) <= 0.0
                or torch.any(~torch.isfinite(probs_sum))
                or torch.all(probs_sum <= 0)
            ):
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)  # [B,1]
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated.append(next_token)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # One-step forward with cache to get logits for next step
            t = cur_len
            h = self.tok_emb(next_token)
            self._maybe_build_rope(
                t + 1, head_dim, self.config.rope_theta, h.device, h.dtype
            )
            for i, blk in enumerate(self.blocks):
                past = caches[i]
                h, present = blk(
                    h,
                    self._rope_sin,
                    self._rope_cos,
                    swa_window=self.config.swa_window,
                    past_kv=past,
                    pos_offset=t,
                    use_cache=True,
                )
                caches[i] = present
            h = self.norm_f(h)
            last_logits = self.lm_head(h)[:, -1, :]
            cur_len += 1

        if generated:
            gen_ids = torch.cat(generated, dim=1)
            return torch.cat([input_ids[:, :total_len], gen_ids], dim=1)
        return input_ids
