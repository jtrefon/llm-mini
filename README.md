# Tiny Transformer Starter (MBP\u2011friendly)

A clean, from\u2011scratch, decoder\u2011only Transformer with **RMSNorm + RoPE + GQA + SwiGLU**, built for a **MacBook M\u2011series (MPS)** first run. Uses **PyTorch Lightning** for painless training. Later, move to an RTX 4090 and crank depth/context & FlashAttention.

## Quickstart

```bash
# 1) Create venv and install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) (Optional) Login to HF if you use private tokenizers
# huggingface-cli login

# 3) Edit config.yaml if you want. Defaults are laptop-safe.

# 4) Train (MBP M\u2011series)
python train.py --config config.yaml

# 5) Inference
python infer.py --ckpt checkpoints/final.ckpt --tokenizer gpt2 --prompt "The history of Dubai is"
```

## Notes
- **Tokenizer**: default is `gpt2` for zero\u2011friction. Swap to a Llama\u2011family tokenizer later if you have access. Model `vocab_size` adapts automatically.
- **Dataset**: defaults to `shahrukhx01/wikipedia-bookscorpus-en-preprocessed` (clean + chunked, a good starter). Change `data.dataset` to `wikimedia/wikipedia` (e.g., `20231101.en`) if you want fresher raw articles; youll still pack them here.
- **Sequence packing**: the dataset is tokenized and **concatenated then chunked** into `(seq_len+1)` blocks for next\u2011token prediction.
- **GQA**: reduces KV cache size at inference. Configurable via `n_kv_heads`.
- **SWA (Sliding\u2011Window Attention)**: set `model.swa_window > 0` to restrict attention to last *W* tokens (near\u2011linear cost). Keep it `0` initially.
- **Precision**: `16-mixed` on MPS works well. If you hit NaNs, try full `32`.

## Scaling up (4090)
- Switch `hardware.accelerator: gpu` and bump `model.n_layers`, `d_model`, `training.seq_len: 2048\u20134096`.
- Install `flash-attn` and replace SDPA with FlashAttention kernels if you like (left as an exercise, since MPS doesn\u2019t support it).

## Finetuning & Reasoning (later)
- Add instruction SFT with a chat template and DPO/ORPO for alignment.
- Add a “reasoning switch”: sample k>1 candidates (self\u2011consistency) and optional verifiers.

---

# End of embedded repo
