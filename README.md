# Tiny Transformer Starter (MBP‑friendly)

A clean, from‑scratch, decoder‑only Transformer with **RMSNorm + RoPE + GQA + SwiGLU**, built for a **MacBook M‑series (MPS)** first run. Uses **PyTorch Lightning** for painless training. Easy, hackable, and ready to scale.

## Features

- **Architecture**: Modern Llama-style components.
    - **RMSNorm**: Pre-normalization for better stability.
    - **RoPE**: Rotary Positional Embeddings for better long-context performance.
    - **SwiGLU**: Gated linear unit activation in the MLP.
    - **GQA**: Grouped-Query Attention for faster inference and lower memory usage.
- **Optimized for MPS**: Defaults configured for Apple Silicon Metal (MPS) acceleration.
- **PyTorch Lightning**: Organized training loop with checkpointing, logging, and callbacks.
- **Zero Dependencies**: Uses standard Hugging Face `transformers` and `datasets` just for tokenization and data loading.

## Quickstart

### 1. Installation

```bash
# Create venv and install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Training

Train on your laptop with the default tiny config:

```bash
# Optional: Login to HF if you use private tokenizers/datasets
# huggingface-cli login

# Train (MBP M‑series friendly defaults)
python train.py --config config.yaml
```

### 3. Inference

Generate text from a trained checkpoint:

```bash
python infer.py --ckpt checkpoints/final.ckpt --tokenizer gpt2 --prompt "The history of Dubai is"
```

### 4. Evaluation

Score checkpoints on simple probes:

```bash
python evaluate_checkpoints.py \
  --checkpoints 'checkpoints/*.ckpt' \
  --tokenizer gpt2 \
  --top-k 5 \
  --cases cases/wiki_basics.yaml
```

## Architecture Details

This codebase implements a "mini" version of modern LLM architectures (like Llama 2/3, Mistral).

-   **Rotary Positional Embeddings (RoPE)**: Applied to Queries and Keys to encode position without absolute positional embeddings.
-   **Grouped Query Attention (GQA)**: Keys and Values are shared across groups of Query heads. controlled by `n_heads` and `n_kv_heads` in `config.yaml`.
-   **SwiGLU MLP**: The feed-forward network uses the Swish-Gated Linear Unit formulation. `d_ff` is typically 4 * `d_model` * (2/3).
-   **RMSNorm**: Used instead of LayerNorm for normalization.

## Project Structure

```
├── model.py              # The Transformer architecture (PyTorch nn.Module)
├── data.py               # Dataset processing (PackedLMDataset, streams)
├── train.py              # Training script & PyTorch Lightning Module
├── infer.py              # Simple inference script
├── config.yaml           # Configuration file
├── checkpoints/          # Saved model checkpoints
└── logs/                 # Training logs
```

## Scaling Up (GPU / 4090)

To move to a dedicated GPU environment:

1.  Update `config.yaml`:
    -   `hardware.accelerator: gpu`
    -   Increase `model.n_layers`, `model.d_model`, `training.seq_len`.
2.  (Optional) Install `flash-attn` for faster attention kernels (MPS does not support FlashAttention yet).

## License

MIT License. See [LICENSE](LICENSE) file.
