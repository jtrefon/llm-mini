# Tiny Transformer (Educational LLM From Scratch)

An educational, decoder-only Transformer you can read end-to-end: **model**, **data packing**, **training**, **inference**, **checkpoint eval**, and **SFT fine-tuning**.

This is not the Hugging Face `transformers` library (it only uses HF for tokenizers/datasets).

## Why this exists

Most “train an LLM” repos are either too big to understand, or too abstract to teach. This project aims to be:

- **Readable**: small, explicit code (no hidden magic).
- **Hackable**: change one thing and see what breaks/helps.
- **Educational**: enough modern features to be realistic (RoPE, RMSNorm, SwiGLU, GQA, KV cache).

Start here: `docs/LEARNING_PATH.md`.

## What you’ll learn (practical)

- How a decoder-only Transformer produces logits and trains with next-token cross-entropy (`model.py`, `train.py`)
- How tokenization + document boundaries + packing affect training (`data.py`)
- Why RoPE/RMSNorm/SwiGLU/GQA exist and where they sit in code (`model.py`)
- How sampling works (temperature, top-p, top-k, repetition penalty) (`infer.py`)
- How to compare checkpoints quickly before fine-tuning (`evaluate_checkpoints.py`, `eval_val.py`)
- How simple instruction fine-tuning works (mask prompt tokens, train on responses) (`finetune.py`)

## Quickstart

### 0) Requirements

- Python `3.11+`
- PyTorch `2.3+`
- One of: Apple Silicon (MPS), NVIDIA (CUDA), or CPU

### 1) Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Smoke test (fast, local text)

Runs ~50 steps on `data/tiny_corpus.txt` to prove the pipeline works.

```bash
python3 train.py --config config_smoke.yaml
python3 infer.py --ckpt checkpoints/final.ckpt --tokenizer gpt2 --prompt "Attention is"
```

### 3) Train (bigger, HF dataset)

`config.yaml` defaults to streaming Wikipedia.

```bash
python3 train.py --config config.yaml
```

### 4) Inference

```bash
python3 infer.py --ckpt checkpoints/final.ckpt --tokenizer gpt2 --prompt "The history of Dubai is"
```

### 5) Evaluate checkpoints

```bash
python3 evaluate_checkpoints.py --checkpoints 'checkpoints/*.ckpt' --tokenizer gpt2 --cases cases/wiki_basics.yaml
python3 eval_val.py --ckpt checkpoints/best.ckpt --config config.yaml --tokenizer auto
python3 list.py
```

### 6) Fine-tune (SFT)

```bash
python3 finetune.py --config config_finetune.yaml
```

## Project structure

```
.
├── model.py                 # Decoder-only Transformer (RoPE, RMSNorm, SwiGLU, GQA, KV cache)
├── data.py                  # HF + local text loaders, packing into fixed-length sequences
├── train.py                 # Pretraining loop (PyTorch Lightning), checkpointing, logging
├── finetune.py              # Instruction SFT on Alpaca-style data (masked prompt loss)
├── infer.py                 # Load checkpoint + sample text (temperature/top-p/top-k/etc.)
├── evaluate_checkpoints.py  # Quick ranking on small prompt/target case files
├── eval_val.py              # True val_loss / val_ppl for a checkpoint + config dataloader
├── list.py                  # Inspect checkpoints (epoch/step/loss metadata)
├── config_smoke.yaml        # Fast local smoke test config
├── config.yaml              # Default training config (HF dataset)
├── config_finetune.yaml     # Default SFT config (HF dataset)
└── docs/                    # Educational walkthroughs
```

## Docs

- `docs/LEARNING_PATH.md` — suggested order + “what to read first”
- `docs/CONFIG_REFERENCE.md` — config knobs and what they do
- `docs/TROUBLESHOOTING.md` — common issues (MPS/CUDA/CPU, OOM, offline mode)

## Contributing

See `CONTRIBUTING.md`.

## License

Apache 2.0. See `LICENSE`.
