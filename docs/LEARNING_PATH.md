# Learning Path

This repo is meant to be read like a guided code walkthrough. A good first pass is:

## 1) Run something end-to-end

- Smoke train: `python3 train.py --config config_smoke.yaml`
- Smoke infer: `python3 infer.py --ckpt checkpoints/final.ckpt --tokenizer gpt2 --prompt "Attention is"`

Once this works, you have a concrete baseline for every change you make.

## 2) Understand the model (shapes + responsibilities)

Open `model.py` and follow these in order:

- `GPTConfig`: the “contract” for model hyperparameters
- `GPTMini.__init__`: embeddings → blocks → final norm → LM head (optionally tied)
- `TransformerBlock`: RMSNorm → attention residual → RMSNorm → SwiGLU residual
- `GQAMultiheadAttention.forward`: Q/K/V projection, RoPE, causal masking, SDPA vs MPS path
- `GPTMini.generate`: autoregressive loop with KV cache and sampling controls

Suggested exercises:

- Set `n_kv_heads == n_heads` and compare speed/memory to GQA.
- Toggle `gradient_checkpointing` and confirm identical grads (tests cover this).
- Increase `rope_theta` and inspect how it affects long-context behavior.

## 3) Understand the data path (why “packing” matters)

Open `data.py`:

- `StreamingPackedLMDataset`: tokenizes docs → appends EOS → packs into fixed-length sequences
- `PackedLMDataset`: same idea but materialized in memory
- `make_dataloaders`: chooses HF streaming vs local text (`data/tiny_corpus.txt`)

Key idea: packing changes the training distribution. If you don’t separate documents (EOS),
you can teach the model to “leak” from one document into the next.

## 4) Understand training loop + checkpoints

Open `train.py`:

- `LitCausalLM`: computes logits → cross entropy loss → logs metrics
- `configure_optimizers`: AdamW parameter groups (decay vs no-decay), LR scheduling
- `find_latest_checkpoint`: resume behavior
- `pl.Trainer(...)`: how devices/precision/accumulation/checkpoints are wired

Suggested exercises:

- Reduce `training.max_steps` to 1000 and see how quickly loss drops on the smoke corpus.
- Change `training.grad_accum_steps` and confirm effective batch changes without OOM.

## 5) Inference + evaluation

- `infer.py`: loads a Lightning checkpoint and samples text; supports `--seed` and sampling controls.
- `evaluate_checkpoints.py`: quick ranking on small “cases” files (prompt + optional expected target).
- `eval_val.py`: computes true `val_loss`/`val_ppl` using the config’s validation dataloader.

## 6) Fine-tuning (SFT)

Open `finetune.py`:

- `InstructPairDataset`: builds Alpaca-style prompt/response and **masks prompt tokens** in the loss
- `LitSFT`: reuses the base LM training module; adds an instruction accuracy metric

If you only understand one SFT concept: *don’t train on the prompt tokens*.

