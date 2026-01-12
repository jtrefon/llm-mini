# Config reference

Configs are YAML files passed to `train.py` / `finetune.py` with `--config`.

The top-level keys are:

- `model`: architecture hyperparameters
- `training`: optimizer, schedule, batching, logging, checkpoints
- `hardware`: device selection and dataloader worker settings
- `data`: dataset/tokenizer selection and packing behavior

## `model`

- `n_layers`: number of Transformer blocks
- `d_model`: hidden size
- `n_heads`: attention heads (queries)
- `n_kv_heads`: KV heads (for GQA); set equal to `n_heads` to disable GQA
- `d_ff`: MLP hidden size (SwiGLU is implemented as a gated projection)
- `dropout`: residual dropout (attention residual dropout is intentionally disabled)
- `rope_theta`: RoPE base (larger can help extrapolate to longer contexts)
- `tie_embeddings`: ties token embedding weights to LM head (common in LLMs)
- `swa_window`: sliding-window attention size (0 disables)
- `gradient_checkpointing`: recompute activations to save memory (slower, less RAM)

## `training`

- `seq_len`: tokens per training sequence
- `micro_batch_size`: per-step batch size (before gradient accumulation)
- `grad_accum_steps`: number of steps to accumulate gradients before optimizer step
- `max_steps`: total optimizer steps (not “examples”)
- `save_every`: checkpoint frequency (in optimizer steps; multiplied by `grad_accum_steps` internally)
- `lr`, `betas`, `eps`, `weight_decay`: AdamW hyperparameters
- `lr_schedule`: `"warmup_cosine"` (recommended) or `"plateau"`
- `warmup_ratio`: fraction of steps used for warmup
- `precision`: Lightning precision setting (`32`, `"16-mixed"`, `"bf16-mixed"`)
- `auto_resume`: resume from latest checkpoint in `checkpoints/` when present
- `reset_lr_on_resume`: if true, resets LR scheduler state when resuming

## `hardware`

- `accelerator`: `"auto"`, `"mps"`, `"cuda"`, or `"cpu"`
- `devices`: number of devices (typically `1` for this repo)
- `num_workers`: dataloader workers (MPS often prefers `0`)

## `data`

- `dataset`: either a Hugging Face dataset name (e.g. `wikimedia/wikipedia`) or a local text file:
  - `data/tiny_corpus.txt`
  - `file:/absolute/path/to/corpus.txt`
- `offline`: if true, sets HF offline env vars and requires cached datasets/tokenizers
- `streaming`: use streaming datasets (recommended for large corpora)
- `tokenizer_name`: HF tokenizer name/path (e.g. `gpt2`)
- `pack_sequences`: pack documents into fixed-length blocks (recommended)
- `max_doc_len`: optional per-document cap during tokenization (0/empty disables)
- `train_docs`, `val_docs`: optional doc limits for quicker runs

