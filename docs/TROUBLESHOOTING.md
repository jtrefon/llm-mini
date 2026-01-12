# Troubleshooting

## “`python` not found”

Use `python3` (or activate the venv first):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 --version
```

## MPS issues (Apple Silicon)

- If you see dataloader worker / multiprocessing issues, set `hardware.num_workers: 0`.
- If you see OOM, reduce:
  - `training.seq_len`
  - `model.d_model` / `model.n_layers`
  - increase `training.grad_accum_steps` instead of `micro_batch_size`

## CUDA issues (NVIDIA)

- Make sure your installed PyTorch matches your CUDA version.
- If you OOM, the same knobs apply: `seq_len`, model size, batch size.

## “Offline mode enabled” errors

If `data.offline: true`, the code sets `HF_DATASETS_OFFLINE=1` and `HF_HUB_OFFLINE=1`.
That means **no downloads** are allowed.

Fix:

- Set `data.offline: false` in your config for a first run, or
- Pre-cache the tokenizer/dataset on the machine and keep offline mode on.

## “Dataset not found” / HF Hub rate limits

- Use streaming (`data.streaming: true`) for large corpora.
- For quick local experiments, use `config_smoke.yaml` (local file dataset).

## Resume / checkpoint shape mismatch

If you changed model sizes (layers/width/heads), old checkpoints may be incompatible.

- Easiest fix: `python3 train.py --config your_config.yaml --no_resume`
