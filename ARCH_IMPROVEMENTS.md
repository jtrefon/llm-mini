# Architecture Review: Issues, Fixes, and Refactor Plan

This document tracks the issues identified in the pretraining, model, fine‑tuning, and inference stack, along with implemented fixes and safe rollback notes.

## Summary

- Scope: Root-level training (`train.py`, `data.py`), model (`model.py`), SFT (`finetune.py`), and inference (`infer.py`).
- Status: Critical and high-impact fixes implemented; non-breaking, focused changes.

## Issues and Fixes

### 1) Document Boundaries Missing in Pretraining
- Issue: Training concatenates documents without explicit EOS separators, risking cross‑doc leakage and degraded generation quality.
- Fix: Ensure an EOS token terminates every document during tokenization.
- Files: `data.py`
- Rollback: Safe — EOS affects only future training. To revert, remove EOS appending.

### 2) SFT LR Scheduler Conflicts
- Issue: SFT mixed a per‑step LambdaLR (warmup+cosine) with a callback that mutates LR on validation plateau. LambdaLR overwrote the plateau changes on the next step.
- Fix: Use a warmup callback to ramp to base LR, then manage LR solely via Reduce‑On‑Plateau. Removed step‑based LambdaLR from SFT.
- Files: `finetune.py`
- Rollback: Switch back to the previous LambdaLR if desired and remove the plateau callback or vice versa.

### 3) Weight Decay Applied to All Params
- Issue: AdamW applied weight decay to biases and norm weights, which can hurt stability.
- Fix: Create optimizer parameter groups: apply weight decay to matrix weights (ndim ≥ 2); set weight decay to 0 for biases and norm scale params.
- Files: `train.py`, `finetune.py`
- Rollback: Replace param groups with a single-parameter group optimizer.

### 4) RoPE Cache Rebuilds Every Decode Step
- Issue: RoPE caches rebuilt whenever sequence length changes, including at each generation step, wasting time.
- Fix: Grow‑only RoPE cache. Reuse if `new_len <= cached_len`; only rebuild when a longer sequence is required or settings change.
- Files: `model.py`
- Rollback: Revert `_maybe_build_rope` to rebuild on any length change.

### 5) Inference UX and Stability
- Issue: Inference printed the full prompt + generation, offered no seeding, and silently accepted state dict mismatches.
- Fixes:
  - Print only the generated continuation after the input prompt.
  - Add `--seed` for reproducibility.
  - Warn on shape/tokenizer mismatches when loading checkpoints.
  - Keep `--instruct`, `--top_k`, `--repetition_penalty` for repetition control.
- Files: `infer.py`
- Rollback: Remove the new flags and revert to full-sequence printing.

### 6) SFT Safety (previously implemented)
- Fixes already merged earlier:
  - Load base pretraining weights via `training.base_ckpt_path`.
  - Assert supervised tokens exist per SFT batch; dataset truncation preserves response tokens.
  - Add decoding guardrails (`top_k`, `repetition_penalty`) and `--instruct` template.
- Files: `finetune.py`, `model.py`, `infer.py`, `config.yaml`

## Deferred/Optional (Not Implemented in this pass)

- KV Cache for Generation: Adds complexity (RoPE offset/masks). Recommended as a follow‑up once the above stabilizes.
- Data Streaming Improvements: Current pipeline can scale by toggling `streaming`; a dedicated iterable dataset can be added later.

## Validation Checklist

- Pretraining:
  - Logs reflect tokens/sec; quality improves with EOS boundaries.
  - Optimizer reports two param groups (decay/no‑decay).
- SFT:
  - Startup log: base weights loaded (if configured).
  - No “0 supervised tokens” errors.
  - LR warmup then plateau reductions; no per‑step scheduler state.
- Inference:
  - `--seed` yields reproducible outputs.
  - Only continuation text printed with `--instruct` template.

## Rollback Notes

- All changes are local and guarded by defaults. You can revert any section by restoring the prior functions in the referenced files. No persistent formats were changed (checkpoints remain compatible).

