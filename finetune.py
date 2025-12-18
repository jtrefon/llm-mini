# finetune.py
import argparse
import os, glob, math, yaml, shutil
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer

from model import GPTMini, GPTConfig
from train import (
    LitCausalLM,
    WarmupLRCallback,
)  # reuse your existing LightningModule & checkpoint utilities

class LitSFT(LitCausalLM):
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        logits = self(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        ppl = torch.exp(loss)
        self.log('val_ppl', ppl, prog_bar=True, on_step=False, on_epoch=True)
        # Instruction-specific metric: accuracy on non-ignored labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        pred = shift_logits.argmax(dim=-1)
        mask = shift_labels != -100
        acc = (pred == shift_labels)[mask].float().mean()
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

# ---- Nice-to-haves for CUDA/MPS
try:
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass


# =========================
# Data: Alpaca -> (prompt, response) with masked labels
# =========================
class InstructPairDataset(Dataset):
    """
    Supervised fine-tuning dataset for instruction following.
    Labels are next-token targets with prompt tokens masked (-100). The last prompt
    token predicts the first response token; loss is only applied where labels >= 0.
    """
    def __init__(self, hf_ds: HFDataset, tokenizer: AutoTokenizer, seq_len: int):
        self.ds = hf_ds
        self.tok = tokenizer
        self.seq_len = seq_len
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or self.tok.unk_token

    @staticmethod
    def _alpaca_to_pair(ex: Dict[str, str]) -> Dict[str, str]:
        instr = (ex.get("instruction") or "").strip()
        inp   = (ex.get("input") or "").strip()
        out   = (ex.get("output") or "").strip()
        prompt = f"### Instruction:\n{instr}\n"
        if inp:
            prompt += f"\n### Input:\n{inp}\n"
        prompt += "\n### Response:\n"
        return {"prompt": prompt, "response": out}

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        ex = self._alpaca_to_pair(self.ds[int(idx)])
        # tokenize prompt/response separately so we can mask prompt
        prompt_ids = self.tok(ex["prompt"], add_special_tokens=True, truncation=False)["input_ids"]
        resp_text = ex["response"].strip()
        if self.tok.eos_token:
            resp_text += self.tok.eos_token
        resp_ids = self.tok(resp_text, add_special_tokens=False, truncation=False)["input_ids"]

        # Build input sequence and next-token labels (shifted) while masking prompt
        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(input_ids)
        prompt_len = len(prompt_ids)
        # Start from prompt_len-1 so the last prompt token predicts first response token
        start_t = max(0, prompt_len - 1)
        for t in range(start_t, len(input_ids) - 1):
            labels[t] = input_ids[t + 1]

        # Hard truncate from the left if too long (prefer tail where response lives)
        if len(input_ids) > self.seq_len:
            cut = len(input_ids) - self.seq_len
            input_ids = input_ids[cut:]
            labels    = labels[cut:]
            # ensure at least some supervised tokens remain; if not, rebuild labels for window
            if all(l == -100 for l in labels):
                window = (prompt_ids + resp_ids)[-self.seq_len:]
                total_len = len(prompt_ids) + len(resp_ids)
                start_idx = max(0, total_len - self.seq_len)
                overlap_prompt = max(0, len(prompt_ids) - start_idx)
                labels_win = [-100] * len(window)
                start_t = max(0, overlap_prompt - 1)
                for t in range(start_t, len(window) - 1):
                    labels_win[t] = window[t + 1]
                input_ids = window
                labels    = labels_win

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


def collate_sft(batch: List[Dict[str, torch.Tensor]], pad_id: int, seq_len: int) -> Dict[str, torch.Tensor]:
    max_len = min(max(len(b["input_ids"]) for b in batch), seq_len)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels    = torch.full((len(batch), max_len), -100,  dtype=torch.long)
    for i, b in enumerate(batch):
        x = b["input_ids"][:max_len]
        y = b["labels"][:max_len]
        input_ids[i, :x.size(0)] = x
        labels[i,    :y.size(0)] = y
    # Sanity check: ensure there is at least one supervised token in the batch
    supervised = (labels != -100).sum().item()
    if supervised == 0:
        raise RuntimeError("SFT batch has 0 supervised tokens; check template/truncation!")
    return {"input_ids": input_ids, "labels": labels}


class CollateSFT:
    """Picklable collate function wrapper for DataLoader workers."""
    def __init__(self, pad_id: int, seq_len: int):
        self.pad_id = pad_id
        self.seq_len = seq_len

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return collate_sft(batch, self.pad_id, self.seq_len)


def make_alpaca_loaders(tokenizer: AutoTokenizer, seq_len: int, micro_batch_size: int,
                         train_limit: Optional[int] = None, val_limit: int = 1000, num_workers: int = 0):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    # drop empties
    ds = ds.filter(lambda ex: isinstance(ex.get("output", None), str) and len(ex["output"].strip()) > 0)
    if train_limit:
        ds = ds.select(range(min(train_limit, len(ds))))
    # carve a small val set from the tail (disjoint)
    val_n = min(val_limit, max(500, len(ds)//10))
    val_start = max(0, len(ds) - val_n)
    ds_train = ds.select(range(0, val_start))
    ds_val   = ds.select(range(val_start, len(ds)))

    train_ds = InstructPairDataset(ds_train, tokenizer, seq_len)
    val_ds   = InstructPairDataset(ds_val,   tokenizer, seq_len)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    persistent = bool(num_workers and num_workers > 0)
    prefetch = 1 if persistent else None
    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=CollateSFT(pad_id, seq_len),
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=CollateSFT(pad_id, seq_len),
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )
    return train_loader, val_loader


# =========================
# Callbacks: checkpoint pick + reduce-on-plateau + early stop
# =========================
def pick_checkpoint(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    best = os.path.join(ckpt_dir, "best.ckpt")
    if os.path.exists(best):
        return best
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")), key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


class ReduceLROnPlateauOnVal(pl.Callback):
    """Reduce LR when val_loss plateaus (uses your config thresholds)."""
    def __init__(self, patience=2, factor=0.5, min_lr=1e-6, cooldown=0, threshold=0.0,
                 monitor="val_loss", mode="min", warmup_steps: int = 0):
        self.patience = patience; self.factor = factor; self.min_lr = min_lr
        self.cooldown = cooldown; self.threshold = threshold
        self.monitor = monitor; self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.bad = 0; self.cool = 0
        self._warmup_steps = int(max(0, warmup_steps))

    def _improved(self, cur):
        return (self.best - cur) > self.threshold if self.mode == "min" else (cur - self.best) > self.threshold

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_step < self._warmup_steps:
            return
        if self.monitor not in trainer.callback_metrics:
            return
        cur = float(trainer.callback_metrics[self.monitor])
        if self._improved(cur):
            self.best = cur; self.bad = 0
            if self.cool > 0: self.cool -= 1
            return
        self.bad += 1
        if self.cool > 0:
            self.cool -= 1
            return
        if self.bad >= self.patience:
            opt = trainer.optimizers[0]
            for pg in opt.param_groups:
                pg["lr"] = max(self.min_lr, pg["lr"] * self.factor)
            # Can't call self.log() here per PL; print instead
            try:
                new_lr = opt.param_groups[0]["lr"]
                print(f"[SFT] LR reduced to {new_lr:.6g}")
            except Exception:
                pass
            try:
                for cb in trainer.callbacks:
                    if isinstance(cb, pl.callbacks.early_stopping.EarlyStopping):
                        cb.wait_count = 0
            except Exception:
                pass
            self.bad = 0; self.cool = self.cooldown


class EarlyStoppingWithWarmup(pl.callbacks.early_stopping.EarlyStopping):
    """Early stopping that ignores validations until a warmup step threshold.

    Useful for SFT where metrics may wobble initially or reach a low value and then
    recover after scheduler adjustments. This delays early stopping decisions until
    `min_steps` have elapsed.
    """

    def __init__(self, *args, min_steps: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_steps = int(max(0, min_steps))

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_step < self._min_steps:
            # Skip early stopping check during warmup window
            return
        return super().on_validation_end(trainer, pl_module)


# Removed StableCheckpointCopies; rely on a single ModelCheckpoint with save_last=True


# =========================
# Main
# =========================
def main(config_path: str = "config_finetune.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"[SFT] Loading config: {config_path}")

    pl.seed_everything(cfg["training"]["seed"])

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer_name"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    # Silence HF long-sequence warnings; we handle truncation later
    try:
        tok.model_max_length = int(1e9)
    except Exception:
        pass

    # Data
    seq_len = cfg["training"]["seq_len"]
    mb = cfg["training"]["micro_batch_size"]
    nw = cfg["hardware"]["num_workers"]
    train_limit = cfg["data"].get("train_docs", None)
    val_limit   = cfg["data"].get("val_docs", 1000)
    train_loader, val_loader = make_alpaca_loaders(tok, seq_len, mb, train_limit, val_limit, nw)

    # Model (reuse your config dims)
    mcfg = GPTConfig(
        vocab_size=len(tok),
        n_layers=cfg["model"]["n_layers"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_kv_heads=cfg["model"]["n_kv_heads"],
        d_ff=cfg["model"]["d_ff"],
        dropout=cfg["model"]["dropout"],  # you can set 0.05 later if overfitting
        rope_theta=cfg["model"]["rope_theta"],
        tie_embeddings=cfg["model"]["tie_embeddings"],
        swa_window=cfg["model"]["swa_window"],
        gradient_checkpointing=bool(cfg["model"].get("gradient_checkpointing", False)),
    )
    lit = LitSFT(cfg, tok)
    lit.net = GPTMini(mcfg)  # ensure module matches mcfg

    # Optionally initialize from a provided base pretraining checkpoint
    def load_base_into_lit(lit_module: pl.LightningModule, base_ckpt_path: str):
        try:
            ckpt = torch.load(base_ckpt_path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            # Accept either 'net.*' keys (Lightning) or raw module keys
            if any(k.startswith("net.") for k in state.keys()):
                base_sd = {k.split("net.", 1)[1]: v for k, v in state.items() if k.startswith("net.")}
            else:
                base_sd = state
            compat = lit_module.net.load_state_dict(base_sd, strict=False)
            missing = len(getattr(compat, "missing_keys", []))
            unexpected = len(getattr(compat, "unexpected_keys", []))
            print(f"[SFT] Loaded base weights from {base_ckpt_path}: missing={missing}, unexpected={unexpected}")
            return True
        except Exception as e:
            print(f"[SFT] WARNING: Failed to load base weights from {base_ckpt_path}: {e}")
            return False

    base_ckpt = cfg["training"].get("base_ckpt_path", None)
    if not base_ckpt:
        raise ValueError("[SFT] training.base_ckpt_path must point to a pretrained checkpoint before fine-tuning")
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"[SFT] base_ckpt_path does not exist: {base_ckpt}")

    ok = load_base_into_lit(lit, base_ckpt)
    if not ok:
        raise RuntimeError(f"[SFT] Failed to load base weights from {base_ckpt}")

    # Optimizer & schedule: gentle for SFT
    base_lr = float(cfg["training"].get("lr", 1e-5))
    cfg["training"]["lr"] = base_lr  # reflect in hparams
    # Parameter groups: apply weight decay to weights (ndim>=2), not to biases/norms
    wd = float(cfg["training"]["weight_decay"])
    betas = tuple(float(x) for x in cfg["training"]["betas"])
    eps = float(cfg["training"]["eps"])
    decay_params = []
    nodecay_params = []
    for n, p in lit.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": wd},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=base_lr, betas=betas, eps=eps)

    # Unify scheduling: Warmup to base LR, then rely on Reduce-On-Plateau
    # Do not attach a per-step LambdaLR to avoid overwriting plateau adjustments
    lit.configure_optimizers = lambda: {"optimizer": optimizer}

    # Hardware
    accelerator = cfg["hardware"]["accelerator"]
    devices = cfg["hardware"]["devices"]
    # Total training steps (used by ES warmup and validation cadence)
    max_steps = int(cfg["training"].get("max_steps", 1000))

    # Callbacks
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    ckpt_dir = cfg["training"].get("ckpt_dir", cfg["training"].get("checkpoint_dir", "checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="sft-best",
        monitor="val_loss", mode="min", save_top_k=1, save_last="sft-last",
        auto_insert_metric_name=False,
    )
    es_cfg  = cfg["training"].get("early_stopping", {"enabled": True, "patience": 9, "min_delta": 0.001})
    es_enabled = bool(es_cfg.get("enabled", True))
    es_patience = int(es_cfg.get("patience", 9))
    es_min_delta = float(es_cfg.get("min_delta", 0.001))
    es_warm_frac = float(es_cfg.get("warmup_fraction", 0.3))  # ignore ES before this fraction of steps
    es_min_steps = int(max_steps * max(0.0, min(1.0, es_warm_frac)))
    es_cb   = None
    if es_enabled:
        es_cb = EarlyStoppingWithWarmup(
            monitor="val_loss",
            mode="min",
            patience=es_patience,
            min_delta=es_min_delta,
            verbose=True,
            min_steps=es_min_steps,
            check_on_train_epoch_end=False,
        )
    rop_cfg = cfg["training"].get("reduce_on_plateau", {"factor":0.5,"patience":2,"min_lr":1e-6,"cooldown":0,"threshold":0.0})
    warmup_steps = int(cfg["training"].get("warmup_ratio", 0.03) * max_steps)

    print(
        "[SFT] "
        f"max_steps={max_steps} lr={base_lr:.6g} warmup_steps={warmup_steps} "
        f"es(patience={es_patience}, min_delta={es_min_delta}, warmup_fraction={es_warm_frac}) "
        f"rop(patience={rop_cfg.get('patience', 2)}, factor={rop_cfg.get('factor', 0.5)}, threshold={rop_cfg.get('threshold', 0.0)})"
    )
    rop_cb  = ReduceLROnPlateauOnVal(
        patience=rop_cfg.get("patience", 2),
        factor=  rop_cfg.get("factor", 0.5),
        min_lr=  rop_cfg.get("min_lr", 1e-6),
        cooldown=rop_cfg.get("cooldown", 0),
        threshold=rop_cfg.get("threshold", 0.0),
        monitor="val_loss", mode="min", warmup_steps=warmup_steps
    )
    lr_mon  = LearningRateMonitor(logging_interval="epoch")

    # Trainer settings
    # Align validation and snapshot saving with the end of each epoch ("epic end").
    # Define an epoch by optimizer steps using steps_per_epoch and grad_accum_steps.
    steps_per_epoch = int(cfg["training"].get("steps_per_epoch", 0) or 0)
    grad_accum_steps = int(cfg["training"].get("grad_accum_steps", 1))
    # Number of training batches (forward passes) per epoch
    batches_per_epoch = steps_per_epoch * max(1, grad_accum_steps) if steps_per_epoch > 0 else None

    callbacks_list = [
        ckpt_cb,
        rop_cb,
        lr_mon,
        WarmupLRCallback(warmup_steps),
    ]
    if es_cb is not None:
        callbacks_list.insert(2, es_cb)

    # Limit per-epoch batches to enforce the above epoch definition. If not provided,
    # fall back to Lightning defaults or user override.
    limit_batches_cfg = cfg["training"].get("limit_train_batches", None)
    if batches_per_epoch is not None and batches_per_epoch > 0 and limit_batches_cfg is None:
        limit_train_batches = batches_per_epoch
    else:
        limit_train_batches = limit_batches_cfg if limit_batches_cfg is not None else 1.0

    # Compute max_epochs from desired total optimizer steps. This ensures validation happens
    # exactly at the end of each logical epoch.
    if steps_per_epoch and steps_per_epoch > 0:
        max_epochs = int(math.ceil(max_steps / float(steps_per_epoch)))
    else:
        # Fallback: approximate 10 epochs across training if steps_per_epoch not set
        approx_epochs = 10
        steps_per_epoch = max(1, max_steps // approx_epochs)
        batches_per_epoch = steps_per_epoch * max(1, grad_accum_steps)
        if limit_batches_cfg is None:
            limit_train_batches = batches_per_epoch
        max_epochs = approx_epochs

    os.makedirs("logs", exist_ok=True)
    loggers = [
        TensorBoardLogger(save_dir="logs", name="lightning_logs"),
        CSVLogger(save_dir="logs", name="lightning_logs"),
    ]
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_steps=max_steps,
        max_epochs=max_epochs,
        precision=cfg["training"]["precision"],
        accumulate_grad_batches=cfg["training"]["grad_accum_steps"],
        log_every_n_steps=10,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
        limit_train_batches=limit_train_batches,
        num_sanity_val_steps=0,
        logger=loggers,
        callbacks=callbacks_list,
    )

    # Start SFT training (do not pass ckpt_path; we've already loaded weights if selected)
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # End-of-training: no extra copies; best and last are already maintained during training.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_finetune.yaml")
    args = parser.parse_args()
    main(args.config)
