# finetune.py
import os, glob, math, yaml
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer

from model import GPTMini, GPTConfig
from train import LitCausalLM, WarmupCosine  # reuse your existing LightningModule & scheduler

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
    Labels are -100 for prompt tokens (ignored by CE), real ids for response tokens.
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

        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + resp_ids  # mask prompt

        # hard truncate from the left if too long (keep the tail where the response lives)
        if len(input_ids) > self.seq_len:
            cut = len(input_ids) - self.seq_len
            input_ids = input_ids[cut:]
            labels    = labels[cut:]
            # ensure at least some supervised tokens remain
            if all(l == -100 for l in labels):
                input_ids = (prompt_ids + resp_ids)[-self.seq_len:]
                labels    = ([-100]*len(prompt_ids) + resp_ids)[-self.seq_len:]

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
    return {"input_ids": input_ids, "labels": labels}


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
    train_loader = DataLoader(
        train_ds, batch_size=micro_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
        collate_fn=lambda b: collate_sft(b, pad_id, seq_len),
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=micro_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
        collate_fn=lambda b: collate_sft(b, pad_id, seq_len),
        persistent_workers=False,
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
                 monitor="val_loss", mode="min"):
        self.patience = patience; self.factor = factor; self.min_lr = min_lr
        self.cooldown = cooldown; self.threshold = threshold
        self.monitor = monitor; self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.bad = 0; self.cool = 0

    def _improved(self, cur):
        return (self.best - cur) > self.threshold if self.mode == "min" else (cur - self.best) > self.threshold

    def on_validation_end(self, trainer, pl_module):
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
        if self.bad > self.patience:
            opt = trainer.optimizers[0]
            for pg in opt.param_groups:
                pg["lr"] = max(self.min_lr, pg["lr"] * self.factor)
            pl_module.log("lr_reduced_to", opt.param_groups[0]["lr"], prog_bar=True)
            self.bad = 0; self.cool = self.cooldown


# =========================
# Main
# =========================
def main(config_path="config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    pl.seed_everything(cfg["training"]["seed"])

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer_name"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token

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
    )
    lit = LitCausalLM(cfg, tok)
    lit.net = GPTMini(mcfg)  # ensure module matches mcfg

    # Optimizer & schedule: gentle for SFT
    base_lr = cfg["training"].get("lr", 5e-5)
    cfg["training"]["lr"] = base_lr  # reflect in hparams
    optimizer = torch.optim.AdamW(
        lit.parameters(),
        lr=base_lr,
        betas=tuple(cfg["training"]["betas"]),
        eps=cfg["training"]["eps"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    max_steps = cfg["training"].get("max_steps", 1000)
    warm = WarmupCosine(cfg["training"].get("warmup_ratio", 0.03), max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm)

    # Override Lightning's optimizer hook to use our SFT settings
    lit.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }

    # Hardware
    accelerator = cfg["hardware"]["accelerator"]
    devices = cfg["hardware"]["devices"]

    # Callbacks
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    ckpt_dir = cfg["training"].get("ckpt_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="sft-best",
        monitor="val_loss", mode="min", save_top_k=1, save_last=True
    )
    es_cfg  = cfg["training"].get("early_stopping", {"patience": 3, "min_delta": 0.0})
    es_cb   = EarlyStopping(monitor="val_loss", mode="min",
                            patience=es_cfg.get("patience", 3),
                            min_delta=es_cfg.get("min_delta", 0.0),
                            verbose=True)
    rop_cfg = cfg["training"].get("reduce_on_plateau", {"factor":0.5,"patience":2,"min_lr":1e-6,"cooldown":0,"threshold":0.0})
    rop_cb  = ReduceLROnPlateauOnVal(
        patience=rop_cfg.get("patience", 2),
        factor=  rop_cfg.get("factor", 0.5),
        min_lr=  rop_cfg.get("min_lr", 1e-6),
        cooldown=rop_cfg.get("cooldown", 0),
        threshold=rop_cfg.get("threshold", 0.0),
        monitor="val_loss", mode="min"
    )
    lr_mon  = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_steps=max_steps,
        precision=cfg["training"]["precision"],
        accumulate_grad_batches=cfg["training"]["grad_accum_steps"],
        log_every_n_steps=10,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        val_check_interval=cfg["training"].get("eval_every", 100),
        callbacks=[ckpt_cb, es_cb, rop_cb, lr_mon],
    )

    # Resume from a checkpoint if found
    resume_ckpt = pick_checkpoint(ckpt_dir)
    if resume_ckpt:
        print(f"[SFT] Resuming from checkpoint: {resume_ckpt}")
        trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt)
    else:
        trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save final SFT checkpoint
    final_path = os.path.join(ckpt_dir, "sft-final.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"[SFT] Saved checkpoint to {final_path}")


if __name__ == "__main__":
    main()
