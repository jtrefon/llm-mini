import os
import yaml
import math
import logging
from datetime import datetime
from pathlib import Path
import re
import shutil
import glob
import warnings
from typing import Optional, List, Any, Dict

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import GPTMini, GPTConfig
from data import make_dataloaders, get_tokenizer

# Set defaults for convenience, but respect external envs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch
    torch.set_float32_matmul_precision("high")
    if torch.backends.cuda is not None:
        torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass


class WarmupCosine:
    def __init__(self, warmup_ratio: float, max_steps: int):
        self.warmup_steps = max(1, int(warmup_ratio * max_steps))
        self.max_steps = max_steps

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))


class LitCausalLM(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any], tokenizer: Any):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.tokenizer = tokenizer
        mcfg = GPTConfig(
            vocab_size=len(tokenizer),
            n_layers=cfg['model']['n_layers'],
            d_model=cfg['model']['d_model'],
            n_heads=cfg['model']['n_heads'],
            n_kv_heads=cfg['model']['n_kv_heads'],
            d_ff=cfg['model']['d_ff'],
            dropout=cfg['model']['dropout'],
            rope_theta=cfg['model']['rope_theta'],
            tie_embeddings=cfg['model']['tie_embeddings'],
            swa_window=cfg['model']['swa_window'],
            gradient_checkpointing=bool(cfg['model'].get('gradient_checkpointing', False)),
        )
        self.net = GPTMini(mcfg)
        
        # Cap model size ~200M params for lightweight usage
        param_count = sum(p.numel() for p in self.net.parameters())
        print(f"Model initialized with {param_count:,} parameters")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.net(input_ids)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch['input_ids']
        labels = batch['labels']
        logits = self(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch['input_ids']
        labels = batch['labels']
        logits = self(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(10.0, device=loss.device)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        ppl = torch.exp(loss)
        self.log('val_ppl', ppl, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Parameter groups: apply weight decay to weights (ndim>=2), not to biases/norms
        wd = float(self.cfg['training']['weight_decay'])
        lr = float(self.cfg['training']['lr'])
        betas = tuple(self.cfg['training']['betas'])
        eps = float(self.cfg['training']['eps'])

        decay_params = []
        nodecay_params = []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        optimizer = AdamW([
            {'params': decay_params, 'weight_decay': wd},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ], lr=lr, betas=betas, eps=eps)

        return optimizer


class ReduceLROnPlateauOnVal(pl.Callback):
    """Reduce LR when val_loss plateaus without conflicting with warmup."""
    def __init__(self, patience=2, factor=0.5, min_lr=1e-6, cooldown=0, threshold=0.0,
                 monitor="val_loss", mode="min", warmup_steps: int = 0):
        self.patience = int(patience)
        self.factor = float(factor)
        self.min_lr = float(min_lr)
        self.cooldown = int(cooldown)
        self.threshold = float(threshold)
        self.monitor = monitor
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.bad = 0
        self.cool = 0
        self._warmup_steps = int(max(0, warmup_steps))

    def _improved(self, cur: float) -> bool:
        return (self.best - cur) > self.threshold if self.mode == "min" else (cur - self.best) > self.threshold

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_step < self._warmup_steps:
            return
        if self.monitor not in trainer.callback_metrics:
            return
        cur = float(trainer.callback_metrics[self.monitor])
        if self._improved(cur):
            self.best = cur
            self.bad = 0
            if self.cool > 0:
                self.cool -= 1
            return
        self.bad += 1
        if self.cool > 0:
            self.cool -= 1
            return
        if self.bad > self.patience and trainer.optimizers:
            opt = trainer.optimizers[0]
            for pg in opt.param_groups:
                pg["lr"] = max(self.min_lr, pg["lr"] * self.factor)
            try:
                new_lr = opt.param_groups[0]["lr"]
                print(f"[Train] LR reduced to {new_lr:.6g}")
            except Exception:
                pass
            self.bad = 0
            self.cool = self.cooldown


class WarmupLRCallback(pl.Callback):
    """Linearly warm up LR over first N steps."""

    def __init__(self, warmup_steps: int):
        super().__init__()
        self.warmup_steps = max(1, int(warmup_steps))
        self.base_lrs = None

    def on_fit_start(self, trainer, pl_module):
        if not trainer.optimizers:
            return
        opt = trainer.optimizers[0]
        self.base_lrs = [g.get('lr', 0.0) for g in opt.param_groups]
        step = getattr(trainer, 'global_step', 0)
        if step >= self.warmup_steps:
            for g, base in zip(opt.param_groups, self.base_lrs):
                g['lr'] = base
        else:
            frac = float(step) / float(self.warmup_steps) if self.warmup_steps > 0 else 1.0
            for g, base in zip(opt.param_groups, self.base_lrs):
                g['lr'] = base * frac

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not trainer.optimizers or self.base_lrs is None:
            return
        step = trainer.global_step
        if step >= self.warmup_steps:
            return
        frac = float(step + 1) / float(self.warmup_steps)
        opt = trainer.optimizers[0]
        for g, base in zip(opt.param_groups, self.base_lrs):
            g['lr'] = base * frac


def find_latest_checkpoint(config_ckpt_dir: str = 'checkpoints') -> Optional[str]:
    """Find the most advanced checkpoint by filename step parsing."""
    patterns = [
        f"{config_ckpt_dir}/*.ckpt",
        "lightning_logs/version_*/checkpoints/*.ckpt",
    ]
    checkpoints = []
    for pat in patterns:
        checkpoints.extend(glob.glob(pat))

    if not checkpoints:
        return None

    def parse_step(path: str) -> int:
        name = os.path.basename(path)
        m = re.search(r"step=(\d+)", name)
        if m: return int(m.group(1))
        m = re.search(r"global_step=(\d+)", name)
        if m: return int(m.group(1))
        return -1

    scored = []
    for p in checkpoints:
        step = parse_step(p)
        mtime = os.path.getmtime(p)
        scored.append((step, mtime, p))

    # Sort by step desc, then mtime desc
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


class MetricsLoggingCallback(pl.Callback):
    """Logs key metrics to a rotating timestamped file in logs/ directory."""
    def __init__(self, log_file: Path):
        super().__init__()
        self.log_file = log_file

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        train_loss = metrics.get('train_loss_epoch')
        val_loss = metrics.get('val_loss')
        val_ppl  = metrics.get('val_ppl')
        lr = None
        if trainer.optimizers:
            try:
                lr = trainer.optimizers[0].param_groups[0].get('lr', None)
            except Exception:
                pass
        epoch = trainer.current_epoch
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = (
            f"{ts} | epoch={epoch}"
            f" | train_loss={float(train_loss):.4f}" if train_loss is not None else f"{ts} | epoch={epoch} | train_loss=-"
        )
        line += f" | val_loss={float(val_loss):.4f}" if val_loss is not None else " | val_loss=-"
        line += f" | val_ppl={float(val_ppl):.4f}" if val_ppl is not None else " | val_ppl=-"
        line += f" | lr={lr:.6f}" if isinstance(lr, (int, float)) else " | lr=-"
        self._append_line(line)

    def _append_line(self, text: str):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'a') as f:
            f.write(text + "\n")


class RobustEarlyStopping(EarlyStopping):
    """EarlyStopping that gracefully skips checks if the metric is missing."""
    def on_train_epoch_end(self, trainer, pl_module):
        # Skip if metric is not available yet (common at end of training epoch before validation)
        if self.monitor not in trainer.callback_metrics:
            return
        super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        # Skip if metric is not available
        if self.monitor not in trainer.callback_metrics:
            return
        super().on_validation_end(trainer, pl_module)


def setup_logging() -> Path:
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"training_{stamp}.log"
    return log_file


def main(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Suppress specific PyTorch Lightning warnings
    warnings.filterwarnings("ignore", message="You're resuming from a checkpoint that ended before the epoch ended")
    pl.seed_everything(cfg['training']['seed'])

    tokenizer = get_tokenizer(cfg)
    train_loader, val_loader = make_dataloaders(cfg, tokenizer)

    lit = LitCausalLM(cfg, tokenizer)
    
    # File logging
    log_file = setup_logging()
    print(f"Training logs: {log_file}")
    metrics_logger_cb = MetricsLoggingCallback(log_file)

    # Resume logic: auto-resume from latest if exists
    ckpt_dir = cfg['training'].get('checkpoint_dir', 'checkpoints')
    resume_checkpoint = find_latest_checkpoint(ckpt_dir)
    if resume_checkpoint:
        print(f"Resuming from latest checkpoint: {resume_checkpoint}")
    else:
        print("Starting fresh training...")

    # Device selection
    accelerator = cfg['hardware'].get('accelerator', 'auto')
    devices = cfg['hardware'].get('devices', 1)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='epoch={epoch}-step={step}-val_loss={val_loss:.3f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True,
        save_on_train_epoch_end=False,
    )

    es_cfg = cfg['training'].get('early_stopping', {})
    if es_cfg.get('enabled', True):
        early_stop_cb = RobustEarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=es_cfg.get('patience', 5),
            min_delta=es_cfg.get('min_delta', 0.001),
            verbose=True,
        )
    else:
        early_stop_cb = None

    rop_cfg = cfg['training'].get('reduce_on_plateau', {})
    warmup_steps_cb = max(1, int(cfg['training'].get('warmup_ratio', 0.0) * cfg['training']['max_steps']))
    rop_cb = ReduceLROnPlateauOnVal(
        patience=rop_cfg.get('patience', 3),
        factor=rop_cfg.get('factor', 0.5),
        min_lr=rop_cfg.get('min_lr', 1e-6),
        monitor='val_loss',
        mode='min',
        warmup_steps=warmup_steps_cb,
    )

    # Warmup callback
    max_steps = cfg['training']['max_steps']
    warmup_cb = WarmupLRCallback(warmup_steps_cb)

    callbacks = [c for c in [checkpoint_cb, early_stop_cb, rop_cb, metrics_logger_cb, warmup_cb] if c is not None]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_steps=max_steps,
        precision=cfg['training'].get('precision', 32),
        accumulate_grad_batches=cfg['training'].get('grad_accum_steps', 1),
        log_every_n_steps=cfg['training'].get('log_every_n_steps', 10),
        logger=pl.loggers.CSVLogger("logs"),
        enable_checkpointing=True,
        callbacks=callbacks,
        gradient_clip_val=cfg['training'].get('gradient_clip_val', 1.0),
        limit_val_batches=cfg['training'].get('limit_val_batches', 1.0),
    )

    print("Starting training loop...")
    trainer.fit(
        lit,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_checkpoint
    )
    print("Training completed.")

    # Save final
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'final.ckpt')
    trainer.save_checkpoint(ckpt_path)
    print(f"Saved final checkpoint to {ckpt_path}")


if __name__ == '__main__':
    main()
