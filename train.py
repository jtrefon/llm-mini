import os
# Set env early to affect torch/PL and dataloader workers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.pop("MallocStackLogging", None)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import yaml
import math
import logging
from datetime import datetime
from pathlib import Path
import re
import shutil
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import GPTMini, GPTConfig
from data import make_dataloaders, get_tokenizer
import glob
import warnings


class WarmupCosine:
    def __init__(self, warmup_ratio: float, max_steps: int):
        self.warmup_steps = max(1, int(warmup_ratio * max_steps))
        self.max_steps = max_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))


class EarlyStoppingRespectConfig(EarlyStopping):
    """EarlyStopping that preserves patience from config on resume.

    PyTorch Lightning restores callback state from checkpoints, including the
    "patience" and current "wait_count". If the checkpoint was created with a
    different patience (e.g., 3) and you later increase it in config (e.g., 5),
    the restored state would override your new value and stop too early.

    This subclass keeps the configured patience across resume, and optionally
    resets the wait counter so you don't immediately stop after resuming.
    """

    def __init__(self, *args, configured_patience: int | None = None, reset_wait_on_resume: bool = True, reset_best_on_resume: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep the configured patience regardless of what the checkpoint contains
        self._configured_patience = configured_patience if configured_patience is not None else self.patience
        self._reset_wait_on_resume = reset_wait_on_resume
        self._reset_best_on_resume = reset_best_on_resume
        self.patience = self._configured_patience

    def load_state_dict(self, state_dict):
        # Restore best_score and counters, then enforce new patience
        super().load_state_dict(state_dict)
        self.patience = self._configured_patience
        if self._reset_wait_on_resume:
            self.wait_count = 0
        if self._reset_best_on_resume:
            # Reset best score so patience counts from resume point
            if self.mode == 'min':
                self.best_score = torch.tensor(float('inf'), device=self.best_score.device)
            else:
                self.best_score = torch.tensor(float('-inf'), device=self.best_score.device)


class LitCausalLM(pl.LightningModule):
    def __init__(self, cfg, tokenizer):
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
            swa_window=cfg['model']['swa_window']
        )
        self.net = GPTMini(mcfg)

    def forward(self, input_ids):
        return self.net(input_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        logits = self(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        logits = self(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
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

        # Reduce LR on plateau (monitor validation loss)
        r_cfg = self.cfg['training'].get('reduce_on_plateau', {})
        plateau_sched = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(r_cfg.get('factor', self.cfg['training']['reduce_on_plateau']['factor'])),
            patience=r_cfg.get('patience', self.cfg['training']['reduce_on_plateau']['patience']),
            threshold=float(r_cfg.get('threshold', self.cfg['training']['reduce_on_plateau']['threshold'])),
            cooldown=r_cfg.get('cooldown', self.cfg['training']['reduce_on_plateau']['cooldown']),
            min_lr=float(r_cfg.get('min_lr', self.cfg['training']['reduce_on_plateau']['min_lr'])),
        )

        scheds = [
            {
                'scheduler': plateau_sched,
                'reduce_on_plateau': True,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        ]
        return [optimizer], scheds


class WarmupLRCallback(pl.Callback):
    """Linearly warm up LR over first N steps without interfering with plateau scheduler.

    It scales each param group lr from 0 -> base_lr across warmup_steps, then does nothing.
    This avoids using a step-based scheduler that would override ReduceLROnPlateau.
    """

    def __init__(self, warmup_steps: int):
        super().__init__()
        self.warmup_steps = max(1, int(warmup_steps))
        self.base_lrs = None

    def on_fit_start(self, trainer, pl_module):
        if not trainer.optimizers:
            return
        opt = trainer.optimizers[0]
        # Capture target/base lrs
        self.base_lrs = [g.get('lr', 0.0) for g in opt.param_groups]
        # Start from near-zero to ramp smoothly
        for g in opt.param_groups:
            g['lr'] = 0.0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not trainer.optimizers or self.base_lrs is None:
            return
        step = trainer.global_step  # counts seen batches across fit
        if step >= self.warmup_steps:
            return  # Warmup done; do not touch LR further
        frac = float(step + 1) / float(self.warmup_steps)
        opt = trainer.optimizers[0]
        for g, base in zip(opt.param_groups, self.base_lrs):
            g['lr'] = base * frac


def find_latest_checkpoint():
    """Find the most advanced checkpoint (by global step), or newest if unknown.

    Searches both `checkpoints/` and legacy `lightning_logs/.../checkpoints/`.
    Prefers higher parsed step from filename patterns, else falls back to mtime.
    """
    patterns = [
        "checkpoints/*.ckpt",  # current default dir
        "lightning_logs/version_*/checkpoints/*.ckpt",  # legacy PL dir
    ]
    checkpoints = []
    for pat in patterns:
        checkpoints.extend(glob.glob(pat))

    if not checkpoints:
        return None

    def parse_step(path: str) -> int:
        name = os.path.basename(path)
        # epoch={E}-step={S}-...
        m = re.search(r"step=(\d+)", name)
        if m:
            return int(m.group(1))
        # global_step={S}
        m = re.search(r"global_step=(\d+)", name)
        if m:
            return int(m.group(1))
        # legacy like "E-S.ckpt" -> treat as epoch-step
        m = re.match(r"(\d+)-(\d+)\.ckpt$", name)
        if m:
            try:
                return int(m.group(2))
            except Exception:
                return -1
        return -1

    scored = []
    for p in checkpoints:
        step = parse_step(p)
        mtime = os.path.getmtime(p)
        scored.append((step, mtime, p))

    # Sort by step desc, then mtime desc
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def discover_checkpoints():
    patterns = [
        "checkpoints/*.ckpt",
        "lightning_logs/version_*/checkpoints/*.ckpt",
    ]
    seen = set()
    result = []
    for pat in patterns:
        for p in glob.glob(pat):
            if p not in seen:
                seen.add(p)
                result.append(p)
    # newest first for display
    result.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return result


def parse_ckpt_metadata(path):
    """Best-effort parse of epoch, step, val_loss from filename."""
    name = os.path.basename(path)
    epoch = step = None
    val_loss = None
    m = re.search(r"epoch=(\d+).*step=(\d+).*val_loss=([0-9]+(?:\.[0-9]+)?)", name)
    if m:
        return int(m.group(1)), int(m.group(2)), float(m.group(3))
    m = re.search(r"epoch=(\d+).*step=(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2)), None
    m = re.search(r"global_step=(\d+)", name)
    if m:
        return None, int(m.group(1)), None
    m = re.match(r"(\d+)-(\d+)\.ckpt$", name)
    if m:
        try:
            return int(m.group(1)), int(m.group(2)), None
        except Exception:
            pass
    return epoch, step, val_loss


def select_checkpoint_interactively():
    ckpts = discover_checkpoints()
    if not ckpts:
        return None

    infos = []
    for p in ckpts:
        ep, st, vl = parse_ckpt_metadata(p)
        infos.append({
            'path': p,
            'name': os.path.basename(p),
            'epoch': ep,
            'step': st,
            'val_loss': vl,
            'mtime': os.path.getmtime(p),
        })

    def best_by_val():
        c = [i for i in infos if i['val_loss'] is not None]
        return min(c, key=lambda i: i['val_loss']) if c else None

    def best_by_step():
        c = [i for i in infos if i['step'] is not None]
        return max(c, key=lambda i: i['step']) if c else None

    def newest():
        return max(infos, key=lambda i: i['mtime'])

    print("\n🔍 Found the following checkpoints:\n")
    for idx, i in enumerate(infos, 1):
        ep = i['epoch'] if i['epoch'] is not None else '-'
        st = i['step'] if i['step'] is not None else '-'
        vl = f"{i['val_loss']:.3f}" if i['val_loss'] is not None else '-'
        print(f"[{idx}] {i['name']}\t(epoch={ep}, step={st}, val_loss={vl})")

    print("\nEnter a number to resume from that checkpoint.")
    print("Or press: 'b' = lowest val_loss, 's' = highest step, 'n' = start fresh, Enter = default (b>s>newest)")

    while True:
        sel = input("Selection: ").strip().lower()
        if sel == '':
            choice = best_by_val() or best_by_step() or newest()
            print(f"Default selection: {choice['name']}")
            return choice['path']
        if sel in ('n', 'no'):
            return None
        if sel == 'b':
            ch = best_by_val()
            if ch:
                print(f"Best val_loss: {ch['name']}")
                return ch['path']
            print("No checkpoints with val_loss found.")
            continue
        if sel == 's':
            ch = best_by_step()
            if ch:
                print(f"Highest step: {ch['name']}")
                return ch['path']
            print("No checkpoints with step found.")
            continue
        if sel.isdigit():
            i = int(sel)
            if 1 <= i <= len(infos):
                print(f"Selected: {infos[i-1]['name']}")
                return infos[i-1]['path']
        print("Invalid selection. Try again.")


def prompt_checkpoint_recovery(checkpoint_path):
    """Ask user if they want to recover from checkpoint."""
    checkpoint_name = os.path.basename(checkpoint_path)

    print(f"\n🔍 Found existing checkpoint: {checkpoint_name}")
    print(f"📁 Location: {checkpoint_path}")

    # Try to extract epoch and step info from filename (several patterns)
    try:
        epoch_part = None
        step_part = None
        m = re.search(r"epoch=(\d+).*step=(\d+)", checkpoint_name)
        if m:
            epoch_part, step_part = m.group(1), m.group(2)
        else:
            m = re.search(r"global_step=(\d+)", checkpoint_name)
            if m:
                step_part = m.group(1)
            else:
                m = re.match(r"(\d+)-(\d+)\.ckpt$", checkpoint_name)
                if m:
                    epoch_part, step_part = m.group(1), m.group(2)
        if epoch_part or step_part:
            info = []
            if epoch_part:
                info.append(f"Epoch {epoch_part}")
            if step_part:
                info.append(f"Step {step_part}")
            print("📊 Checkpoint info: " + ", ".join(info))
    except Exception:
        pass

    while True:
        response = input("\nDo you want to RESUME training from this checkpoint? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


class MetricsLoggingCallback(pl.Callback):
    """Logs key metrics to a rotating timestamped file in logs/ directory."""
    def __init__(self, log_file: Path):
        super().__init__()
        self.log_file = log_file

    def on_fit_start(self, trainer, pl_module):
        self._log_header()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Collect metrics at the end of validation epoch (where val_loss is available)
        metrics = trainer.callback_metrics
        train_loss = metrics.get('train_loss_epoch')
        val_loss = metrics.get('val_loss')
        # Current LR from first optimizer param group
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
        line += f" | lr={lr:.6f}" if isinstance(lr, (int, float)) else " | lr=-"
        self._append_line(line)

    def on_train_epoch_start(self, trainer, pl_module):
        # Log LR at the start of the new epoch, after any ReduceLROnPlateau step
        lr = None
        if trainer.optimizers:
            try:
                lr = trainer.optimizers[0].param_groups[0].get('lr', None)
            except Exception:
                pass
        epoch = trainer.current_epoch
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"{ts} | epoch={epoch} | train_loss=- | val_loss=- | lr={lr:.6f}" if isinstance(lr, (int, float)) else f"{ts} | epoch={epoch} | train_loss=- | val_loss=- | lr=-"
        self._append_line(line)

    def _log_header(self):
        header = "timestamp | epoch | train_loss | val_loss | lr"
        self._append_line(header)

    def _append_line(self, text: str):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'a') as f:
            f.write(text + "\n")


def setup_logging() -> Path:
    """Prepare logs directory and return a new log file path."""
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"training_{stamp}.log"
    # Optionally create/refresh a 'latest_training.log' symlink
    latest = logs_dir / 'latest_training.log'
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(log_file.name)
    except Exception:
        pass
    return log_file


def main(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # env already set at import time

    # Suppress specific PyTorch Lightning warnings
    warnings.filterwarnings("ignore", message="You're resuming from a checkpoint that ended before the epoch ended")

    pl.seed_everything(cfg['training']['seed'])

    tokenizer = get_tokenizer(cfg)
    train_loader, val_loader = make_dataloaders(cfg, tokenizer)

    # Debug: Check if validation loader has data
    print(f"Training dataset size: {len(train_loader.dataset)} sequences")
    print(f"Validation dataset size: {len(val_loader.dataset)} sequences")

    if len(val_loader.dataset) == 0:
        print("!  Warning: Validation dataset is empty!")
        print("? Consider reducing train_docs or increasing val_docs in config.yaml")

    lit = LitCausalLM(cfg, tokenizer)

    # File logging
    log_file = setup_logging()
    print(f"Training logs will be saved to: {log_file}")
    metrics_logger_cb = MetricsLoggingCallback(log_file)

    # Check for existing checkpoints and allow interactive selection if possible
    resume_checkpoint = None
    any_ckpt = discover_checkpoints()
    if any_ckpt:
        try:
            if os.isatty(0):
                resume_checkpoint = select_checkpoint_interactively()
                if resume_checkpoint:
                    print(f"Will resume from: {resume_checkpoint}")
                else:
                    print("Starting fresh training...")
            else:
                # Non-interactive: pick best available automatically
                auto = find_latest_checkpoint()
                if auto:
                    resume_checkpoint = auto
                    print(f"Non-interactive resume from: {auto}")
                else:
                    print("Starting fresh training...")
        except Exception as e:
            print(f"! Checkpoint selection failed ({e}). Falling back to fresh training.")
    else:
        print("No existing checkpoints found. Starting fresh training...")

    # Device selection
    accelerator = cfg['hardware']['accelerator']
    devices = cfg['hardware']['devices']

    # Always write new checkpoints to a single, consistent directory
    ckpt_dir = cfg['training'].get('checkpoint_dir', 'checkpoints')
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='epoch={epoch}-step={step}-val_loss={val_loss:.3f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=True,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )

    # Step-based checkpoints to allow resume even before first validation
    step_save_every = cfg['training'].get('save_every', None)
    step_checkpoint_cb = None
    if step_save_every and isinstance(step_save_every, int) and step_save_every > 0:
        step_checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='global_step={step}',
            save_top_k=-1,
            every_n_train_steps=step_save_every,
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
        )

    # Early stopping on validation loss, evaluated after validation not train epoch end
    es_cfg = cfg['training'].get('early_stopping', {})
    early_stop_cb = EarlyStoppingRespectConfig(
        monitor='val_loss',
        mode='min',
        patience=es_cfg.get('patience', cfg['training']['early_stopping']['patience']),
        min_delta=es_cfg.get('min_delta', cfg['training']['early_stopping']['min_delta']),
        check_on_train_epoch_end=False,
        configured_patience=es_cfg.get('patience', cfg['training']['early_stopping']['patience']),
        reset_wait_on_resume=True,
        reset_best_on_resume=True,
        verbose=True,
    )

    # Optionally cap steps per epoch to sync with validation cadence
    steps_per_epoch = cfg['training'].get('steps_per_epoch', None)

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=devices,
        max_steps=cfg['training']['max_steps'],
        precision=cfg['training']['precision'],
        accumulate_grad_batches=cfg['training']['grad_accum_steps'],
        log_every_n_steps=cfg['training'].get('log_every_n_steps', 10),
        logger=False,  # Disable logging to avoid TensorBoard warnings
        enable_checkpointing=True,
        callbacks=[c for c in [checkpoint_cb, step_checkpoint_cb, early_stop_cb, metrics_logger_cb] if c is not None],
        gradient_clip_val=cfg['training'].get('gradient_clip_val', 1.0),
        limit_val_batches=cfg['training'].get('limit_val_batches', 1.0),   # Use full validation set
    )

    if steps_per_epoch is not None:
        trainer_kwargs['limit_train_batches'] = steps_per_epoch

    # Add warmup callback that doesn't conflict with plateau
    max_steps = cfg['training']['max_steps']
    warmup_steps = max(1, int(cfg['training']['warmup_ratio'] * max_steps))
    warmup_cb = WarmupLRCallback(warmup_steps)
    trainer_kwargs['callbacks'].append(warmup_cb)

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(
        lit,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_checkpoint  # This will resume from checkpoint if provided
    )

    # Save final checkpoint
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'final.ckpt')
    trainer.save_checkpoint(ckpt_path)
    print(f"Saved final checkpoint to {ckpt_path}")

    # Also copy best model to a stable name for easy discovery
    best_path = checkpoint_cb.best_model_path
    if best_path and os.path.exists(best_path):
        best_copy = os.path.join(ckpt_dir, 'best.ckpt')
        try:
            shutil.copy2(best_path, best_copy)
            print(f"Best checkpoint copied to {best_copy}")
        except Exception as e:
            print(f"Warning: could not copy best checkpoint: {e}")


if __name__ == '__main__':
    main()
