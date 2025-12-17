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
import time
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim import AdamW
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
    def __init__(self, warmup_ratio: float, max_steps: int, min_lr_ratio: float = 0.0):
        self.warmup_steps = max(1, int(warmup_ratio * max_steps))
        self.max_steps = max_steps
        try:
            r = float(min_lr_ratio)
        except Exception:
            r = 0.0
        self.min_lr_ratio = min(1.0, max(0.0, r))

    def __call__(self, step: int) -> float:
        # Scheduler `step` is 0-indexed in most PyTorch schedulers; avoid returning 0
        # so the first scheduled LR is warmup_steps^{-1} * base_lr.
        s = int(step) + 1
        if s <= self.warmup_steps:
            return s / self.warmup_steps
        denom = max(1, self.max_steps - self.warmup_steps)
        progress = (s - self.warmup_steps) / denom
        progress = min(1.0, max(0.0, float(progress)))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Apply an LR floor so the schedule approaches (min_lr_ratio * base_lr) at the end.
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos


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
        self._compiled_forward = None
        self._using_fused_adamw = False
        tcfg = cfg.get("training", {})
        compile_cfg = tcfg.get("torch_compile", None)
        accelerator = str(cfg.get("hardware", {}).get("accelerator", "auto")).lower()
        if compile_cfg is None:
            self._torch_compile_enabled = bool(
                (accelerator in {"cuda", "auto"}) and torch.cuda.is_available()
            )
        else:
            self._torch_compile_enabled = bool(compile_cfg)
        self._torch_compile_mode = str(tcfg.get("torch_compile_mode", "default"))
        
        # Cap model size ~200M params for lightweight usage
        param_count = sum(p.numel() for p in self.net.parameters())
        print(f"Model initialized with {param_count:,} parameters")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._compiled_forward is not None:
            compiler_mod = getattr(torch, "compiler", None)
            mark_step_begin = getattr(compiler_mod, "cudagraph_mark_step_begin", None)
            if mark_step_begin is not None:
                try:
                    mark_step_begin()
                except Exception:
                    pass
            return self._compiled_forward(input_ids)
        return self.net(input_ids)

    def on_fit_start(self) -> None:
        if not getattr(self, "_torch_compile_enabled", False):
            return
        if self._compiled_forward is not None:
            return
        if getattr(self, "device", None) is None or self.device.type != "cuda":
            return
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            return

        try:
            seq_len = int(self.cfg.get("training", {}).get("seq_len", 0) or 0)
        except Exception:
            seq_len = 0
        if seq_len > 0:
            try:
                head_dim = int(self.net.config.d_model) // int(self.net.config.n_heads)
                with torch.no_grad():
                    self.net._maybe_build_rope(
                        seq_len,
                        head_dim,
                        float(self.net.config.rope_theta),
                        self.device,
                        torch.float32,
                    )
            except Exception as exc:
                print(f"[Train] RoPE cache prebuild failed; continuing: {exc}")

        try:
            self._compiled_forward = torch.compile(
                self.net.forward, mode=self._torch_compile_mode
            )
            print(f"[Train] torch.compile enabled (mode={self._torch_compile_mode})")
        except TypeError:
            try:
                self._compiled_forward = torch.compile(self.net.forward)
                print("[Train] torch.compile enabled")
            except Exception as exc:
                self._compiled_forward = None
                print(f"[Train] torch.compile failed; continuing without: {exc}")
        except Exception as exc:
            self._compiled_forward = None
            print(f"[Train] torch.compile failed; continuing without: {exc}")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch['input_ids']
        labels = batch['labels']
        logits = self(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch['input_ids']
        labels = batch['labels']
        logits = self(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(10.0, device=loss.device)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        ppl = torch.exp(loss)
        self.log('val_ppl', ppl, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        if getattr(self, "_using_fused_adamw", False):
            return
        return super().configure_gradient_clipping(
            optimizer, gradient_clip_val, gradient_clip_algorithm
        )

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

        param_groups = [
            {'params': decay_params, 'weight_decay': wd},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        tcfg = self.cfg.get("training", {})
        fused_cfg = tcfg.get("fused_adamw", None)
        accelerator = str(self.cfg.get("hardware", {}).get("accelerator", "auto")).lower()
        if fused_cfg is None:
            want_fused = bool((accelerator in {"cuda", "auto"}) and torch.cuda.is_available())
        else:
            want_fused = bool(fused_cfg)

        if want_fused:
            try:
                optimizer = AdamW(
                    param_groups,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    fused=True,
                )
                print("[Train] Using fused AdamW")
                self._using_fused_adamw = True
            except Exception as exc:
                optimizer = AdamW(param_groups, lr=lr, betas=betas, eps=eps)
                print(f"[Train] Fused AdamW unavailable; falling back: {exc}")
                self._using_fused_adamw = False
        else:
            optimizer = AdamW(param_groups, lr=lr, betas=betas, eps=eps)
            self._using_fused_adamw = False

        lr_schedule = str(self.cfg["training"].get("lr_schedule", "warmup_cosine")).lower()
        if lr_schedule in {"warmup_cosine", "cosine"}:
            max_steps = int(self.cfg["training"]["max_steps"])
            warmup_ratio = float(self.cfg["training"].get("warmup_ratio", 0.0))
            min_lr_ratio = float(self.cfg["training"].get("min_lr_ratio", 0.0) or 0.0)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=WarmupCosine(warmup_ratio, max_steps, min_lr_ratio=min_lr_ratio)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        # Fallback: constant LR (or external callbacks can mutate optimizer LR).
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
        self._resume_step: Optional[int] = None

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        try:
            step = checkpoint.get("global_step", None)
        except Exception:
            step = None
        if step is None:
            return
        try:
            self._resume_step = int(step)
        except Exception:
            self._resume_step = None

    def on_fit_start(self, trainer, pl_module):
        return

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not trainer.optimizers:
            return
        if self._resume_step is not None:
            try:
                step_now = int(getattr(trainer, "global_step", 0) or 0)
            except Exception:
                step_now = 0
            if step_now < int(self._resume_step):
                return

        if self.base_lrs is None:
            opt = trainer.optimizers[0]
            self.base_lrs = [g.get('lr', 0.0) for g in opt.param_groups]

        if self.base_lrs is None:
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

    # Prefer explicit "last" checkpoints if they exist; these are the right choice
    # for continuing training because they include the latest optimizer/scheduler state.
    # They often don't encode step in filename (e.g. last.ckpt, last-v2.ckpt), so
    # step-parsing would incorrectly deprioritize them.
    last_candidates = glob.glob(os.path.join(config_ckpt_dir, "last*.ckpt"))
    last_candidates = [p for p in last_candidates if os.path.isfile(p)]
    if last_candidates:
        last_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return last_candidates[0]

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
    def __init__(self, log_file: Path, print_to_stdout: bool = False):
        super().__init__()
        self.log_file = log_file
        self._print_to_stdout = bool(print_to_stdout)
        self._header_written = False
        self._train_batches_seen = 0
        self._last_logged_step: Optional[int] = None
        self._last_log_time_s: Optional[float] = None
        self._last_logged_train_batches_seen: int = 0
        self._last_train_loss_step: Optional[float] = None
        self._last_val_loss: Optional[float] = None
        self._last_val_ppl: Optional[float] = None

    def _format_eta(self, seconds: Optional[float]) -> str:
        if seconds is None:
            return "-"
        try:
            if not math.isfinite(seconds) or seconds < 0:
                return "-"
            s = int(seconds)
        except Exception:
            return "-"
        h = s // 3600
        m = (s % 3600) // 60
        s = s % 60
        if h > 0:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:d}:{s:02d}"

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero and not self._header_written:
            self._append_line("timestamp | kind | step | max_steps | progress | eta | epoch | train_batches | mb_per_s | tok_per_s | train_loss_step | val_loss | val_ppl | lr")
            self._header_written = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return

        self._train_batches_seen += 1

        tcfg = getattr(pl_module, "cfg", {}).get("training", {}) if hasattr(pl_module, "cfg") else {}
        every_steps = int(tcfg.get("metrics_log_every_n_steps", 50) or 50)
        every_secs = float(tcfg.get("metrics_log_every_seconds", 60) or 60)
        step = int(trainer.global_step)

        now = time.monotonic()
        if self._last_log_time_s is None:
            self._last_log_time_s = now

        step_advanced = (self._last_logged_step is None) or (step != self._last_logged_step)
        log_on_step = (
            step_advanced
            and every_steps > 0
            and step > 0
            and (step % every_steps) == 0
        )
        log_on_time = every_secs > 0 and (now - float(self._last_log_time_s)) >= every_secs

        if not (log_on_step or log_on_time):
            return

        metrics = trainer.callback_metrics
        train_loss_step = metrics.get("train_loss_step")
        if train_loss_step is None:
            train_loss_step = metrics.get("train_loss")
        if train_loss_step is not None:
            try:
                self._last_train_loss_step = float(train_loss_step)
            except Exception:
                pass

        lr = None
        if trainer.optimizers:
            try:
                lr = trainer.optimizers[0].param_groups[0].get("lr", None)
            except Exception:
                lr = None

        dt = max(1e-6, now - float(self._last_log_time_s))
        mb_delta = max(0, int(self._train_batches_seen) - int(self._last_logged_train_batches_seen))
        mb_per_s = float(mb_delta) / dt
        micro_bsz = int(getattr(pl_module, "cfg", {}).get("training", {}).get("micro_batch_size", 1) or 1)
        seq_len = int(getattr(pl_module, "cfg", {}).get("training", {}).get("seq_len", 0) or 0)
        tok_per_s = float(micro_bsz * seq_len) * mb_per_s if seq_len > 0 else 0.0
        grad_accum = int(tcfg.get("grad_accum_steps", 1) or 1)
        opt_steps_per_s = (mb_per_s / float(grad_accum)) if grad_accum > 0 else 0.0

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        epoch = int(trainer.current_epoch)
        max_steps = int(getattr(trainer, "max_steps", 0) or 0)
        if max_steps > 0:
            pct = 100.0 * float(step) / float(max_steps)
            progress = f"{pct:.2f}%"
            eta_s = (float(max_steps - step) / opt_steps_per_s) if opt_steps_per_s > 0 else None
        else:
            progress = "-"
            eta_s = None
        eta = self._format_eta(eta_s)

        line = f"{ts} | kind=train | step={step} | max_steps={max_steps} | progress={progress} | eta={eta} | epoch={epoch}"
        line += f" | train_batches={int(self._train_batches_seen)} | mb_per_s={mb_per_s:.2f} | tok_per_s={tok_per_s:.0f}"
        line += f" | train_loss_step={float(self._last_train_loss_step):.6g}" if self._last_train_loss_step is not None else " | train_loss_step=-"
        line += f" | val_loss={float(self._last_val_loss):.6g}" if self._last_val_loss is not None else " | val_loss=-"
        line += f" | val_ppl={float(self._last_val_ppl):.6g}" if self._last_val_ppl is not None else " | val_ppl=-"
        line += f" | lr={float(lr):.6g}" if isinstance(lr, (int, float)) else " | lr=-"
        self._append_line(line)
        if self._print_to_stdout:
            try:
                trainer.print(line)
            except Exception:
                pass

        self._last_logged_step = step
        self._last_log_time_s = now
        self._last_logged_train_batches_seen = int(self._train_batches_seen)

    def on_validation_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        metrics = trainer.callback_metrics
        val_loss = metrics.get("val_loss")
        val_ppl = metrics.get("val_ppl")
        if val_loss is not None:
            try:
                self._last_val_loss = float(val_loss)
            except Exception:
                pass
        if val_ppl is not None:
            try:
                self._last_val_ppl = float(val_ppl)
            except Exception:
                pass
        lr = None
        if trainer.optimizers:
            try:
                lr = trainer.optimizers[0].param_groups[0].get("lr", None)
            except Exception:
                lr = None
        step = int(trainer.global_step)
        epoch = int(trainer.current_epoch)
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        max_steps = int(getattr(trainer, "max_steps", 0) or 0)
        if max_steps > 0:
            pct = 100.0 * float(step) / float(max_steps)
            progress = f"{pct:.2f}%"
        else:
            progress = "-"
        line = f"{ts} | kind=val | step={step} | max_steps={max_steps} | progress={progress} | eta=- | epoch={epoch}"
        line += f" | train_loss_step={float(self._last_train_loss_step):.6g}" if self._last_train_loss_step is not None else " | train_loss_step=-"
        line += f" | val_loss={float(self._last_val_loss):.6g}" if self._last_val_loss is not None else " | val_loss=-"
        line += f" | val_ppl={float(self._last_val_ppl):.6g}" if self._last_val_ppl is not None else " | val_ppl=-"
        line += f" | lr={float(lr):.6g}" if isinstance(lr, (int, float)) else " | lr=-"
        self._append_line(line)
        if self._print_to_stdout:
            try:
                trainer.print(line)
            except Exception:
                pass

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


class EarlyStoppingRespectConfig(RobustEarlyStopping):
    def __init__(self, *args, warmup_steps: int = 0, **kwargs):
        self._configured_patience = int(kwargs.get("patience", 0) or 0)
        self._configured_min_delta = float(kwargs.get("min_delta", 0.0) or 0.0)
        super().__init__(*args, **kwargs)
        self._warmup_steps = int(max(0, warmup_steps))

    def load_state_dict(self, state_dict):
        # When resuming, Lightning restores EarlyStopping internal counters such as
        # wait_count. If you change patience between runs, the restored wait_count can
        # cause an immediate stop (e.g. stop after only a few validations). We keep the
        # restored best score, but reset the counter and re-apply configured thresholds.
        super().load_state_dict(state_dict)
        try:
            if self._configured_patience > 0:
                self.patience = int(self._configured_patience)
        except Exception:
            pass
        try:
            self.min_delta = float(self._configured_min_delta)
        except Exception:
            pass
        try:
            self.wait_count = 0
        except Exception:
            pass

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_step < self._warmup_steps:
            return
        super().on_train_epoch_end(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        if trainer.global_step < self._warmup_steps:
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

    tcfg = cfg.get("training", {})
    escfg = tcfg.get("early_stopping", {})
    print(
        "[Config] "
        f"lr_schedule={tcfg.get('lr_schedule')} "
        f"lr={tcfg.get('lr')} "
        f"warmup_ratio={tcfg.get('warmup_ratio')} "
        f"min_lr_ratio={tcfg.get('min_lr_ratio', 0.0)} "
        f"max_steps={tcfg.get('max_steps')} "
        f"es_enabled={escfg.get('enabled', True)} "
        f"es_patience={escfg.get('patience')} "
        f"es_min_delta={escfg.get('min_delta')}"
    )

    # Suppress specific PyTorch Lightning warnings
    warnings.filterwarnings("ignore", message="You're resuming from a checkpoint that ended before the epoch ended")
    pl.seed_everything(cfg['training']['seed'])

    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from datasets.utils.logging import disable_progress_bar

        disable_progress_bar()
    except Exception:
        pass

    tokenizer = get_tokenizer(cfg)
    train_loader, val_loader = make_dataloaders(cfg, tokenizer)

    lit = LitCausalLM(cfg, tokenizer)
    
    # File logging
    log_file = setup_logging()
    print(f"Training logs: {log_file}")
    print_metrics_cfg = cfg.get('training', {}).get('print_metrics', None)
    print_to_stdout = (not sys.stdout.isatty()) if print_metrics_cfg is None else bool(print_metrics_cfg)
    metrics_logger_cb = MetricsLoggingCallback(log_file, print_to_stdout=print_to_stdout)

    # Resume logic: auto-resume from latest if exists
    ckpt_dir = cfg['training'].get('checkpoint_dir', 'checkpoints')
    auto_resume = bool(cfg['training'].get('auto_resume', True))
    resume_checkpoint = find_latest_checkpoint(ckpt_dir) if auto_resume else None
    if resume_checkpoint:
        print(f"Resuming from latest checkpoint: {resume_checkpoint}")
    else:
        print("Starting fresh training...")

    # Device selection
    accelerator = cfg['hardware'].get('accelerator', 'auto')
    devices = cfg['hardware'].get('devices', 1)

    # Validation/checkpoint cadence: treat `steps_per_epoch` / `save_every` as optimizer-step units.
    steps_per_epoch = int(cfg["training"].get("steps_per_epoch", 0) or 0)
    grad_accum = int(cfg["training"].get("grad_accum_steps", 1) or 1)
    save_every = int(cfg["training"].get("save_every", 0) or 0)
    val_check_interval = None
    if steps_per_epoch > 0:
        val_check_interval = max(1, steps_per_epoch * grad_accum)

    # Callbacks
    # NOTE: periodic checkpoints may be triggered on training-step boundaries
    # (via every_n_train_steps). At that time validation metrics like `val_loss`
    # may not exist yet, so avoid using `val_loss` in the filename template.
    ckpt_every_opt_steps = save_every if save_every > 0 else steps_per_epoch
    epoch_checkpoint_kwargs = dict(
        dirpath=ckpt_dir,
        filename='epoch={epoch}-step={step}',
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )
    if ckpt_every_opt_steps and ckpt_every_opt_steps > 0:
        epoch_checkpoint_kwargs["every_n_train_steps"] = max(1, int(ckpt_every_opt_steps) * grad_accum)
    epoch_checkpoint_cb = ModelCheckpoint(**epoch_checkpoint_kwargs)

    best_checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        save_last=False,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )

    es_cfg = cfg['training'].get('early_stopping', {})
    if es_cfg.get('enabled', True):
        max_steps = int(cfg['training']['max_steps'])
        warmup_fraction = float(es_cfg.get('warmup_fraction', 0.0) or 0.0)
        warmup_steps_es = int(max(0, warmup_fraction) * max_steps)
        early_stop_cb = EarlyStoppingRespectConfig(
            monitor='val_loss',
            mode='min',
            patience=es_cfg.get('patience', 5),
            min_delta=es_cfg.get('min_delta', 0.001),
            verbose=True,
            warmup_steps=warmup_steps_es,
        )
    else:
        early_stop_cb = None

    lr_schedule = str(cfg["training"].get("lr_schedule", "warmup_cosine")).lower()

    # Callback ordering matters: on_validation_end hooks run in list order. For plateau
    # training we want LR reductions to occur before early-stopping decisions.
    callbacks: List[pl.Callback] = [
        epoch_checkpoint_cb,
        best_checkpoint_cb,
        metrics_logger_cb,
    ]

    # Optional: plateau scheduler mode (mutates optimizer LR via callback).
    if lr_schedule in {"plateau", "reduce_on_plateau"}:
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
        warmup_cb = WarmupLRCallback(warmup_steps_cb)
        callbacks.extend([rop_cb, warmup_cb])

    if early_stop_cb is not None:
        callbacks.append(early_stop_cb)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_steps=int(cfg['training']['max_steps']),
        precision=int(cfg['training'].get('precision', 32)) if str(cfg['training'].get('precision', 32)).isdigit() else cfg['training'].get('precision', 32),
        accumulate_grad_batches=cfg['training'].get('grad_accum_steps', 1),
        fast_dev_run=bool(cfg['training'].get('fast_dev_run', False)),
        overfit_batches=cfg['training'].get('overfit_batches', 0.0),
        limit_train_batches=cfg['training'].get('limit_train_batches', 1.0),
        log_every_n_steps=cfg['training'].get('log_every_n_steps', 10),
        logger=pl.loggers.CSVLogger("logs"),
        enable_model_summary=bool(cfg['training'].get('enable_model_summary', False)),
        enable_progress_bar=bool(cfg['training'].get('enable_progress_bar', sys.stdout.isatty())),
        enable_checkpointing=True,
        callbacks=callbacks,
        gradient_clip_val=cfg['training'].get('gradient_clip_val', 1.0),
        limit_val_batches=cfg['training'].get('limit_val_batches', 1.0),
        val_check_interval=val_check_interval if val_check_interval is not None else 1.0,
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
