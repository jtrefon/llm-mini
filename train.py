import os
import yaml
import math
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from model import GPTMini, GPTConfig
from data import make_dataloaders, get_tokenizer
import glob


class WarmupCosine:
    def __init__(self, warmup_ratio: float, max_steps: int):
        self.warmup_steps = max(1, int(warmup_ratio * max_steps))
        self.max_steps = max_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return step / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))


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
        optimizer = AdamW(self.parameters(), lr=self.cfg['training']['lr'], betas=tuple(self.cfg['training']['betas']), eps=self.cfg['training']['eps'], weight_decay=self.cfg['training']['weight_decay'])
        max_steps = self.cfg['training']['max_steps']
        warm = WarmupCosine(self.cfg['training']['warmup_ratio'], max_steps)
        scheduler = LambdaLR(optimizer, lr_lambda=warm)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


def find_latest_checkpoint():
    """Find the latest checkpoint in lightning_logs directory."""
    checkpoint_pattern = "lightning_logs/version_*/checkpoints/*.ckpt"
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        return None

    # Sort by modification time (newest first)
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint


def prompt_checkpoint_recovery(checkpoint_path):
    """Ask user if they want to recover from checkpoint."""
    checkpoint_name = os.path.basename(checkpoint_path)

    print(f"\n🔍 Found existing checkpoint: {checkpoint_name}")
    print(f"📁 Location: {checkpoint_path}")

    # Try to extract epoch and step info from filename
    if "epoch=" in checkpoint_name and "step=" in checkpoint_name:
        try:
            epoch_part = checkpoint_name.split("epoch=")[1].split("-")[0]
            step_part = checkpoint_name.split("step=")[1].split(".")[0]
            print(f"📊 Checkpoint info: Epoch {epoch_part}, Step {step_part}")
        except:
            pass

    while True:
        response = input("\n❓ Do you want to RESUME training from this checkpoint? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def main(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # tends to hang on osx if true/enabled
    pl.seed_everything(cfg['training']['seed'])

    tokenizer = get_tokenizer(cfg)
    train_loader, val_loader = make_dataloaders(cfg, tokenizer)

    lit = LitCausalLM(cfg, tokenizer)

    # Check for existing checkpoints
    latest_checkpoint = find_latest_checkpoint()
    resume_checkpoint = None

    if latest_checkpoint:
        if prompt_checkpoint_recovery(latest_checkpoint):
            resume_checkpoint = latest_checkpoint
            print(f"✅ Will resume from: {latest_checkpoint}")
        else:
            print("🆕 Starting fresh training...")
    else:
        print("🆕 No existing checkpoints found. Starting fresh training...")

    # Device selection
    accelerator = cfg['hardware']['accelerator']
    devices = cfg['hardware']['devices']

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_steps=cfg['training']['max_steps'],
        precision=cfg['training']['precision'],
        accumulate_grad_batches=cfg['training']['grad_accum_steps'],
        log_every_n_steps=10,
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        val_check_interval=200,  # Validate every 200 steps
        limit_val_batches=1.0,   # Use full validation set
    )

    trainer.fit(
        lit,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_checkpoint  # This will resume from checkpoint if provided
    )

    # Save final checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = 'checkpoints/final.ckpt'
    trainer.save_checkpoint(ckpt_path)
    print(f"Saved final checkpoint to {ckpt_path}")


if __name__ == '__main__':
    main()
