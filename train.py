import os
import yaml
import math
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from model import GPTMini, GPTConfig
from data import make_dataloaders, get_tokenizer


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


def main(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # tends to hang on osx if true/enabled
    pl.seed_everything(cfg['training']['seed'])

    tokenizer = get_tokenizer(cfg)
    train_loader = make_dataloaders(cfg, tokenizer)

    lit = LitCausalLM(cfg, tokenizer)

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
        val_check_interval=None
    )

    trainer.fit(lit, train_dataloaders=train_loader)

    # Save final checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    ckpt_path = 'checkpoints/final.ckpt'
    trainer.save_checkpoint(ckpt_path)
    print(f"Saved final checkpoint to {ckpt_path}")


if __name__ == '__main__':
    main()
