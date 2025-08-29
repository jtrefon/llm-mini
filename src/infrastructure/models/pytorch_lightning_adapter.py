"""PyTorch Lightning adapter for domain GPT model.

This adapter bridges the domain model (framework-independent) with
PyTorch Lightning (framework-specific), providing training capabilities
while maintaining clean separation of concerns.
"""
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from typing import Any, Optional

from src.domain.models.gpt_mini import GPTMini
from src.domain.entities.model_config import ModelConfiguration
from src.domain.entities.training_config import TrainingConfiguration
from src.shared.factories.optimizer_factory import create_adamw_optimizer
from src.shared.factories.scheduler_factory import create_warmup_cosine_scheduler


class PyTorchLightningGPTModel(pl.LightningModule):
    """PyTorch Lightning adapter for GPT domain model.

    This adapter provides PyTorch Lightning integration for the domain GPT model,
    handling training, validation, optimization, and logging while keeping the
    core model logic in the domain layer.
    """

    def __init__(self, model_config: ModelConfiguration, training_config: TrainingConfiguration):
        """Initialize PyTorch Lightning model adapter.

        Args:
            model_config: Model architecture configuration
            training_config: Training hyperparameters and settings
        """
        super().__init__()
        self.save_hyperparameters()

        # Domain model (framework-independent)
        self.model = GPTMini(model_config)

        # Training configuration
        self.training_config = training_config

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            input_ids: Input token sequence

        Returns:
            Model logits
        """
        return self.model(input_ids)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute single training step.

        Args:
            batch: Batch containing input_ids and labels
            batch_idx: Batch index

        Returns:
            Training loss
        """
        input_ids = batch['input_ids']
        labels = batch['labels']

        # Forward pass
        logits = self(input_ids)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Execute single validation step.

        Args:
            batch: Batch containing input_ids and labels
            batch_idx: Batch index

        Returns:
            Validation loss
        """
        input_ids = batch['input_ids']
        labels = batch['labels']

        # Forward pass
        logits = self(input_ids)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Create optimizer using shared factory
        optimizer = create_adamw_optimizer(self.parameters(), self.training_config)

        # Create scheduler using shared factory
        scheduler = create_warmup_cosine_scheduler(optimizer, self.training_config)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 128,
                temperature: float = 0.8, top_p: float = 0.9,
                eos_token_id: Optional[int] = None) -> torch.Tensor:
        """Generate text using the domain model.

        Args:
            input_ids: Input token sequence
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            eos_token_id: End-of-sequence token ID

        Returns:
            Generated token sequence
        """
        return self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id
        )

    def get_model_summary(self) -> str:
        """Get model architecture summary.

        Returns:
            String summary of model architecture
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return f"""
GPT Mini Model Summary:
- Layers: {self.model.config.n_layers}
- Model dimension: {self.model.config.d_model}
- Attention heads: {self.model.config.n_heads}
- KV heads: {self.model.config.n_kv_heads}
- Feed-forward dim: {self.model.config.d_ff}
- Vocabulary size: {self.model.config.vocab_size}
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}
- RoPE theta: {self.model.config.rope_theta}
- Tie embeddings: {self.model.config.tie_embeddings}
- SWA window: {self.model.config.swa_window}
        """.strip()
