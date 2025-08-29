"""PyTorch model factory - Infrastructure layer model creation."""
import torch
from typing import Any

from src.domain.entities.model_config import ModelConfiguration


class PyTorchModelFactory:
    """Factory for creating PyTorch models.

    This factory encapsulates model creation logic, providing
    a clean interface for the application layer while isolating
    PyTorch-specific implementation details.
    """

    def create_model(self, config: ModelConfiguration) -> Any:
        """Create a GPT model instance.

        Args:
            config: Model configuration

        Returns:
            PyTorch model instance
        """
        # Import here to avoid circular dependencies
        from src.domain.models.gpt_mini import GPTMini
        return GPTMini(config)

    def load_model_from_checkpoint(self, checkpoint_path: str, config: ModelConfiguration) -> Any:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Model configuration

        Returns:
            Loaded PyTorch model instance
        """
        model = self.create_model(config)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)

        return model
