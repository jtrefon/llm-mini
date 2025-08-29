"""Infrastructure models - Framework-specific model adapters.

This package contains adapters and factories that bridge domain models
with specific ML frameworks (PyTorch, PyTorch Lightning, etc.).
"""

from .pytorch_model_factory import PyTorchModelFactory
from .pytorch_lightning_adapter import PyTorchLightningGPTModel

__all__ = [
    'PyTorchModelFactory',
    'PyTorchLightningGPTModel'
]
