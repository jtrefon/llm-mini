"""Domain entities - Core business objects.

This package contains the fundamental business entities that represent
the core concepts in the transformer training domain:

- ModelConfiguration: Model architecture hyperparameters
- TrainingConfiguration: Training process hyperparameters
"""

from .model_config import ModelConfiguration
from .training_config import TrainingConfiguration

__all__ = [
    'ModelConfiguration',
    'TrainingConfiguration'
]
