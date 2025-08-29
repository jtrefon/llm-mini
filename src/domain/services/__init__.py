"""Domain services - Application business logic interfaces.

This package defines abstract interfaces for core business operations,
following the Service Layer pattern to encapsulate complex business logic
and orchestrate multiple repository operations.
"""

from .training_service import TrainingService, TrainingResult

__all__ = [
    'TrainingService',
    'TrainingResult'
]
