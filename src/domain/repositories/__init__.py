"""Repository interfaces for data access.

This package defines abstract interfaces for data access operations,
following the Repository pattern to decouple business logic from
data storage implementations.
"""

from .checkpoint_repository import CheckpointRepository, CheckpointInfo

__all__ = [
    'CheckpointRepository',
    'CheckpointInfo'
]
