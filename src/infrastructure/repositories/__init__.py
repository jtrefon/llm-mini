"""Infrastructure layer repository implementations.

This package contains concrete implementations of repository interfaces
using specific storage technologies and frameworks.
"""

from .file_checkpoint_repository import FileCheckpointRepository

__all__ = [
    'FileCheckpointRepository'
]
