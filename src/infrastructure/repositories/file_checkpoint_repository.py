"""File system checkpoint repository implementation."""
import re
import glob
from pathlib import Path
from typing import List, Optional, Union

from src.domain.repositories.checkpoint_repository import CheckpointRepository, CheckpointInfo


class FileCheckpointRepository(CheckpointRepository):
    """File system implementation of checkpoint repository.

    Handles checkpoint discovery, parsing, and management on the local file system.
    Supports both current checkpoint directory and legacy PyTorch Lightning structure.
    """

    def __init__(self, checkpoints_dir: Union[str, Path], legacy_logs_dir: Optional[Union[str, Path]] = None):
        """Initialize the file system checkpoint repository.

        Args:
            checkpoints_dir: Primary directory for checkpoints
            legacy_logs_dir: Optional legacy PyTorch Lightning logs directory
        """
        self.checkpoints_dir = Path(checkpoints_dir)
        self.legacy_logs_dir = Path(legacy_logs_dir) if legacy_logs_dir else None

    def find_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Find checkpoint with highest global step count."""
        checkpoints = self._discover_all_checkpoints()
        if not checkpoints:
            return None

        # Sort by step desc, then mtime desc
        checkpoints.sort(key=lambda x: (x.step or -1, x.mtime), reverse=True)
        return checkpoints[0]

    def find_best_checkpoint(self) -> Optional[CheckpointInfo]:
        """Find checkpoint with lowest validation loss."""
        checkpoints = self._discover_all_checkpoints()
        valid_checkpoints = [c for c in checkpoints if c.val_loss is not None]

        if not valid_checkpoints:
            return None

        return min(valid_checkpoints, key=lambda x: x.val_loss)

    def list_checkpoints(self) -> List[CheckpointInfo]:
        """List all checkpoints sorted by modification time (newest first)."""
        checkpoints = self._discover_all_checkpoints()
        checkpoints.sort(key=lambda x: x.mtime, reverse=True)
        return checkpoints

    def parse_checkpoint_metadata(self, path: Path) -> CheckpointInfo:
        """Parse epoch, step, val_loss from checkpoint filename.

        Supports multiple filename patterns:
        - epoch=X-step=Y-val_loss=Z.ckpt
        - global_step=Y.ckpt
        - E-S.ckpt (legacy format)
        """
        name = path.name
        epoch = step = val_loss = None

        # Primary pattern: epoch=X-step=Y-val_loss=Z.ckpt
        match = re.search(r"epoch=(\d+).*step=(\d+).*val_loss=([0-9]+(?:\.[0-9]+)?)", name)
        if match:
            epoch, step, val_loss = int(match.group(1)), int(match.group(2)), float(match.group(3))
        else:
            # Secondary pattern: global_step=Y
            match = re.search(r"global_step=(\d+)", name)
            if match:
                step = int(match.group(1))
            else:
                # Legacy pattern: E-S.ckpt
                match = re.match(r"(\d+)-(\d+)\.ckpt$", name)
                if match:
                    epoch, step = int(match.group(1)), int(match.group(2))

        return CheckpointInfo(path, epoch, step, val_loss)

    def checkpoint_exists(self, path: Path) -> bool:
        """Check if checkpoint exists at path."""
        return path.exists() and path.is_file()

    def _discover_all_checkpoints(self) -> List[CheckpointInfo]:
        """Discover all checkpoint files from configured directories."""
        patterns = [str(self.checkpoints_dir / "*.ckpt")]

        if self.legacy_logs_dir:
            patterns.append(str(self.legacy_logs_dir / "version_*" / "checkpoints" / "*.ckpt"))

        checkpoints = []
        seen_paths = set()

        for pattern in patterns:
            for path_str in glob.glob(pattern):
                path = Path(path_str)
                if path not in seen_paths and self.checkpoint_exists(path):
                    seen_paths.add(path)
                    checkpoints.append(self.parse_checkpoint_metadata(path))

        return checkpoints
