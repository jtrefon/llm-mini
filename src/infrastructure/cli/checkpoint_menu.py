"""Console-based checkpoint resume selection menu.

Encapsulates interactive and non-interactive checkpoint selection logic so that
entrypoints remain thin and adhere to clean architecture.
"""
from __future__ import annotations

import sys
from typing import Optional

from src.domain.repositories.checkpoint_repository import CheckpointRepository, CheckpointInfo


def _best_by_val(ckpts: list[CheckpointInfo]) -> Optional[CheckpointInfo]:
    vals = [c for c in ckpts if c.val_loss is not None]
    return min(vals, key=lambda x: x.val_loss) if vals else None


def _best_by_step(ckpts: list[CheckpointInfo]) -> Optional[CheckpointInfo]:
    steps = [c for c in ckpts if c.step is not None]
    return max(steps, key=lambda x: x.step) if steps else None


def _newest(ckpts: list[CheckpointInfo]) -> Optional[CheckpointInfo]:
    return ckpts[0] if ckpts else None  # list_checkpoints already newest first


def select_resume_checkpoint(repo: CheckpointRepository) -> Optional[str]:
    """Return a checkpoint path string to resume from, or None to start fresh.

    In interactive terminals, show a menu to select. In non-interactive contexts,
    auto-select best by val_loss, then highest step, then newest.
    """
    ckpts = repo.list_checkpoints()
    if not ckpts:
        return None

    try:
        if sys.stdin.isatty():
            print("\n🔍 Found the following checkpoints:\n")
            for idx, c in enumerate(ckpts, 1):
                ep = c.epoch if c.epoch is not None else '-'
                st = c.step if c.step is not None else '-'
                vl = f"{c.val_loss:.3f}" if c.val_loss is not None else '-'
                print(f"[{idx}] {c.name}\t(epoch={ep}, step={st}, val_loss={vl})")

            print("\nEnter a number to RESUME from that checkpoint.")
            print("Or press: 'b' = lowest val_loss, 's' = highest step, 'n' = start fresh, Enter = default (b>s>newest)")

            while True:
                sel = input("Selection: ").strip().lower()
                if sel == '':
                    choice = _best_by_val(ckpts) or _best_by_step(ckpts) or _newest(ckpts)
                    if choice:
                        print(f"Default selection: {choice.name}")
                        return str(choice.path)
                    return None
                if sel in ('n', 'no'):
                    return None
                if sel == 'b':
                    ch = _best_by_val(ckpts)
                    if ch:
                        print(f"Best val_loss: {ch.name}")
                        return str(ch.path)
                    print("No checkpoints with val_loss found.")
                    continue
                if sel == 's':
                    ch = _best_by_step(ckpts)
                    if ch:
                        print(f"Highest step: {ch.name}")
                        return str(ch.path)
                    print("No checkpoints with step found.")
                    continue
                if sel.isdigit():
                    i = int(sel)
                    if 1 <= i <= len(ckpts):
                        ch = ckpts[i-1]
                        print(f"Selected: {ch.name}")
                        return str(ch.path)
                print("Invalid selection. Try again.")
        else:
            best = _best_by_val(ckpts) or _best_by_step(ckpts) or _newest(ckpts)
            if best:
                print(f"Non-interactive resume from: {best.name}")
                return str(best.path)
            return None
    except Exception as e:
        print(f"! Checkpoint selection failed ({e}). Proceeding with fresh training.")
        return None
