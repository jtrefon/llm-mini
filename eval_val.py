#!/usr/bin/env python3

import argparse
import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import yaml

from data import make_dataloaders
from infer import load_from_lightning_ckpt


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected config at {path} to be a mapping")
    return cfg


@torch.no_grad()
def evaluate_val_loss(
    model: torch.nn.Module,
    val_loader: Any,
    device: torch.device,
    max_batches: Optional[int],
) -> tuple[float, float]:
    model.eval()

    total_nll = 0.0
    total_tokens = 0

    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)

        vocab = logits.size(-1)
        loss_sum = F.cross_entropy(
            logits.view(-1, vocab),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        token_count = int((labels != -100).sum().item())

        total_nll += float(loss_sum.item())
        total_tokens += token_count

    if total_tokens == 0:
        raise RuntimeError("No non-ignored validation tokens were evaluated.")

    avg_nll = total_nll / float(total_tokens)
    ppl = math.exp(avg_nll)
    return avg_nll, ppl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute true val_loss/val_ppl for a Lightning checkpoint using config.yaml validation dataloader."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to Lightning .ckpt")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml used to build the validation dataloader",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Preferred device (falls back gracefully if unavailable)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="auto",
        help="Tokenizer name/path, or 'auto' to infer from checkpoint (recommended)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on number of validation batches",
    )

    args = parser.parse_args()

    cfg = _load_config(args.config)

    model, tokenizer, device = load_from_lightning_ckpt(
        args.ckpt,
        tokenizer_name=args.tokenizer,
        device=args.device,
    )

    _, val_loader = make_dataloaders(cfg, tokenizer)

    avg_nll, ppl = evaluate_val_loss(
        model=model,
        val_loader=val_loader,
        device=device,
        max_batches=args.max_batches,
    )

    print(f"val_loss={avg_nll:.6f}")
    print(f"val_ppl={ppl:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
