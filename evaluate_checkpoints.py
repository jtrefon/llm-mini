#!/usr/bin/env python3
"""
Utility for quick sanity-checking Lightning checkpoints on simple next-token tests.

Given a set of prompts (optionally with expected continuations), the script loads
each checkpoint, calculates the probability assigned to the supplied targets, and
prints the top-k next-token predictions. This is meant as a lightweight ranking
tool for choosing a baseline checkpoint before fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from glob import glob
import re
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn.functional as F

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is optional but recommended
    yaml = None

from infer import load_from_lightning_ckpt


DEFAULT_CASES = [
    {
        "name": "France capital",
        "prompt": "The capital of France is",
        "target": " Paris",
    },
    {
        "name": "Germany capital",
        "prompt": "The capital of Germany is",
        "target": " Berlin",
    },
    {
        "name": "Simple arithmetic",
        "prompt": "2 + 2 =",
        "target": " 4",
    },
    {
        "name": "Antonym",
        "prompt": "The opposite of hot is",
        "target": " cold",
    },
    {
        "name": "Common fact",
        "prompt": "Water freezes at",
        "target": " 0",
    },
]


@dataclass
class TestCase:
    prompt: str
    target: Optional[str] = None
    name: Optional[str] = None


@dataclass
class CaseResult:
    case: TestCase
    top_predictions: List[tuple[str, float]]
    target_prob: Optional[float]
    target_rank: Optional[int]
    target_nll: Optional[float]
    token_count: int
    top1_hit: Optional[bool]


@dataclass
class CheckpointSummary:
    path: Path
    evaluated_tokens: int
    total_nll: float
    top1_hits: int
    cases_with_targets: int

    @property
    def avg_nll(self) -> Optional[float]:
        if self.evaluated_tokens == 0:
            return None
        return self.total_nll / self.evaluated_tokens

    @property
    def ppl(self) -> Optional[float]:
        avg = self.avg_nll
        if avg is None:
            return None
        return math.exp(avg)

    @property
    def accuracy(self) -> Optional[float]:
        if self.cases_with_targets == 0:
            return None
        return self.top1_hits / self.cases_with_targets


def _format_token(token: str) -> str:
    """Represent tokens including leading spaces/newlines visibly."""
    safe = token.replace("\n", "\\n")
    return repr(safe)


def _expand_case_files(spec: str) -> List[Path]:
    """Expand a case spec into concrete files.

    Supports:
    - glob patterns (e.g. "cases/*.yaml")
    - directories (e.g. "cases/")
    - comma-separated lists (e.g. "cases/a.yaml,cases/b.yaml")
    """

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    paths: List[Path] = []
    for part in parts:
        part_path = Path(part)
        if part_path.is_dir():
            for ext in ("*.yaml", "*.yml", "*.json"):
                paths.extend(Path(p) for p in glob(str(part_path / ext)))
            continue

        matches = glob(part)
        if matches:
            paths.extend(Path(p) for p in matches)
            continue

        if part_path.exists():
            paths.append(part_path)

    deduped = sorted(set(paths))
    return [p for p in deduped if p.is_file()]


def _load_cases_file(file_path: Path) -> List[TestCase]:
    data = None
    if file_path.suffix.lower() in {".json"}:
        data = json.loads(file_path.read_text())
    elif file_path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("pyyaml is required to load YAML cases")
        data = yaml.safe_load(file_path.read_text())
    else:
        raise ValueError(f"Cases file must be .json or .yaml/.yml, got: {file_path}")

    if not isinstance(data, Iterable):
        raise ValueError(f"Cases file must contain a list of prompts: {file_path}")

    cases: List[TestCase] = []
    for raw in data:
        if not isinstance(raw, dict) or "prompt" not in raw:
            raise ValueError(f"Each case must be a mapping with at least a 'prompt' field: {file_path}")
        cases.append(TestCase(
            prompt=str(raw["prompt"]),
            target=str(raw.get("target")) if raw.get("target") is not None else None,
            name=str(raw.get("name")) if raw.get("name") is not None else None,
        ))
    return cases


def _maybe_load_cases(path: Optional[str]) -> List[TestCase]:
    if not path:
        return [TestCase(**c) for c in DEFAULT_CASES]

    files = _expand_case_files(path)
    if not files:
        print(f"[warn] no cases matched spec {path!r}, falling back to built-in defaults.")
        return [TestCase(**c) for c in DEFAULT_CASES]

    cases: List[TestCase] = []
    for file_path in files:
        cases.extend(_load_cases_file(file_path))
    return cases


def _expand_checkpoints(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        matches = glob(pat)
        if not matches and Path(pat).exists():
            matches = [pat]
        for m in matches:
            paths.append(Path(m))
    deduped = sorted(set(paths))
    return [p for p in deduped if p.is_file()]


_STEP_RE = re.compile(r"(?:global_)?step[=\-_]?(\d+)", re.IGNORECASE)


def _checkpoint_sort_key(path: Path) -> tuple:
    """
    Sort checkpoints by extracted step/epoch number, falling back to name.
    Ensures sequential ordering across `global_step=XXXX.ckpt` style paths.
    """
    name = path.name
    match = _STEP_RE.search(name)
    step = int(match.group(1)) if match else None
    # Move "last.ckpt" to the end by treating step as infinity.
    if name.startswith("last"):
        step = float("inf")
    return (step if step is not None else float("inf"), name)


def _tokenize_text(tokenizer, text: str) -> torch.Tensor:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded.get("input_ids")
    if input_ids is None or input_ids.numel() == 0:
        raise ValueError(f"Tokenization produced no tokens for text: {text!r}")
    return input_ids


def evaluate_case(model, tokenizer, device, case: TestCase, top_k: int) -> CaseResult:
    prompt_ids = _tokenize_text(tokenizer, case.prompt)
    target_ids = None
    if case.target:
        target_ids = _tokenize_text(tokenizer, case.target)

    if target_ids is not None:
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    else:
        input_ids = prompt_ids

    input_ids = input_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids)

    prompt_len = prompt_ids.size(1)
    first_logits = logits[:, prompt_len - 1, :]
    probs = F.softmax(first_logits, dim=-1)
    top_probs, top_idx = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)

    top_predictions: List[tuple[str, float]] = []
    for prob, idx in zip(top_probs[0], top_idx[0]):
        token = tokenizer.decode([idx.item()])
        top_predictions.append((token, prob.item()))

    token_count = 0
    target_prob = None
    target_rank = None
    target_nll = None
    top1_hit = None

    if target_ids is not None:
        target_ids = target_ids.to(device)
        target_len = target_ids.size(1)
        token_count = target_len

        relevant_logits = logits[:, prompt_len - 1: prompt_len - 1 + target_len, :]
        log_probs = F.log_softmax(relevant_logits, dim=-1)
        arange_idx = torch.arange(target_len, device=device)
        selected = log_probs[0, arange_idx, target_ids[0]]
        target_nll = float(-selected.sum().item())

        first_target_id = int(target_ids[0, 0].item())
        first_prob = probs[0, first_target_id].item()
        target_prob = first_prob
        rank = int((probs[0] > probs[0, first_target_id]).sum().item() + 1)
        target_rank = rank
        top1_hit = bool(top_idx[0, 0].item() == first_target_id)

    return CaseResult(
        case=case,
        top_predictions=top_predictions,
        target_prob=target_prob,
        target_rank=target_rank,
        target_nll=target_nll,
        token_count=token_count,
        top1_hit=top1_hit,
    )


def evaluate_checkpoint(
    ckpt_path: Path,
    tokenizer_name: str,
    device: str,
    cases: List[TestCase],
    top_k: int,
) -> tuple[List[CaseResult], CheckpointSummary]:
    print(f"\n### Evaluating checkpoint: {ckpt_path}")
    model, tokenizer, model_device = load_from_lightning_ckpt(
        str(ckpt_path),
        tokenizer_name=tokenizer_name,
        device=device,
    )

    case_results: List[CaseResult] = []
    summary = CheckpointSummary(
        path=ckpt_path,
        evaluated_tokens=0,
        total_nll=0.0,
        top1_hits=0,
        cases_with_targets=0,
    )

    for case in cases:
        try:
            result = evaluate_case(model, tokenizer, model_device, case, top_k=top_k)
        except Exception as exc:
            print(f"  ! Failed on prompt {case.prompt!r}: {exc}")
            continue

        case_results.append(result)

        label = case.name or case.prompt
        print(f'  Prompt: "{label}"')
        print("    Top predictions:")
        for token, prob in result.top_predictions:
            print(f"      {_format_token(token)} | prob={prob:.4f}")

        if result.target_prob is not None:
            summary.evaluated_tokens += result.token_count
            summary.total_nll += result.target_nll or 0.0
            summary.cases_with_targets += 1
            if result.top1_hit:
                summary.top1_hits += 1
            print(f"    Target prob: {result.target_prob:.4f} (rank={result.target_rank})")
            if result.token_count > 0 and result.target_nll is not None:
                avg_nll = result.target_nll / result.token_count
                ppl = math.exp(avg_nll)
                print(f"    Target NLL: {result.target_nll:.4f} (avg={avg_nll:.4f}, ppl={ppl:.2f})")
        else:
            print("    (no target provided; skipping scoring)")

    # Cleanup to release memory between checkpoints
    del model
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    if summary.cases_with_targets > 0:
        print("  --- Summary ---")
        if summary.avg_nll is not None:
            print(f"    Avg NLL/token: {summary.avg_nll:.4f}")
            print(f"    Perplexity:    {summary.ppl:.2f}")
        if summary.accuracy is not None:
            print(f"    Top-1 accuracy: {summary.accuracy*100:.2f}% "
                  f"({summary.top1_hits}/{summary.cases_with_targets})")
    else:
        print("  No targets evaluated for this checkpoint.")

    return case_results, summary


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Lightning checkpoints on simple next-token tests.")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=["checkpoints/*.ckpt"],
        help="Checkpoint paths or glob patterns (default: checkpoints/*.ckpt)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name or path; defaults to the tokenizer in the config file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to training config for autodetecting tokenizer (default: config.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Preferred device (falls back gracefully if unavailable).",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="cases/*.yaml",
        help=(
            "Cases file spec: JSON/YAML file, glob pattern, directory, or comma-separated list "
            "(default: cases/*.yaml)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of next-token predictions to display per prompt.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Limit the number of checkpoints to evaluate (after sorting).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    # Determine tokenizer defaulting to config.yaml if available.
    tokenizer_name = args.tokenizer
    if tokenizer_name is None:
        cfg_path = Path(args.config)
        if cfg_path.exists() and yaml is not None:
            try:
                cfg = yaml.safe_load(cfg_path.read_text())
                tokenizer_name = cfg.get("data", {}).get("tokenizer_name")
                if tokenizer_name:
                    print(f"[info] Using tokenizer from {cfg_path}: {tokenizer_name}")
            except Exception as exc:
                print(f"[warn] Failed to read tokenizer from {cfg_path}: {exc}")
        if not tokenizer_name:
            tokenizer_name = "gpt2"
            print("[info] Falling back to tokenizer: gpt2")

    cases = _maybe_load_cases(args.cases)
    checkpoints = _expand_checkpoints(args.checkpoints)
    if not checkpoints:
        print("No checkpoints found for the supplied patterns.", file=sys.stderr)
        return 1

    checkpoints = sorted(checkpoints, key=_checkpoint_sort_key)
    if args.max_checkpoints is not None:
        checkpoints = checkpoints[: max(0, args.max_checkpoints)]

    summaries: List[CheckpointSummary] = []
    for ckpt in checkpoints:
        _, summary = evaluate_checkpoint(
            ckpt_path=ckpt,
            tokenizer_name=tokenizer_name,
            device=args.device,
            cases=cases,
            top_k=args.top_k,
        )
        summaries.append(summary)

    ranked = [
        s for s in summaries
        if s.avg_nll is not None
    ]
    ranked.sort(key=lambda s: s.avg_nll)

    if ranked:
        print("\n=== Ranking (by lowest avg NLL) ===")
        for idx, summary in enumerate(ranked, start=1):
            acc_str = "n/a"
            if summary.accuracy is not None:
                acc_str = f"{summary.accuracy*100:.2f}%"
            ppl_str = "n/a"
            if summary.ppl is not None:
                ppl_str = f"{summary.ppl:.2f}"
            print(f"{idx:>2}. {summary.path} | avg NLL={summary.avg_nll:.4f} | ppl={ppl_str} | top1={acc_str}")
    else:
        print("\nNo ranking available (no targets evaluated).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
