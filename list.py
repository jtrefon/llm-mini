#!/usr/bin/env python3
"""
list.py — List model checkpoints and their metadata (epoch, steps, losses)

Usage:
  python list.py [--dir DIR] [--json] [--limit N] [--sort mtime|step|val]

By default, it searches in:
  - checkpoints/*.ckpt
  - lightning_logs/version_*/checkpoints/*.ckpt

It extracts metadata from:
  - Checkpoint contents (epoch, global_step, and, if present, monitor metrics)
  - Filename patterns (epoch, step, val_loss) as a fallback

It is compatible with checkpoints produced by both train.py and finetune.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch


SEARCH_PATTERNS = [
    "checkpoints/*.ckpt",
    "lightning_logs/version_*/checkpoints/*.ckpt",
]


@dataclass
class CkptInfo:
    path: str
    name: str
    mtime: float
    epoch: Optional[int] = None
    step: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    monitor: Optional[str] = None
    best_model_score: Optional[float] = None
    is_best: bool = False
    is_last: bool = False

    def to_view(self) -> Dict[str, Any]:
        return {
            "file": self.name,
            "epoch": self.epoch if self.epoch is not None else "-",
            "step": self.step if self.step is not None else "-",
            "train_loss": f"{self.train_loss:.4f}" if isinstance(self.train_loss, (int, float)) else "-",
            "val_loss": f"{self.val_loss:.4f}" if isinstance(self.val_loss, (int, float)) else "-",
            "monitor": self.monitor or "-",
            "best_model_score": f"{self.best_model_score:.4f}" if isinstance(self.best_model_score, (int, float)) else "-",
            "mtime": datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "tags": ",".join([t for t in (["best"] if self.is_best else []) + (["last"] if self.is_last else [])]) or "-",
        }


FILENAME_PATTERNS = [
    re.compile(r"epoch=(?P<epoch>\d+).*step=(?P<step>\d+).*val_loss=(?P<val>[0-9]+(?:\.[0-9]+)?)"),
    re.compile(r"epoch=(?P<epoch>\d+).*step=(?P<step>\d+)"),
    re.compile(r"global_step=(?P<step>\d+)"),
    re.compile(r"(?P<epoch>\d+)-(?P<step>\d+)\.ckpt$"),
]


def parse_from_filename(name: str) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    epoch = step = None
    val = None
    for pat in FILENAME_PATTERNS:
        m = pat.search(name)
        if m:
            if "epoch" in m.groupdict():
                try:
                    epoch = int(m.group("epoch"))
                except Exception:
                    pass
            if "step" in m.groupdict():
                try:
                    step = int(m.group("step"))
                except Exception:
                    pass
            if "val" in m.groupdict():
                try:
                    val = float(m.group("val"))
                except Exception:
                    pass
            break
    return epoch, step, val


def discover_checkpoints(extra_dir: Optional[str] = None) -> List[str]:
    pats = list(SEARCH_PATTERNS)
    if extra_dir:
        pats.insert(0, os.path.join(extra_dir, "*.ckpt"))
    seen = set()
    results: List[str] = []
    for pat in pats:
        for p in glob.glob(pat):
            if p not in seen:
                seen.add(p)
                results.append(p)
    results.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return results


def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def extract_from_callbacks(cb_state: Dict[str, Any]) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[float]]:
    """Attempt to extract monitor name, best score, val_loss, train_loss_epoch from callback state.

    Works across several PL versions best-effort.
    """
    monitor = None
    best_score = None
    val_loss = None
    train_loss = None

    try:
        # cb_state is typically a dict keyed by callback IDs -> state dicts
        for _cb_id, sd in (cb_state.items() if isinstance(cb_state, dict) else []):
            if not isinstance(sd, dict):
                continue
            if monitor is None and isinstance(sd.get("monitor", None), str):
                monitor = sd.get("monitor")
            # best score tracked by ModelCheckpoint
            if best_score is None and sd.get("best_model_score", None) is not None:
                try:
                    best_score = float(sd.get("best_model_score"))
                except Exception:
                    pass
            # monitor_candidates can hold metrics at save time
            mc = sd.get("monitor_candidates", None)
            if isinstance(mc, dict):
                if val_loss is None and mc.get("val_loss", None) is not None:
                    val_loss = safe_float(mc.get("val_loss"))
                if train_loss is None and mc.get("train_loss_epoch", None) is not None:
                    train_loss = safe_float(mc.get("train_loss_epoch"))
                if train_loss is None and mc.get("train_loss", None) is not None:
                    train_loss = safe_float(mc.get("train_loss"))
    except Exception:
        pass

    return monitor, best_score, val_loss, train_loss


def read_checkpoint_meta(path: str) -> CkptInfo:
    name = os.path.basename(path)
    mtime = os.path.getmtime(path)
    ep_f, st_f, vl_f = parse_from_filename(name)

    info = CkptInfo(path=path, name=name, mtime=mtime, epoch=ep_f, step=st_f, val_loss=vl_f)
    info.is_best = name in ("best.ckpt", "sft-best.ckpt") or name.startswith("sft-best")
    info.is_last = name == "last.ckpt"

    try:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)  # PyTorch 2.2+
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        # Standard keys
        if info.epoch is None and isinstance(ckpt.get("epoch", None), int):
            info.epoch = int(ckpt["epoch"]) 
        if info.step is None and isinstance(ckpt.get("global_step", None), int):
            info.step = int(ckpt["global_step"]) 

        # Callback state (for monitor metrics)
        cb = ckpt.get("callbacks", None)
        if isinstance(cb, dict):
            mon, best_score, val_loss, train_loss = extract_from_callbacks(cb)
            info.monitor = info.monitor or mon
            if info.val_loss is None and val_loss is not None:
                info.val_loss = val_loss
            if info.train_loss is None and train_loss is not None:
                info.train_loss = train_loss
            if info.best_model_score is None and best_score is not None:
                info.best_model_score = best_score
        # Some PL versions store logged metrics directly
        monitor_candidates = ckpt.get("monitor_candidates", None)
        if isinstance(monitor_candidates, dict):
            if info.val_loss is None and monitor_candidates.get("val_loss") is not None:
                info.val_loss = safe_float(monitor_candidates.get("val_loss"))
            if info.train_loss is None and monitor_candidates.get("train_loss_epoch") is not None:
                info.train_loss = safe_float(monitor_candidates.get("train_loss_epoch"))
    except Exception as e:
        # Failed to read checkpoint details; keep filename-derived info
        pass

    return info


def format_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No checkpoints found."
    headers = ["file", "epoch", "step", "train_loss", "val_loss", "monitor", "best_model_score", "mtime", "tags"]
    # compute column widths
    colw = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    def line(obj: Dict[str, Any]) -> str:
        return "  ".join(str(obj.get(h, "")).ljust(colw[h]) for h in headers)
    out = [line({h: h for h in headers})]
    out.append("  ".join("-" * colw[h] for h in headers))
    for r in rows:
        out.append(line(r))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="List model checkpoints and metadata")
    ap.add_argument("--dir", default=None, help="Directory to search for .ckpt files (in addition to defaults)")
    ap.add_argument("--json", action="store_true", help="Output JSON instead of table")
    ap.add_argument("--limit", type=int, default=None, help="Limit the number of entries shown")
    ap.add_argument("--sort", choices=["mtime", "step", "val"], default="mtime", help="Sort by criterion")
    args = ap.parse_args()

    ckpts = discover_checkpoints(args.dir)
    infos = [read_checkpoint_meta(p) for p in ckpts]

    # sorting
    if args.sort == "mtime":
        infos.sort(key=lambda i: i.mtime, reverse=True)
    elif args.sort == "step":
        infos.sort(key=lambda i: (i.step or -1, i.mtime), reverse=True)
    elif args.sort == "val":
        # lower val is better; unknowns go to bottom
        infos.sort(key=lambda i: (float("inf") if i.val_loss is None else i.val_loss, -i.mtime))

    if args.limit is not None:
        infos = infos[: args.limit]

    if args.json:
        print(json.dumps([asdict(i) for i in infos], indent=2))
    else:
        rows = [i.to_view() for i in infos]
        print(format_table(rows))


if __name__ == "__main__":
    main()
