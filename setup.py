"""Minimal setup for the educational, script-first repo.

This project is intentionally small and keeps most code at the repository root
(e.g. model.py, train.py). We expose those as `py_modules` so `pip install .`
works for users who prefer it, without pretending we have a full `src/` package.
"""

from setuptools import setup


setup(
    name="tiny-transformer-starter",
    version="0.1.0",
    description="Educational, from-scratch decoder-only Transformer (training + inference + SFT).",
    python_requires=">=3.11",
    py_modules=[
        "data",
        "eval_val",
        "evaluate_checkpoints",
        "finetune",
        "infer",
        "list",
        "model",
        "train",
    ],
)
