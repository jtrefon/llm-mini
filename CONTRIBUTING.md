# Contributing

Thanks for helping improve this educational project.

## What makes a good contribution here

- Clarity improvements (docs, comments, small refactors that improve readability)
- Fixes for correctness bugs (masking, shapes, device handling, etc.)
- Small, well-scoped features that support learning (e.g. better CLI flags, better eval cases)

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
pytest
```

## PR guidelines

- Keep PRs small and focused (one idea per PR).
- Update docs/configs when behavior changes.
- If you add a feature, add at least a small test when practical.

