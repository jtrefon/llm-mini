"""Pytest configuration to ensure the package under src/ is importable.
This keeps test imports like `from src...` working without extra plugins.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root (which contains the `src/` directory) to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import shared fixtures so they're available to all tests
# This allows using fixtures from `tests/fixtures/conftest.py` project-wide
try:
    from tests.fixtures.conftest import *  # noqa: F401,F403
except Exception:
    # Keep tests importable even if fixtures module changes
    pass
