import sys
from pathlib import Path


# Allow tests to import root-level modules (model.py, data.py, etc.) when running
# `pytest` without installing the project as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

