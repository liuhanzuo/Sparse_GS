"""Make ``scripts`` importable as a package so ``python -m scripts.train`` works."""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as `python -m scripts.*`
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Windows bootstrap: activate MSVC, ninja, patch headers
os.environ.setdefault("SPARSE_GS_SKIP_BOOTSTRAP", "0")

from ._bootstrap import PROJECT_ROOT  # noqa: F401,E402

__all__ = ["PROJECT_ROOT"]
