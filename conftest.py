"""Make ``import otfs`` work from a plain checkout, with or without installation."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "python"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
