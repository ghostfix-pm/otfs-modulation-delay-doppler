"""Thin wrapper keeping the documented command working.

    python python/otfs_sim.py [--frames 400] [--seed 3]

The implementation lives in the :mod:`otfs` package next to this file; see
``python -m otfs --help`` for the full interface (sub-commands, ``--grid``,
``--json``/``--csv`` output and an explicit figure directory).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from otfs.cli import main

if __name__ == "__main__":
    sys.exit(main())
