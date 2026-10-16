"""Shared test setup.

The repository has two importable roots - ``vision`` for the pipeline and agent,
and the repository root for the ``service`` package - and neither is installed.
Both are put on the path here once, so individual test modules only need to
import what they are testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for _candidate in (ROOT / "vision", ROOT):
    _s = str(_candidate)
    if _s not in sys.path:
        sys.path.insert(0, _s)
