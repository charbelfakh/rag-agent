"""Put the repo root on ``sys.path`` so ``providers.*`` imports work.

``python -m scripts.…`` from the repo root already resolves imports; this module
exists for direct runs like ``python scripts/ingest/ingest.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
