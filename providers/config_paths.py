"""Central project paths for config and runtime artifacts."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

VENDORS_JSON = CONFIG_DIR / "vendors.json"
INGEST_MANIFEST_PATH = Path(
    os.getenv("INGEST_MANIFEST_PATH", str(DATA_DIR / "ingest_manifest.json"))
)
