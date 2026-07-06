"""Per-user app-data locations for secret files.

Credentials (OAuth tokens, saved LLM API keys) live outside the repo tree in
the OS per-user application-data folder — %APPDATA%\\rag-agent on Windows,
~/.rag-agent elsewhere (ForgeStation pattern) — so they can never be swept up
by a repo backup, `git add -f`, or a shared checkout. Files that previously
lived under the repo's data/ folder are migrated on first access.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

APP_DIR_NAME = "rag-agent"


def app_data_dir() -> Path:
    """Per-user app-data directory (created on demand, owner-only on POSIX)."""
    configured = os.getenv("RAG_AGENT_APP_DIR", "").strip()
    if configured:
        base = Path(configured)
    else:
        appdata = os.getenv("APPDATA", "").strip()
        if appdata:
            base = Path(appdata) / APP_DIR_NAME
        else:
            base = Path.home() / f".{APP_DIR_NAME}"
    base.mkdir(parents=True, exist_ok=True)
    try:  # best-effort owner-only perms on POSIX (no-op on Windows)
        os.chmod(base, 0o700)
    except OSError:
        pass
    return base


def secret_file(name: str, legacy_path: Path | None = None) -> Path:
    """Resolve a secret file in the app-data dir, migrating a legacy copy.

    If the app-data file does not exist yet but `legacy_path` (the old
    location under the repo's data/ folder) does, the file is moved so
    existing credentials survive the relocation.
    """
    target = app_data_dir() / name
    if not target.exists() and legacy_path is not None and legacy_path.exists():
        try:
            shutil.move(str(legacy_path), str(target))
            logger.info("Migrated %s -> %s", legacy_path, target)
        except OSError:
            logger.warning("Could not migrate %s to %s; using legacy path", legacy_path, target)
            return legacy_path
    return target
