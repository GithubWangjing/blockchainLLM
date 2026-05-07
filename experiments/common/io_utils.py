"""File-system helpers shared across experiments."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_slug() -> str:
    """Return a filesystem-friendly timestamp string."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

