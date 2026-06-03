"""Common utility functions used by multiple layers."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

WHITESPACE_PATTERN = re.compile(r"\s+")


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""

    return datetime.now(UTC)


def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""

    return utc_now().isoformat()


def generate_id(prefix: str | None = None) -> str:
    """Generate a unique identifier with an optional prefix."""

    value = uuid.uuid4().hex
    if prefix is None:
        return value
    normalized_prefix = prefix.strip().lower().replace(" ", "_")
    return f"{normalized_prefix}_{value}"


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim leading or trailing spaces."""

    return WHITESPACE_PATTERN.sub(" ", text).strip()


def sha256_text(text: str) -> str:
    """Create a deterministic SHA-256 digest for text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return the resolved path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, default: Any | None = None) -> Any:
    """Read JSON from disk, returning a default value when the file is absent."""

    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    """Write JSON to disk using a stable, human-readable format."""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")

