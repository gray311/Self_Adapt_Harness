"""Small, strict readers for SAH's native Cordis patch documents."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_patch(path: Path) -> list[dict[str, Any]]:
    """Load a Cordis patch sequence and reject non-mapping operations."""
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or []
    if not isinstance(payload, list) or any(
        not isinstance(row, dict) for row in payload
    ):
        raise ValueError(f"{source}: Cordis patch must be a sequence of mappings")
    return payload


def row_config(path: Path, row_id: str) -> dict[str, Any]:
    """Return the unique enabled row config for ``row_id``."""
    matched = [row for row in load_patch(path) if row.get("id") == row_id]
    if len(matched) != 1:
        raise ValueError(f"{path}: expected exactly one Cordis row {row_id!r}")
    row = matched[0]
    if row.get("disabled") is True or not isinstance(row.get("config"), dict):
        raise ValueError(f"{path}: Cordis row {row_id!r} is disabled or malformed")
    return dict(row["config"])


def system_persona(path: Path) -> str:
    """Read the non-empty `system-prompt.config.persona` string."""
    persona = row_config(path, "system-prompt").get("persona")
    if not isinstance(persona, str) or not persona.strip():
        raise ValueError(f"{path}: system-prompt persona must be non-empty")
    return persona.strip()
