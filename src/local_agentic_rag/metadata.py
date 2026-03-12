from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .utils import sidecar_candidates


def load_sidecar_metadata(path: Path) -> dict[str, Any]:
    for candidate in sidecar_candidates(path):
        if not candidate.exists():
            continue
        if candidate.suffix == ".json":
            return json.loads(candidate.read_text(encoding="utf-8"))
        return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    return {}


def normalize_access_principals(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError("access_principals must be a string or a list of strings.")
