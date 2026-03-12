from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".docx"}
SIDECAR_SUFFIXES = {".meta.yaml", ".meta.yml", ".meta.json"}
PARSER_VERSION = "2026.03.12"


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def estimate_tokens(text: str) -> int:
    if not text.strip():
        return 0
    words = re.findall(r"\w+|\S", text)
    return max(1, math.ceil(len(words) * 0.75))


def discover_documents(root: Path) -> list[Path]:
    discovered: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if any(str(path).endswith(suffix) for suffix in SIDECAR_SUFFIXES):
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            discovered.append(path)
    return sorted(discovered)


def sidecar_candidates(path: Path) -> list[Path]:
    return [
        Path(f"{path}.meta.yaml"),
        Path(f"{path}.meta.yml"),
        Path(f"{path}.meta.json"),
    ]


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or "untitled"


def chunked(iterable: Iterable[str], size: int) -> list[list[str]]:
    batch: list[str] = []
    batches: list[list[str]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches
