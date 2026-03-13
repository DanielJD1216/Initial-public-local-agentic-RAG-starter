from __future__ import annotations

import re

from .models import ParsedSection


def should_auto_restrict_document(
    *,
    title: str,
    sections: list[ParsedSection],
    markers: list[str],
) -> bool:
    haystack = _normalized_haystack(title=title, sections=sections)
    for marker in markers:
        normalized_marker = _normalize_text(marker)
        if not normalized_marker:
            continue
        if normalized_marker in haystack:
            return True
    return False


def _normalized_haystack(*, title: str, sections: list[ParsedSection]) -> str:
    parts = [title]
    for section in sections:
        parts.append(section.section_path)
        parts.append(section.text)
    return _normalize_text(" ".join(parts))


def _normalize_text(value: str) -> str:
    lowered = value.lower()
    collapsed = re.sub(r"\s+", " ", lowered)
    return collapsed.strip()
