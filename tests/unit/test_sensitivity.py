from __future__ import annotations

from local_agentic_rag.models import ParsedSection
from local_agentic_rag.sensitivity import should_auto_restrict_document


def test_auto_restrict_detects_confidential_marker() -> None:
    assert should_auto_restrict_document(
        title="Comp Plan",
        sections=[ParsedSection(text="Confidential. Bonus adjustments are planned for July.", section_path="Comp Plan")],
        markers=["confidential", "internal only"],
    )


def test_auto_restrict_ignores_plain_public_copy() -> None:
    assert not should_auto_restrict_document(
        title="Public FAQ",
        sections=[ParsedSection(text="This handbook explains support hours and pricing.", section_path="Public FAQ")],
        markers=["confidential", "internal only"],
    )
