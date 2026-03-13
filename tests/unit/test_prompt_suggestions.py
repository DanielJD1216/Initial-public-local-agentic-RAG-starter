from __future__ import annotations

from local_agentic_rag.models import ChunkRecord
from local_agentic_rag.prompt_suggestions import build_prompt_suggestions, default_prompt_suggestions


def _chunk(*, title: str, section_path: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"{title}-{section_path}",
        doc_id=title,
        source_path=f"/tmp/{title}.md",
        content_type="text/markdown",
        checksum="checksum",
        parser_version="parser-v1",
        title=title,
        ingested_at="2026-03-13T00:00:00+00:00",
        access_scope="public",
        access_principals=["*"],
        chunk_index=0,
        section_path=section_path,
        text=text,
        location_label=section_path,
        token_count=32,
    )


def test_prompt_suggestions_cover_detected_topics() -> None:
    suggestions = build_prompt_suggestions(
        [
            _chunk(
                title="Acme Services Handbook",
                section_path="Acme Services Handbook > Support Coverage",
                text="Standard requests receive a first response within 4 business hours.",
            ),
            _chunk(
                title="Incident Playbook",
                section_path="Incident Playbook > Postmortems",
                text="Every customer-facing incident needs a postmortem within 2 business days.",
            ),
            _chunk(
                title="Restricted Internal Roadmap",
                section_path="Internal Roadmap Notes > Salary Adjustment Window",
                text="The salary adjustment review window is planned for the third week of June.",
            ),
        ]
    )

    prompts = [item.prompt for item in suggestions]
    assert any("standard support first response time" in prompt for prompt in prompts)
    assert any("postmortem due" in prompt for prompt in prompts)
    assert any("salary adjustment review window" in prompt for prompt in prompts)


def test_prompt_suggestions_fall_back_when_no_chunks_are_available() -> None:
    suggestions = build_prompt_suggestions([])

    assert [item.to_dict() for item in suggestions] == [item.to_dict() for item in default_prompt_suggestions()]
