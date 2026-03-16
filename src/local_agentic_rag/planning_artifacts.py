from __future__ import annotations

import hashlib
import json
import re

from .models import ChunkRecord, DocumentEntity, DocumentMetadata, DocumentPlanningArtifact, ParsedSection

PLANNING_ARTIFACT_VERSION = "planning-artifacts/v1"

MONTH_PATTERN = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b"
    r"(?:\s+\d{1,2})?(?:,\s*\d{4})?",
    re.I,
)
NUMERIC_DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b")
DOC_ID_PATTERN = re.compile(r"\b[A-Z]{2,}[A-Z0-9-]{1,}\d+[A-Z0-9-]*\b")
REVISION_PATTERN = re.compile(r"\brev(?:ision)?[\s:-]*[A-Z0-9.-]+\b", re.I)
CAPITALIZED_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b")

ORGANIZATION_MARKERS = {
    "company",
    "services",
    "team",
    "department",
    "committee",
    "group",
    "inc",
    "llc",
    "corp",
    "corporation",
}


def compute_planning_fingerprint(*, ingest_fingerprint: str, artifact_version: str = PLANNING_ARTIFACT_VERSION) -> str:
    payload = {
        "artifact_version": artifact_version,
        "ingest_fingerprint": ingest_fingerprint,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_planning_artifacts(
    *,
    metadata: DocumentMetadata,
    sections: list[ParsedSection],
    chunks: list[ChunkRecord],
    artifact_version: str = PLANNING_ARTIFACT_VERSION,
) -> tuple[DocumentPlanningArtifact, list[DocumentEntity]]:
    normalized_title = _normalize_text(metadata.title)
    section_outline = _section_outline(sections)
    short_summary = _build_summary(metadata.title, sections, chunks)
    planning_fingerprint = compute_planning_fingerprint(
        ingest_fingerprint=metadata.ingest_fingerprint,
        artifact_version=artifact_version,
    )
    artifact = DocumentPlanningArtifact(
        doc_id=metadata.doc_id,
        artifact_version=artifact_version,
        planning_fingerprint=planning_fingerprint,
        normalized_title=normalized_title,
        short_summary=short_summary,
        section_outline=section_outline,
    )
    entities = _extract_entities(metadata, sections, artifact)
    return artifact, entities


def _build_summary(title: str, sections: list[ParsedSection], chunks: list[ChunkRecord]) -> str:
    snippets: list[str] = []
    for section in sections[:3]:
        cleaned = _clean_whitespace(section.text)
        if cleaned:
            snippets.append(cleaned)
    if not snippets:
        snippets = [_clean_whitespace(chunk.text) for chunk in chunks[:3] if _clean_whitespace(chunk.text)]
    summary_source = " ".join(snippets)
    sentences = re.split(r"(?<=[.!?])\s+", summary_source)
    selected: list[str] = []
    total_length = 0
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        selected.append(stripped)
        total_length += len(stripped)
        if len(selected) >= 2 or total_length >= 320:
            break
    if not selected:
        fallback = _clean_whitespace(title)
        return fallback[:320]
    return " ".join(selected)[:320]


def _section_outline(sections: list[ParsedSection]) -> list[str]:
    outline: list[str] = []
    seen: set[str] = set()
    for section in sections:
        label = _clean_whitespace(section.section_path) or "Document"
        lowered = label.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        outline.append(label[:120])
        if len(outline) >= 12:
            break
    return outline or ["Document"]


def _extract_entities(
    metadata: DocumentMetadata,
    sections: list[ParsedSection],
    artifact: DocumentPlanningArtifact,
) -> list[DocumentEntity]:
    combined_parts = [metadata.title, artifact.short_summary, *artifact.section_outline]
    combined_parts.extend(section.text for section in sections[:4])
    combined_text = "\n".join(part for part in combined_parts if part).strip()
    entities: dict[tuple[str, str], DocumentEntity] = {}

    def register(entity_type: str, value: str) -> None:
        cleaned = _clean_whitespace(value)
        if not cleaned:
            return
        normalized = _normalize_text(cleaned)
        key = (entity_type, normalized)
        if key in entities:
            return
        entities[key] = DocumentEntity(
            doc_id=metadata.doc_id,
            entity_type=entity_type,
            entity_value=cleaned[:160],
            normalized_value=normalized,
        )

    for match in MONTH_PATTERN.findall(combined_text):
        register("date", match)
    for match in NUMERIC_DATE_PATTERN.findall(combined_text):
        register("date", match)
    for match in DOC_ID_PATTERN.findall(combined_text):
        register("document_id", match)
    for match in REVISION_PATTERN.findall(combined_text):
        register("revision", match)
    for match in CAPITALIZED_ENTITY_PATTERN.findall(combined_text):
        entity_type = _classify_named_entity(match)
        register(entity_type, match)

    register("title", metadata.title)
    return list(entities.values())


def _classify_named_entity(value: str) -> str:
    lowered = value.lower()
    if any(marker in lowered for marker in ORGANIZATION_MARKERS):
        return "organization"
    return "person"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _clean_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())
