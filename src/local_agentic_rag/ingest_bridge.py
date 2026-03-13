from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Protocol

import httpx


BRIDGE_SCHEMA_VERSION = "bridge-enrichment/v1"
LOCAL_INGEST_MODEL = "local-heuristic"


@dataclass(slots=True)
class SemanticChunkPlan:
    text: str
    section_path: str
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    location_label: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class EnrichedSection:
    text: str
    section_path: str
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class EnrichedDocument:
    title: str
    sections: list[EnrichedSection]
    metadata: dict[str, object]
    semantic_chunks: list[SemanticChunkPlan]


@dataclass(slots=True)
class BridgeHealthResult:
    base_url: str
    reachable: bool | None
    model: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "base_url": self.base_url,
            "reachable": self.reachable,
            "model": self.model,
            "error": self.error,
        }


class IngestEnrichmentClient(Protocol):
    def enrich_document(
        self,
        *,
        source_path: str,
        content_type: str,
        detected_title: str,
        sections: list[dict[str, object]],
        stage_flags: dict[str, bool],
    ) -> EnrichedDocument:
        ...


def normalize_bridge_base_url(raw_base_url: str | None) -> str:
    value = (raw_base_url or "").strip()
    if not value:
        return "http://127.0.0.1:8787"
    if "://" not in value:
        value = f"http://{value}"
    return value.rstrip("/")


def compute_ingest_fingerprint(
    *,
    parser_version: str,
    mode: str,
    ingest_model: str,
    cleanup: bool,
    semantic_chunking: bool,
    metadata_enrichment: bool,
    max_chunk_tokens: int,
    overlap_tokens: int,
    schema_version: str = BRIDGE_SCHEMA_VERSION,
) -> str:
    payload = {
        "parser_version": parser_version,
        "mode": mode,
        "ingest_model": ingest_model,
        "cleanup": cleanup,
        "semantic_chunking": semantic_chunking,
        "metadata_enrichment": metadata_enrichment,
        "max_chunk_tokens": max_chunk_tokens,
        "overlap_tokens": overlap_tokens,
        "schema_version": schema_version,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def discover_ingest_bridge(
    base_url: str | None,
    *,
    model: str,
    timeout_seconds: int = 5,
    transport: httpx.BaseTransport | None = None,
) -> BridgeHealthResult:
    normalized = normalize_bridge_base_url(base_url)
    try:
        with httpx.Client(timeout=timeout_seconds, transport=transport) as client:
            response = client.get(f"{normalized}/health")
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        return BridgeHealthResult(
            base_url=normalized,
            reachable=False,
            model=model,
            error=f"Bridge responded with HTTP {exc.response.status_code} at {normalized}.",
        )
    except httpx.HTTPError as exc:
        return BridgeHealthResult(
            base_url=normalized,
            reachable=False,
            model=model,
            error=f"Could not reach ingest bridge at {normalized}: {exc}.",
        )
    except ValueError:
        return BridgeHealthResult(
            base_url=normalized,
            reachable=False,
            model=model,
            error=f"Ingest bridge at {normalized} returned an unreadable health response.",
        )

    reported_model = str(payload.get("model") or model)
    return BridgeHealthResult(
        base_url=normalized,
        reachable=True,
        model=reported_model,
        error=str(payload.get("error")) if payload.get("error") else None,
    )


class LocalhostBridgeEnrichmentClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: int = 60,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = normalize_bridge_base_url(base_url)
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def enrich_document(
        self,
        *,
        source_path: str,
        content_type: str,
        detected_title: str,
        sections: list[dict[str, object]],
        stage_flags: dict[str, bool],
    ) -> EnrichedDocument:
        payload = {
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "model": self.model,
            "stages": stage_flags,
            "document": {
                "source_path": source_path,
                "content_type": content_type,
                "detected_title": detected_title,
                "sections": sections,
            },
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = client.post(f"{self.base_url}/api/ingest/enrich", json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ingest bridge at {self.base_url} timed out after {self.timeout_seconds}s while enriching `{source_path}`."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ingest bridge returned HTTP {exc.response.status_code} while enriching `{source_path}`."
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Could not reach ingest bridge at {self.base_url}: {exc}.") from exc
        except ValueError as exc:
            raise ValueError(f"Ingest bridge at {self.base_url} returned invalid JSON.") from exc

        return _parse_enriched_document(body)


def _parse_enriched_document(body: object) -> EnrichedDocument:
    if not isinstance(body, dict):
        raise ValueError("Bridge response must be a JSON object.")

    title = str(body.get("title") or "").strip()
    if not title:
        raise ValueError("Bridge response is missing `title`.")

    raw_sections = body.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ValueError("Bridge response must include at least one normalized section.")
    sections = [_parse_section(item) for item in raw_sections]

    raw_metadata = body.get("metadata", {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("Bridge response field `metadata` must be an object.")
    metadata = dict(raw_metadata)

    raw_chunks = body.get("semantic_chunks", [])
    if not isinstance(raw_chunks, list):
        raise ValueError("Bridge response field `semantic_chunks` must be a list.")
    semantic_chunks = [_parse_semantic_chunk(item) for item in raw_chunks]

    return EnrichedDocument(
        title=title,
        sections=sections,
        metadata=metadata,
        semantic_chunks=semantic_chunks,
    )


def _parse_section(payload: object) -> EnrichedSection:
    if not isinstance(payload, dict):
        raise ValueError("Each normalized section must be an object.")
    text = str(payload.get("text") or "").strip()
    section_path = str(payload.get("section_path") or "").strip()
    if not text or not section_path:
        raise ValueError("Each normalized section must include non-empty `text` and `section_path`.")
    return EnrichedSection(
        text=text,
        section_path=section_path,
        page_number=_optional_int(payload.get("page_number")),
        line_start=_optional_int(payload.get("line_start")),
        line_end=_optional_int(payload.get("line_end")),
    )


def _parse_semantic_chunk(payload: object) -> SemanticChunkPlan:
    if not isinstance(payload, dict):
        raise ValueError("Each semantic chunk must be an object.")
    text = str(payload.get("text") or "").strip()
    section_path = str(payload.get("section_path") or "").strip()
    if not text or not section_path:
        raise ValueError("Each semantic chunk must include non-empty `text` and `section_path`.")
    return SemanticChunkPlan(
        text=text,
        section_path=section_path,
        page_number=_optional_int(payload.get("page_number")),
        line_start=_optional_int(payload.get("line_start")),
        line_end=_optional_int(payload.get("line_end")),
        location_label=str(payload.get("location_label") or "").strip(),
    )


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid line or page numbers.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected an integer-compatible value, got `{value}`.") from exc
