import httpx
import pytest

from local_agentic_rag.ingest_bridge import LocalhostBridgeEnrichmentClient, discover_ingest_bridge


def test_bridge_health_reports_reachable_bridge() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"status": "ok", "model": "bridge-test"})
    )
    result = discover_ingest_bridge(
        "127.0.0.1:8787",
        model="bridge-test",
        timeout_seconds=5,
        transport=transport,
    )
    assert result.reachable is True
    assert result.base_url == "http://127.0.0.1:8787"
    assert result.model == "bridge-test"


def test_bridge_client_parses_valid_enrichment_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ingest/enrich"
        return httpx.Response(
            200,
            json={
                "title": "Bridge Title",
                "metadata": {"title": "Bridge Title"},
                "sections": [
                    {
                        "text": "Normalized section text.",
                        "section_path": "Bridge/Overview",
                        "page_number": 1,
                    }
                ],
                "semantic_chunks": [
                    {
                        "text": "Normalized section text.",
                        "section_path": "Bridge/Overview",
                        "page_number": 1,
                        "location_label": "Bridge/Overview | semantic",
                    }
                ],
            },
        )

    client = LocalhostBridgeEnrichmentClient(
        base_url="127.0.0.1:8787",
        model="bridge-test",
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )
    enriched = client.enrich_document(
        source_path="/tmp/doc.md",
        content_type="text/markdown",
        detected_title="Doc",
        sections=[{"text": "Section text", "section_path": "Overview"}],
        stage_flags={"cleanup": True, "semantic_chunking": True, "metadata_enrichment": True},
    )

    assert enriched.title == "Bridge Title"
    assert enriched.sections[0].section_path == "Bridge/Overview"
    assert enriched.semantic_chunks[0].location_label == "Bridge/Overview | semantic"


def test_bridge_client_rejects_malformed_json() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="not-json"))
    client = LocalhostBridgeEnrichmentClient(
        base_url="127.0.0.1:8787",
        model="bridge-test",
        timeout_seconds=5,
        transport=transport,
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        client.enrich_document(
            source_path="/tmp/doc.md",
            content_type="text/markdown",
            detected_title="Doc",
            sections=[{"text": "Section text", "section_path": "Overview"}],
            stage_flags={"cleanup": True, "semantic_chunking": True, "metadata_enrichment": True},
        )


def test_bridge_client_handles_unreachable_bridge() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = LocalhostBridgeEnrichmentClient(
        base_url="127.0.0.1:8787",
        model="bridge-test",
        timeout_seconds=5,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RuntimeError, match="Could not reach ingest bridge"):
        client.enrich_document(
            source_path="/tmp/doc.md",
            content_type="text/markdown",
            detected_title="Doc",
            sections=[{"text": "Section text", "section_path": "Overview"}],
            stage_flags={"cleanup": True, "semantic_chunking": True, "metadata_enrichment": True},
        )


def test_bridge_client_rejects_invalid_schema() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"sections": [], "metadata": {}}))
    client = LocalhostBridgeEnrichmentClient(
        base_url="127.0.0.1:8787",
        model="bridge-test",
        timeout_seconds=5,
        transport=transport,
    )

    with pytest.raises(ValueError, match="missing `title`"):
        client.enrich_document(
            source_path="/tmp/doc.md",
            content_type="text/markdown",
            detected_title="Doc",
            sections=[{"text": "Section text", "section_path": "Overview"}],
            stage_flags={"cleanup": True, "semantic_chunking": True, "metadata_enrichment": True},
        )
