from __future__ import annotations

from starlette.testclient import TestClient

from local_agentic_rag.web_server import WebRuntimeManager, create_web_app
from tests.conftest import FakeChatClient, FakeEmbeddingClient, build_test_runtime


def test_web_server_status_ingest_and_permission_flow(tmp_path) -> None:
    _runtime, config_path, docs_path = build_test_runtime(tmp_path, permissions_enabled=True)
    manager = WebRuntimeManager(
        config_path=config_path,
        embedding_client=FakeEmbeddingClient(),
        chat_client=FakeChatClient(),
    )
    app = create_web_app(config_path=config_path, manager=manager)
    client = TestClient(app)

    status_before = client.get("/api/status")
    assert status_before.status_code == 200
    assert status_before.json()["corpus"]["document_count"] == 0

    ingest_response = client.post("/api/ingest", json={"documents_path": str(docs_path)})
    assert ingest_response.status_code == 200
    ingest_payload = ingest_response.json()
    assert len(ingest_payload["report"]["processed"]) == 4
    assert ingest_payload["status"]["corpus"]["restricted_document_count"] == 1

    blocked = client.post(
        "/api/ask",
        json={
            "question": "When is the salary adjustment review window planned?",
            "principals": ["staff"],
        },
    )
    assert blocked.status_code == 200
    assert not blocked.json()["grounded"]

    allowed = client.post(
        "/api/ask",
        json={
            "question": "When is the salary adjustment review window planned?",
            "principals": ["owners"],
        },
    )
    assert allowed.status_code == 200
    assert allowed.json()["grounded"]
    assert "third week of June" in allowed.json()["answer"]
