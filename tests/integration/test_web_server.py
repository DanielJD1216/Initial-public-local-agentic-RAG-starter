from __future__ import annotations

from starlette.testclient import TestClient

from local_agentic_rag.ingest_bridge import BridgeHealthResult, normalize_bridge_base_url
from local_agentic_rag.ollama_admin import OllamaDiscoveryResult, normalize_ollama_base_url
from local_agentic_rag.web_server import WebRuntimeManager, create_web_app
from tests.conftest import FakeBridgeEnrichmentClient, FakeChatClient, FakeEmbeddingClient, build_test_runtime


def fake_ollama_discoverer(base_url: str | None, *, timeout_seconds: int = 5) -> OllamaDiscoveryResult:
    del timeout_seconds
    normalized = normalize_ollama_base_url(base_url)
    model_map = {
        "http://127.0.0.1:11434": ["alt-chat", "alt-embed", "fake-chat", "fake-embed"],
        "http://127.0.0.1:22434": ["alt-chat", "alt-embed", "fake-chat", "fake-embed"],
    }
    if normalized not in model_map:
        return OllamaDiscoveryResult(
            base_url=normalized,
            reachable=False,
            models=[],
            error=f"Could not reach Ollama at {normalized}.",
        )
    return OllamaDiscoveryResult(
        base_url=normalized,
        reachable=True,
        models=model_map[normalized],
    )


def fake_bridge_health_checker(base_url: str | None, *, model: str, timeout_seconds: int = 5) -> BridgeHealthResult:
    del timeout_seconds
    normalized = normalize_bridge_base_url(base_url)
    if normalized != "http://127.0.0.1:8787":
        return BridgeHealthResult(
            base_url=normalized,
            reachable=False,
            model=model,
            error=f"Could not reach ingest bridge at {normalized}.",
        )
    return BridgeHealthResult(base_url=normalized, reachable=True, model=model)


def test_web_server_status_ingest_and_permission_flow(tmp_path) -> None:
    _runtime, config_path, docs_path = build_test_runtime(tmp_path, permissions_enabled=True)
    manager = WebRuntimeManager(
        config_path=config_path,
        ollama_discoverer=fake_ollama_discoverer,
        bridge_health_checker=fake_bridge_health_checker,
        embedding_client=FakeEmbeddingClient(),
        chat_client=FakeChatClient(),
    )
    app = create_web_app(config_path=config_path, manager=manager)
    client = TestClient(app)

    status_before = client.get("/api/status")
    assert status_before.status_code == 200
    assert status_before.json()["corpus"]["document_count"] == 0
    assert status_before.json()["local_models"]["source"] == "config"
    assert status_before.json()["local_models"]["ollama"]["reachable"]
    assert status_before.json()["ingest"]["mode"] == "local"
    assert status_before.json()["ingest"]["corpus"]["mode"] is None
    assert status_before.json()["suggested_prompts"]

    ingest_response = client.post("/api/ingest", json={"documents_path": str(docs_path)})
    assert ingest_response.status_code == 200
    ingest_payload = ingest_response.json()
    assert len(ingest_payload["report"]["processed"]) == 4
    assert ingest_payload["status"]["corpus"]["restricted_document_count"] == 1
    assert any(
        "standard support first response time" in item["prompt"]
        for item in ingest_payload["status"]["suggested_prompts"]
    )

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

    public_prompts = client.post("/api/suggested-prompts", json={"principals": ["staff"]})
    assert public_prompts.status_code == 200
    assert not any("salary adjustment review window" in item["prompt"] for item in public_prompts.json()["prompts"])

    owner_prompts = client.post("/api/suggested-prompts", json={"principals": ["owners"]})
    assert owner_prompts.status_code == 200
    assert any("salary adjustment review window" in item["prompt"] for item in owner_prompts.json()["prompts"])


def test_web_server_can_ingest_browser_selected_folder(tmp_path) -> None:
    _runtime, config_path, docs_path = build_test_runtime(tmp_path, permissions_enabled=False)
    manager = WebRuntimeManager(
        config_path=config_path,
        ollama_discoverer=fake_ollama_discoverer,
        bridge_health_checker=fake_bridge_health_checker,
        embedding_client=FakeEmbeddingClient(),
        chat_client=FakeChatClient(),
    )
    app = create_web_app(config_path=config_path, manager=manager)
    client = TestClient(app)

    handbook = (docs_path / "company_handbook.md").read_bytes()
    handbook_meta = b'title: "Picked Handbook"\naccess_scope: "public"\naccess_principals:\n  - "*"\n'

    ingest_response = client.post(
        "/api/ingest",
        data={"folder_name": "picked-folder"},
        files=[
            ("files", ("picked-folder/company_handbook.md", handbook, "text/markdown")),
            ("files", ("picked-folder/company_handbook.md.meta.yaml", handbook_meta, "application/x-yaml")),
            ("files", ("picked-folder/ignore.bin", b"ignored", "application/octet-stream")),
        ],
    )
    assert ingest_response.status_code == 200
    ingest_payload = ingest_response.json()
    assert ingest_payload["status"]["documents_source"] == "upload"
    assert ingest_payload["status"]["documents_display_path"] == "picked-folder"
    assert len(ingest_payload["report"]["processed"]) == 1

    answer = client.post(
        "/api/ask",
        json={
            "question": "What is the standard support first response time?",
            "principals": ["*"],
        },
    )
    assert answer.status_code == 200
    assert answer.json()["grounded"]
    assert "4 business hours" in answer.json()["answer"]


def test_web_server_model_settings_flow_supports_discovery_apply_reload_and_reindex(tmp_path) -> None:
    _runtime, config_path, docs_path = build_test_runtime(tmp_path, permissions_enabled=False)
    manager = WebRuntimeManager(
        config_path=config_path,
        ollama_discoverer=fake_ollama_discoverer,
        bridge_health_checker=fake_bridge_health_checker,
        embedding_client=FakeEmbeddingClient(),
        chat_client=FakeChatClient(),
    )
    app = create_web_app(config_path=config_path, manager=manager)
    client = TestClient(app)

    initial_status = client.get("/api/status")
    assert initial_status.status_code == 200
    assert initial_status.json()["local_models"]["active"]["chat_model"] == "fake-chat"
    assert initial_status.json()["local_models"]["active"]["embedding_model"] == "fake-embed"
    assert initial_status.json()["local_models"]["source"] == "config"

    discover = client.post("/api/model-settings/discover", json={"base_url": "127.0.0.1:22434"})
    assert discover.status_code == 200
    assert discover.json()["reachable"]
    assert "alt-chat" in discover.json()["models"]

    chat_only_apply = client.post(
        "/api/model-settings/apply",
        json={
            "base_url": "127.0.0.1:22434",
            "chat_model": "alt-chat",
            "embedding_model": "fake-embed",
        },
    )
    assert chat_only_apply.status_code == 200
    chat_only_payload = chat_only_apply.json()
    assert chat_only_payload["applied"]
    assert not chat_only_payload["reindex_required"]
    assert chat_only_payload["status"]["local_models"]["source"] == "session"
    assert chat_only_payload["status"]["local_models"]["active"]["base_url"] == "http://127.0.0.1:22434"
    assert chat_only_payload["status"]["local_models"]["active"]["chat_model"] == "alt-chat"
    assert chat_only_payload["status"]["local_models"]["active"]["embedding_model"] == "fake-embed"

    reload_response = client.post("/api/reload")
    assert reload_response.status_code == 200
    assert reload_response.json()["local_models"]["source"] == "session"
    assert reload_response.json()["local_models"]["active"]["chat_model"] == "alt-chat"

    stage_embedding = client.post(
        "/api/model-settings/apply",
        json={
            "base_url": "127.0.0.1:22434",
            "chat_model": "alt-chat",
            "embedding_model": "alt-embed",
        },
    )
    assert stage_embedding.status_code == 200
    staged_payload = stage_embedding.json()
    assert not staged_payload["applied"]
    assert staged_payload["reindex_required"]
    assert staged_payload["status"]["local_models"]["active"]["embedding_model"] == "fake-embed"
    assert staged_payload["status"]["local_models"]["pending_reindex"]["embedding_model"] == "alt-embed"

    ingest_response = client.post("/api/ingest", json={"documents_path": str(docs_path)})
    assert ingest_response.status_code == 200

    answer_while_staged = client.post(
        "/api/ask",
        json={
            "question": "What is the standard support first response time?",
            "principals": ["*"],
        },
    )
    assert answer_while_staged.status_code == 200
    assert answer_while_staged.json()["grounded"]

    reindex_response = client.post("/api/model-settings/reindex")
    assert reindex_response.status_code == 200
    reindex_payload = reindex_response.json()
    assert reindex_payload["status"]["local_models"]["active"]["chat_model"] == "alt-chat"
    assert reindex_payload["status"]["local_models"]["active"]["embedding_model"] == "alt-embed"
    assert reindex_payload["status"]["local_models"]["pending_reindex"] is None
    assert reindex_payload["status"]["local_models"]["source"] == "session"
    assert len(reindex_payload["report"]["processed"]) == 4

    fresh_manager = WebRuntimeManager(
        config_path=config_path,
        ollama_discoverer=fake_ollama_discoverer,
        bridge_health_checker=fake_bridge_health_checker,
        embedding_client=FakeEmbeddingClient(),
        chat_client=FakeChatClient(),
    )
    fresh_app = create_web_app(config_path=config_path, manager=fresh_manager)
    fresh_client = TestClient(fresh_app)
    fresh_status = fresh_client.get("/api/status")
    assert fresh_status.status_code == 200
    assert fresh_status.json()["local_models"]["source"] == "config"
    assert fresh_status.json()["local_models"]["active"]["chat_model"] == "fake-chat"
    assert fresh_status.json()["local_models"]["active"]["embedding_model"] == "fake-embed"


def test_web_server_can_cancel_staged_embedding_change(tmp_path) -> None:
    _runtime, config_path, _docs_path = build_test_runtime(tmp_path, permissions_enabled=False)
    manager = WebRuntimeManager(
        config_path=config_path,
        ollama_discoverer=fake_ollama_discoverer,
        bridge_health_checker=fake_bridge_health_checker,
        embedding_client=FakeEmbeddingClient(),
        chat_client=FakeChatClient(),
    )
    app = create_web_app(config_path=config_path, manager=manager)
    client = TestClient(app)

    stage_response = client.post(
        "/api/model-settings/apply",
        json={
            "base_url": "127.0.0.1:22434",
            "chat_model": "fake-chat",
            "embedding_model": "alt-embed",
        },
    )
    assert stage_response.status_code == 200
    assert stage_response.json()["reindex_required"]
    assert stage_response.json()["status"]["local_models"]["pending_reindex"]["embedding_model"] == "alt-embed"

    cancel_response = client.post("/api/model-settings/cancel")
    assert cancel_response.status_code == 200
    cancel_payload = cancel_response.json()
    assert cancel_payload["status"]["local_models"]["pending_reindex"] is None
    assert cancel_payload["status"]["local_models"]["active"]["embedding_model"] == "fake-embed"
    assert cancel_payload["status"]["local_models"]["source"] == "config"


def test_web_server_reports_unreachable_ollama_discovery(tmp_path) -> None:
    _runtime, config_path, _docs_path = build_test_runtime(tmp_path, permissions_enabled=False)
    manager = WebRuntimeManager(
        config_path=config_path,
        ollama_discoverer=fake_ollama_discoverer,
        bridge_health_checker=fake_bridge_health_checker,
        embedding_client=FakeEmbeddingClient(),
        chat_client=FakeChatClient(),
    )
    app = create_web_app(config_path=config_path, manager=manager)
    client = TestClient(app)

    discover = client.post("/api/model-settings/discover", json={"base_url": "http://127.0.0.1:9999"})
    assert discover.status_code == 200
    assert not discover.json()["reachable"]
    assert discover.json()["models"] == []
    assert "Could not reach Ollama" in discover.json()["error"]


def test_web_server_reports_bridge_ingest_status_and_local_answers(tmp_path) -> None:
    bridge_client = FakeBridgeEnrichmentClient()
    _runtime, config_path, docs_path = build_test_runtime(
        tmp_path,
        permissions_enabled=False,
        ingest_mode="bridge",
        bridge_client=bridge_client,
    )
    manager = WebRuntimeManager(
        config_path=config_path,
        ollama_discoverer=fake_ollama_discoverer,
        bridge_health_checker=fake_bridge_health_checker,
        embedding_client=FakeEmbeddingClient(),
        chat_client=FakeChatClient(),
        ingest_enrichment_client=bridge_client,
    )
    app = create_web_app(config_path=config_path, manager=manager)
    client = TestClient(app)

    initial_status = client.get("/api/status")
    assert initial_status.status_code == 200
    assert initial_status.json()["ingest"]["mode"] == "bridge"
    assert initial_status.json()["ingest"]["bridge"]["reachable"]

    bridge_health = client.get("/api/ingest/bridge-health")
    assert bridge_health.status_code == 200
    assert bridge_health.json()["reachable"]

    ingest_response = client.post("/api/ingest", json={"documents_path": str(docs_path)})
    assert ingest_response.status_code == 200
    ingest_payload = ingest_response.json()
    assert len(ingest_payload["report"]["processed"]) == 4
    assert ingest_payload["status"]["ingest"]["corpus"]["mode"] == "bridge"
    assert ingest_payload["status"]["ingest"]["corpus"]["chunking_strategy"] == "semantic"
    assert bridge_client.calls == 4

    answer_response = client.post(
        "/api/ask",
        json={
            "question": "What is the standard support first response time?",
            "principals": ["*"],
        },
    )
    assert answer_response.status_code == 200
    assert answer_response.json()["grounded"]
    assert bridge_client.calls == 4
