import yaml

from local_agentic_rag.service import build_runtime
from tests.conftest import (
    FakeBridgeEnrichmentClient,
    FakeChatClient,
    FakeEmbeddingClient,
    build_test_runtime,
    write_test_config,
)


def test_ingest_and_answer_flow(tmp_path) -> None:
    runtime, _config_path, _docs_path = build_test_runtime(tmp_path)
    report = runtime.ingestion.ingest(prune_missing=False)
    assert len(report.processed) == 4
    assert not report.errors

    corpus_summary = runtime.store.get_corpus_summary()
    assert corpus_summary.document_count == 4
    assert corpus_summary.restricted_document_count == 1
    assert "owners" in corpus_summary.principals

    simple = runtime.agent.answer(
        "What is the standard support first response time?",
        active_principals=["*"],
    )
    assert simple.grounded
    assert "4 business hours" in simple.answer
    assert simple.citations

    multi_hop = runtime.agent.answer(
        "Who owns support escalations and when is the postmortem due for a customer-facing incident?",
        active_principals=["*"],
    )
    assert multi_hop.grounded
    assert "Maya Chen" in multi_hop.answer
    assert "2 business days" in multi_hop.answer
    assert len(multi_hop.citations) >= 2

    vague = runtime.agent.answer(
        "What should I know about support?",
        active_principals=["*"],
    )
    assert vague.trace.rewritten_query is not None
    assert vague.trace.attempts[-1].rewritten


def test_bridge_ingest_fingerprint_forces_reingest_when_bridge_model_changes(tmp_path) -> None:
    bridge_client = FakeBridgeEnrichmentClient()
    runtime, config_path, _docs_path = build_test_runtime(
        tmp_path,
        ingest_mode="bridge",
        bridge_client=bridge_client,
    )

    first_report = runtime.ingestion.ingest(prune_missing=False)
    assert len(first_report.processed) == 4
    assert bridge_client.calls == 4

    second_report = runtime.ingestion.ingest(prune_missing=False)
    assert len(second_report.skipped) == 4

    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_payload["ingest"]["bridge_model"] = "fake-bridge-v2"
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

    reloaded_runtime = build_runtime(
        config_path=config_path,
        embedding_client=FakeEmbeddingClient(),
        chat_client=FakeChatClient(),
        ingest_enrichment_client=bridge_client,
    )
    third_report = reloaded_runtime.ingestion.ingest(prune_missing=False)
    assert len(third_report.processed) == 4
    assert bridge_client.calls == 8

    answer = reloaded_runtime.agent.answer(
        "What is the standard support first response time?",
        active_principals=["*"],
    )
    assert answer.grounded
    assert bridge_client.calls == 8


def test_middleweight_runtime_falls_back_when_planning_artifacts_are_missing(tmp_path) -> None:
    runtime, _config_path, _docs_path = build_test_runtime(tmp_path)
    runtime.ingestion.ingest(prune_missing=False)

    with runtime.store._connect() as connection:
        connection.execute("DELETE FROM document_artifacts")
        connection.commit()

    status = runtime.agent.runtime_status()
    assert status.configured_mode == "middleweight"
    assert status.active_mode == "lightweight"
    assert status.reindex_required_for_middleweight


def test_middleweight_trace_uses_title_and_metadata_tools_for_named_document_query(tmp_path) -> None:
    runtime, _config_path, _docs_path = build_test_runtime(tmp_path)
    runtime.ingestion.ingest(prune_missing=False)

    answer = runtime.agent.answer(
        'What does "Incident Playbook" say about postmortems?',
        active_principals=["*"],
    )

    tool_names = [event.tool_name for event in answer.tool_events]
    assert "title_search" in tool_names
    assert "collect_evidence_set" in tool_names


def test_middleweight_reports_conflicting_sources_for_cross_document_fact_conflicts(tmp_path) -> None:
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    (docs_path / "support_a.md").write_text(
        "# Support Guide A\n\n## Support Coverage\n\nStandard requests receive a first response within 4 business hours.\n",
        encoding="utf-8",
    )
    (docs_path / "support_b.md").write_text(
        "# Support Guide B\n\n## Support Coverage\n\nStandard requests receive a first response within 6 business hours.\n",
        encoding="utf-8",
    )
    config_path = write_test_config(tmp_path, docs_path, permissions_enabled=False)
    runtime = build_runtime(
        config_path=config_path,
        embedding_client=FakeEmbeddingClient(),
        chat_client=FakeChatClient(),
    )
    runtime.ingestion.ingest(prune_missing=False)

    result = runtime.agent.answer(
        "Across the documents, what is the standard support first response time?",
        active_principals=["*"],
    )

    assert not result.grounded
    assert result.status == "conflicting_sources"
    assert result.failure_reason == "conflicting_sources"
