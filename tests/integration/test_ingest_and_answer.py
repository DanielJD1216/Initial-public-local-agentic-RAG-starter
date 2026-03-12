from tests.conftest import build_test_runtime


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
