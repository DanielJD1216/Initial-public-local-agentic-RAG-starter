from __future__ import annotations

from pathlib import Path

from local_agentic_rag.service import build_runtime
from tests.conftest import FakeChatClient, FakeEmbeddingClient, write_test_config


def test_unlabeled_confidential_doc_is_auto_restricted_for_public_queries(tmp_path: Path) -> None:
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    (docs_path / "public_handbook.md").write_text(
        "# Public Handbook\n\n## Support\n\nStandard requests receive a first response within 4 business hours.\n",
        encoding="utf-8",
    )
    restricted_path = docs_path / "compensation_notes.md"
    restricted_path.write_text(
        "# Compensation Notes\n\n## Bonus Planning\n\nConfidential. The bonus review window opens on July 10.\n",
        encoding="utf-8",
    )

    config_path = write_test_config(tmp_path, docs_path, permissions_enabled=True)
    runtime = build_runtime(
        config_path=config_path,
        embedding_client=FakeEmbeddingClient(),
        chat_client=FakeChatClient(),
    )

    report = runtime.ingestion.ingest(prune_missing=True)
    assert not report.errors

    stored = runtime.store.get_document_by_source_path(str(restricted_path.resolve()))
    assert stored is not None
    assert stored.access_scope == "restricted"
    assert stored.access_principals == ["owners"]

    public_result = runtime.agent.answer(
        "When does the bonus review window open?",
        active_principals=["*"],
    )
    assert not public_result.grounded

    owner_result = runtime.agent.answer(
        "When does the bonus review window open?",
        active_principals=["owners"],
    )
    assert owner_result.grounded
    assert "July 10" in owner_result.answer
