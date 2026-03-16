from pathlib import Path

from tests.conftest import build_test_runtime


def test_delete_document_cascades_chunks(tmp_path: Path) -> None:
    runtime, _config_path, docs_path = build_test_runtime(tmp_path)
    runtime.ingestion.ingest(prune_missing=False)
    ingest_summary = runtime.store.get_corpus_ingest_summary()
    assert ingest_summary.mode == "local"
    assert ingest_summary.chunking_strategy == "heuristic"
    artifact_status = runtime.store.get_planning_artifact_status()
    assert artifact_status.available
    assert artifact_status.ready_document_count == ingest_summary.document_count

    before = runtime.store.list_all_chunks()
    assert any(chunk.doc_id == "doc-company-handbook-md" for chunk in before)
    artifact = runtime.store.get_document_planning_artifact("doc-company-handbook-md")
    assert artifact is not None
    assert artifact.section_outline

    runtime.store.delete_document_by_source_path(str((docs_path / "company_handbook.md").resolve()))

    after = runtime.store.list_all_chunks()
    assert all(chunk.doc_id != "doc-company-handbook-md" for chunk in after)
    assert runtime.store.get_document_planning_artifact("doc-company-handbook-md") is None
