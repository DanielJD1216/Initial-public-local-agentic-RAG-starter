from pathlib import Path

from tests.conftest import build_test_runtime


def test_delete_document_cascades_chunks(tmp_path: Path) -> None:
    runtime, _config_path, docs_path = build_test_runtime(tmp_path)
    runtime.ingestion.ingest(prune_missing=False)

    before = runtime.store.list_all_chunks()
    assert any(chunk.doc_id == "doc-company-handbook-md" for chunk in before)

    runtime.store.delete_document_by_source_path(str((docs_path / "company_handbook.md").resolve()))

    after = runtime.store.list_all_chunks()
    assert all(chunk.doc_id != "doc-company-handbook-md" for chunk in after)
