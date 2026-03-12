from pathlib import Path

from tests.conftest import build_test_runtime


def test_permissions_and_reindex(tmp_path: Path) -> None:
    runtime, _config_path, docs_path = build_test_runtime(tmp_path, permissions_enabled=True)
    first_report = runtime.ingestion.ingest(prune_missing=False)
    assert not first_report.errors

    restricted = runtime.agent.answer(
        "When is the salary adjustment review window planned?",
        active_principals=["staff"],
    )
    assert not restricted.grounded
    assert restricted.citations == []

    allowed = runtime.agent.answer(
        "When is the salary adjustment review window planned?",
        active_principals=["owners"],
    )
    assert allowed.grounded
    assert "third week of June" in allowed.answer

    (docs_path / "pricing_notes.txt").unlink()
    second_report = runtime.ingestion.ingest(prune_missing=True)
    assert any("pricing_notes.txt" in item for item in second_report.deleted)
