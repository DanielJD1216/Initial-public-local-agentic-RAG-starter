from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from local_agentic_rag import cli


def test_serve_ui_uses_config_root_for_streamlit_script(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_load_config(_config: str):
        return SimpleNamespace(
            root_dir=tmp_path,
            ui=SimpleNamespace(host="127.0.0.1", port=8501),
        )

    def fake_call(command: list[str]) -> int:
        captured["command"] = command
        return 0

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli.subprocess, "call", fake_call)

    with pytest.raises(SystemExit) as exc_info:
        cli.serve_ui(config="config.small.permissions.yaml")

    assert exc_info.value.code == 0
    command = captured["command"]
    assert str(tmp_path / "streamlit_app.py") in command
