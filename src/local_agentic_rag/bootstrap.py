from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass

from .config import AppConfig


@dataclass(slots=True)
class BootstrapReport:
    python_ok: bool
    ollama_installed: bool
    chat_model_available: bool
    embedding_model_available: bool
    vector_backend_ready: bool
    notes: list[str]


def run_bootstrap_checks(config: AppConfig, *, pull_missing: bool = False) -> BootstrapReport:
    notes: list[str] = []
    python_ok = sys.version_info[:2] >= (3, 11)
    if not python_ok:
        notes.append("Python 3.11+ is required.")

    ollama_installed = subprocess.run(
        ["which", "ollama"],
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0
    available_models: set[str] = set()
    if ollama_installed:
        result = subprocess.run(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines()[1:]:
                if line.strip():
                    available_models.add(line.split()[0])
        else:
            notes.append("Ollama is installed, but `ollama list` failed.")
    else:
        notes.append("Ollama is not installed. Install it before runtime.")

    chat_model_available = any(variant in available_models for variant in _model_variants(config.models.chat_model))
    embedding_model_available = any(
        variant in available_models for variant in _model_variants(config.models.embedding_model)
    )

    if pull_missing and ollama_installed:
        for model_name, present in (
            (config.models.chat_model, chat_model_available),
            (config.models.embedding_model, embedding_model_available),
        ):
            if present:
                continue
            subprocess.run(["ollama", "pull", model_name], check=False)
            notes.append(f"Attempted to pull missing model `{model_name}`.")

    vector_backend_ready = True
    if config.retrieval.vector_backend == "faiss":
        vector_backend_ready = importlib.util.find_spec("faiss") is not None
        if not vector_backend_ready:
            notes.append("FAISS backend selected but faiss-cpu is not installed in this environment.")

    notes.append(
        "Configured model profile: "
        f"profile={config.models.profile} "
        f"chat_model={config.models.chat_model} "
        f"embedding_model={config.models.embedding_model}"
    )
    config.ensure_runtime_directories()
    return BootstrapReport(
        python_ok=python_ok,
        ollama_installed=ollama_installed,
        chat_model_available=chat_model_available,
        embedding_model_available=embedding_model_available,
        vector_backend_ready=vector_backend_ready,
        notes=notes,
    )


def _model_variants(model_name: str) -> set[str]:
    variants = {model_name}
    if ":" not in model_name:
        variants.add(f"{model_name}:latest")
    elif model_name.endswith(":latest"):
        variants.add(model_name.rsplit(":", 1)[0])
    return variants
