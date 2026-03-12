from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


MODEL_PROFILES: dict[str, dict[str, str]] = {
    "small": {
        "chat_model": "qwen3:4b",
        "embedding_model": "nomic-embed-text",
    },
    "balanced": {
        "chat_model": "qwen3:8b",
        "embedding_model": "nomic-embed-text",
    },
}


@dataclass(slots=True)
class PathSettings:
    documents: Path
    database: Path
    vector_index: Path
    vector_metadata: Path
    cache_dir: Path

    def ensure_directories(self) -> None:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.vector_index.parent.mkdir(parents=True, exist_ok=True)
        self.vector_metadata.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ModelSettings:
    profile: str
    chat_model: str
    embedding_model: str
    base_url: str
    request_timeout_seconds: int = 120
    disable_thinking: bool = True


@dataclass(slots=True)
class RetrievalSettings:
    top_k: int = 6
    keyword_k: int = 10
    vector_k: int = 10
    max_chunk_tokens: int = 260
    overlap_tokens: int = 40
    min_evidence_score: float = 0.42
    rrf_k: int = 60
    vector_backend: str = "faiss"


@dataclass(slots=True)
class PermissionSettings:
    enabled: bool = False
    default_access_scope: str = "public"
    default_access_principals: list[str] = field(default_factory=lambda: ["*"])
    active_principals: list[str] = field(default_factory=lambda: ["*"])


@dataclass(slots=True)
class UISettings:
    host: str = "127.0.0.1"
    port: int = 8501


@dataclass(slots=True)
class WebSettings:
    host: str = "127.0.0.1"
    port: int = 3000


@dataclass(slots=True)
class MCPSettings:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(slots=True)
class AppConfig:
    version: int
    project_name: str
    root_dir: Path
    paths: PathSettings
    models: ModelSettings
    retrieval: RetrievalSettings
    permissions: PermissionSettings
    ui: UISettings
    web: WebSettings
    mcp: MCPSettings

    def ensure_runtime_directories(self) -> None:
        self.paths.ensure_directories()


def _parse_bool(raw_value: str) -> bool:
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _resolve_path(root_dir: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (root_dir / candidate).resolve()


def _apply_env_overrides(raw_config: dict[str, Any]) -> dict[str, Any]:
    env_map: dict[str, tuple[str, Any]] = {
        "RAG_DOCUMENTS_PATH": ("paths.documents", str),
        "RAG_CHAT_MODEL": ("models.chat_model", str),
        "RAG_EMBEDDING_MODEL": ("models.embedding_model", str),
        "RAG_OLLAMA_BASE_URL": ("models.base_url", str),
        "RAG_MODEL_PROFILE": ("models.profile", str),
        "RAG_DISABLE_THINKING": ("models.disable_thinking", _parse_bool),
        "RAG_PERMISSION_ENABLED": ("permissions.enabled", _parse_bool),
        "RAG_ACTIVE_PRINCIPALS": ("permissions.active_principals", _parse_list),
        "RAG_VECTOR_BACKEND": ("retrieval.vector_backend", str),
    }
    updated = dict(raw_config)
    for env_key, (dotted_path, converter) in env_map.items():
        if env_key not in os.environ:
            continue
        target = updated
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = converter(os.environ[env_key])
    return updated


def _deep_merge_profile_defaults(raw_config: dict[str, Any]) -> dict[str, Any]:
    updated = dict(raw_config)
    models = dict(updated.get("models", {}))
    profile = models.get("profile", "balanced")
    defaults = MODEL_PROFILES.get(profile, MODEL_PROFILES["balanced"])
    models.setdefault("chat_model", defaults["chat_model"])
    models.setdefault("embedding_model", defaults["embedding_model"])
    updated["models"] = models
    return updated


def load_config(config_path: str | Path | None = None) -> AppConfig:
    load_dotenv()
    path = Path(config_path or os.environ.get("RAG_CONFIG_PATH", "config.yaml")).expanduser().resolve()
    root_dir = path.parent
    raw_config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_config = _deep_merge_profile_defaults(raw_config)
    raw_config = _apply_env_overrides(raw_config)

    paths = raw_config["paths"]
    models = raw_config["models"]
    retrieval = raw_config.get("retrieval", {})
    permissions = raw_config.get("permissions", {})
    ui = raw_config.get("ui", {})
    web = raw_config.get("web", {})
    mcp = raw_config.get("mcp", {})

    return AppConfig(
        version=int(raw_config.get("version", 1)),
        project_name=str(raw_config.get("project_name", "Local Agentic RAG Starter")),
        root_dir=root_dir,
        paths=PathSettings(
            documents=_resolve_path(root_dir, str(paths["documents"])),
            database=_resolve_path(root_dir, str(paths["database"])),
            vector_index=_resolve_path(root_dir, str(paths["vector_index"])),
            vector_metadata=_resolve_path(root_dir, str(paths["vector_metadata"])),
            cache_dir=_resolve_path(root_dir, str(paths["cache_dir"])),
        ),
        models=ModelSettings(
            profile=str(models.get("profile", "balanced")),
            chat_model=str(models["chat_model"]),
            embedding_model=str(models["embedding_model"]),
            base_url=str(models.get("base_url", "http://127.0.0.1:11434")),
            request_timeout_seconds=int(models.get("request_timeout_seconds", 120)),
            disable_thinking=bool(models.get("disable_thinking", True)),
        ),
        retrieval=RetrievalSettings(
            top_k=int(retrieval.get("top_k", 6)),
            keyword_k=int(retrieval.get("keyword_k", 10)),
            vector_k=int(retrieval.get("vector_k", 10)),
            max_chunk_tokens=int(retrieval.get("max_chunk_tokens", 260)),
            overlap_tokens=int(retrieval.get("overlap_tokens", 40)),
            min_evidence_score=float(retrieval.get("min_evidence_score", 0.42)),
            rrf_k=int(retrieval.get("rrf_k", 60)),
            vector_backend=str(retrieval.get("vector_backend", "faiss")),
        ),
        permissions=PermissionSettings(
            enabled=bool(permissions.get("enabled", False)),
            default_access_scope=str(permissions.get("default_access_scope", "public")),
            default_access_principals=list(permissions.get("default_access_principals", ["*"])),
            active_principals=list(permissions.get("active_principals", ["*"])),
        ),
        ui=UISettings(
            host=str(ui.get("host", "127.0.0.1")),
            port=int(ui.get("port", 8501)),
        ),
        web=WebSettings(
            host=str(web.get("host", "127.0.0.1")),
            port=int(web.get("port", 3000)),
        ),
        mcp=MCPSettings(
            host=str(mcp.get("host", "127.0.0.1")),
            port=int(mcp.get("port", 8000)),
        ),
    )


def model_profile_summary(profile: str) -> str:
    defaults = MODEL_PROFILES.get(profile, MODEL_PROFILES["balanced"])
    return (
        f"profile={profile} chat_model={defaults['chat_model']} "
        f"embedding_model={defaults['embedding_model']}"
    )
