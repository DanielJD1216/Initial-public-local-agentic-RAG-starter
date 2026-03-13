from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable

from .clients import ChatClient, EmbeddingClient
from .config import AppConfig, load_config
from .ingest_bridge import BridgeHealthResult, IngestEnrichmentClient, discover_ingest_bridge, normalize_bridge_base_url
from .ollama_admin import OllamaDiscoveryResult, discover_ollama_models, normalize_ollama_base_url
from .models import PromptSuggestion
from .prompt_suggestions import build_prompt_suggestions, default_prompt_suggestions
from .service import AppRuntime, build_runtime_from_config
from .utils import SIDECAR_SUFFIXES, SUPPORTED_EXTENSIONS, slugify


@dataclass(slots=True)
class ModelSettingsSelection:
    base_url: str
    chat_model: str
    embedding_model: str
    profile: str = "custom"

    def to_dict(self, *, source: str) -> dict[str, object]:
        return {
            "profile": self.profile,
            "base_url": self.base_url,
            "chat_model": self.chat_model,
            "embedding_model": self.embedding_model,
            "source": source,
        }

    def matches(self, other: "ModelSettingsSelection") -> bool:
        return (
            self.base_url == other.base_url
            and self.chat_model == other.chat_model
            and self.embedding_model == other.embedding_model
        )


@dataclass(slots=True)
class WebRuntimeManager:
    config_path: Path
    runtime_builder: Callable[..., AppRuntime] = build_runtime_from_config
    config_loader: Callable[[str | Path | None], AppConfig] = load_config
    ollama_discoverer: Callable[..., OllamaDiscoveryResult] = discover_ollama_models
    bridge_health_checker: Callable[..., BridgeHealthResult] = discover_ingest_bridge
    embedding_client: EmbeddingClient | None = None
    chat_client: ChatClient | None = None
    ingest_enrichment_client: IngestEnrichmentClient | None = None
    _lock: Lock = field(init=False, repr=False)
    _runtime: AppRuntime = field(init=False, repr=False)
    _documents_display_path: str = field(init=False, repr=False)
    _documents_source: str = field(init=False, repr=False, default="path")
    _session_model_override: ModelSettingsSelection | None = field(init=False, repr=False, default=None)
    _pending_model_selection: ModelSettingsSelection | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._lock = Lock()
        self._runtime = self._build_runtime()
        self._documents_display_path = str(self._runtime.config.paths.documents)
        self._documents_source = "path"

    @property
    def runtime(self) -> AppRuntime:
        return self._runtime

    def status_payload(self) -> dict[str, object]:
        corpus = self.runtime.store.get_corpus_summary()
        ingest_summary = self.runtime.store.get_corpus_ingest_summary()
        active_selection = self._current_model_selection()
        discovery = self._discover_ollama(active_selection.base_url)
        suggested_prompts = self._suggested_prompts(
            principals=list(self.runtime.config.permissions.active_principals),
        )
        local_models_payload = {
            "active": active_selection.to_dict(source=self._current_model_source()),
            "source": self._current_model_source(),
            "pending_reindex": self._pending_model_payload(),
            "ollama": discovery.to_dict(),
        }
        return {
            "project_name": self.runtime.config.project_name,
            "config_path": str(self.config_path),
            "documents_path": str(self.runtime.config.paths.documents),
            "documents_display_path": self._documents_display_path,
            "documents_source": self._documents_source,
            "local_models": local_models_payload,
            "models": local_models_payload,
            "ingest": {
                "mode": self.runtime.config.ingest.mode,
                "bridge": self._current_bridge_status().to_dict(),
                "corpus": ingest_summary.to_dict(),
            },
            "permissions_enabled": self.runtime.config.permissions.enabled,
            "default_principals": list(self.runtime.config.permissions.active_principals),
            "corpus": corpus.to_dict(),
            "suggested_prompts": [item.to_dict() for item in suggested_prompts],
        }

    def reload(self) -> dict[str, object]:
        with self._lock:
            self._runtime = self._build_runtime(documents_path=self.runtime.config.paths.documents)
            return self.status_payload()

    def discover_models(self, *, base_url: str | None = None) -> dict[str, object]:
        with self._lock:
            target_base_url = base_url or self._current_model_selection().base_url
            return self._discover_ollama(target_base_url).to_dict()

    def bridge_health_payload(self) -> dict[str, object]:
        with self._lock:
            return self._current_bridge_status().to_dict()

    def apply_model_settings(
        self,
        *,
        base_url: str,
        chat_model: str,
        embedding_model: str,
    ) -> dict[str, object]:
        with self._lock:
            requested_selection = self._validated_model_selection(
                base_url=base_url,
                chat_model=chat_model,
                embedding_model=embedding_model,
            )
            active_selection = self._current_model_selection()
            if requested_selection.embedding_model != active_selection.embedding_model:
                self._pending_model_selection = requested_selection
                return {
                    "status": self.status_payload(),
                    "applied": False,
                    "reindex_required": True,
                    "message": "Embedding changes are staged. Reindex the current corpus to activate them.",
                }

            self._pending_model_selection = None
            self._apply_active_model_selection(requested_selection)
            return {
                "status": self.status_payload(),
                "applied": True,
                "reindex_required": False,
                "message": "Session model settings applied to the running runtime.",
            }

    def reindex_pending_model_settings(self) -> dict[str, object]:
        with self._lock:
            if self._pending_model_selection is None:
                raise ValueError("There is no staged embedding-model change to reindex.")

            runtime = self._build_runtime(
                documents_path=self.runtime.config.paths.documents,
                model_selection=self._pending_model_selection,
                skip_retriever_validation=True,
            )
            report = runtime.ingestion.ingest(prune_missing=True, force_embeddings=True)
            self._runtime = runtime
            self._session_model_override = self._selection_to_session_override(self._pending_model_selection)
            self._pending_model_selection = None
            return {
                "status": self.status_payload(),
                "report": {
                    "processed": report.processed,
                    "skipped": report.skipped,
                    "deleted": report.deleted,
                    "errors": report.errors,
                },
                "message": "Reindex complete. The staged embedding model is now active.",
            }

    def cancel_pending_model_settings(self) -> dict[str, object]:
        with self._lock:
            self._pending_model_selection = None
            return {
                "status": self.status_payload(),
                "message": "The staged embedding-model change was cleared.",
            }

    def ingest(
        self,
        *,
        documents_path: str | None = None,
        force_embeddings: bool = False,
    ) -> dict[str, object]:
        with self._lock:
            resolved_path = _resolve_documents_override(documents_path)
            runtime = self._build_runtime(documents_path=resolved_path)
            report = runtime.ingestion.ingest(prune_missing=True, force_embeddings=force_embeddings)
            self._runtime = runtime
            if resolved_path is not None:
                self._documents_source = "path"
                self._documents_display_path = str(resolved_path)
            return {
                "status": self.status_payload(),
                "report": {
                    "processed": report.processed,
                    "skipped": report.skipped,
                    "deleted": report.deleted,
                    "errors": report.errors,
                },
            }

    def ingest_uploaded_files(
        self,
        *,
        uploaded_files: list[object],
        folder_name: str | None = None,
        force_embeddings: bool = False,
    ) -> dict[str, object]:
        with self._lock:
            upload_root = self.runtime.config.paths.cache_dir / "web_uploads" / "current"
            staged_paths = _stage_uploaded_files(upload_root, uploaded_files)
            runtime = self._build_runtime(documents_path=upload_root)
            report = runtime.ingestion.ingest(prune_missing=True, force_embeddings=force_embeddings)
            self._runtime = runtime
            self._documents_source = "upload"
            self._documents_display_path = folder_name or _infer_folder_label(staged_paths) or "Uploaded folder"
            return {
                "status": self.status_payload(),
                "report": {
                    "processed": report.processed,
                    "skipped": report.skipped,
                    "deleted": report.deleted,
                    "errors": report.errors,
                },
            }

    def ask(self, *, question: str, principals: list[str]) -> dict[str, object]:
        with self._lock:
            result = self.runtime.agent.answer(question, active_principals=principals)
        return result.to_dict()

    def suggested_prompts_payload(self, *, principals: list[str] | None = None) -> dict[str, object]:
        with self._lock:
            selected_principals = principals if principals is not None else list(self.runtime.config.permissions.active_principals)
            prompts = self._suggested_prompts(principals=selected_principals)
        return {"prompts": [item.to_dict() for item in prompts]}

    def _build_runtime(
        self,
        documents_path: str | Path | None = None,
        *,
        model_selection: ModelSettingsSelection | None = None,
        skip_retriever_validation: bool = False,
    ) -> AppRuntime:
        config = self.config_loader(self.config_path)
        config_selection = ModelSettingsSelection(
            base_url=normalize_ollama_base_url(config.models.base_url),
            chat_model=config.models.chat_model,
            embedding_model=config.models.embedding_model,
            profile=config.models.profile,
        )
        if documents_path is not None:
            candidate = _resolve_documents_override(documents_path)
            config.paths.documents = candidate
        selected_models = model_selection if model_selection is not None else self._session_model_override
        if selected_models is not None:
            config.models.profile = selected_models.profile
            config.models.base_url = selected_models.base_url
            config.models.chat_model = selected_models.chat_model
            config.models.embedding_model = selected_models.embedding_model
            if selected_models.embedding_model != config_selection.embedding_model:
                session_root = config.paths.cache_dir / "web_model_sessions" / slugify(selected_models.embedding_model)
                config.paths.database = session_root / "rag.sqlite3"
                config.paths.vector_index = session_root / "vector.index"
                config.paths.vector_metadata = session_root / "vector.meta.json"
        return self.runtime_builder(
            config=config,
            embedding_client=self.embedding_client,
            chat_client=self.chat_client,
            ingest_enrichment_client=self.ingest_enrichment_client,
            skip_retriever_validation=skip_retriever_validation,
        )

    def _apply_active_model_selection(self, selection: ModelSettingsSelection) -> None:
        self._session_model_override = self._selection_to_session_override(selection)
        self._runtime = self._build_runtime(documents_path=self.runtime.config.paths.documents)

    def _discover_ollama(self, base_url: str) -> OllamaDiscoveryResult:
        return self.ollama_discoverer(base_url, timeout_seconds=5)

    def _current_bridge_status(self) -> BridgeHealthResult:
        if self.runtime.config.ingest.mode != "bridge":
            return BridgeHealthResult(
                base_url=normalize_bridge_base_url(self.runtime.config.ingest.bridge_base_url),
                reachable=None,
                model=self.runtime.config.ingest.bridge_model,
                error=None,
            )
        return self.bridge_health_checker(
            self.runtime.config.ingest.bridge_base_url,
            model=self.runtime.config.ingest.bridge_model,
            timeout_seconds=self.runtime.config.ingest.request_timeout_seconds,
        )

    def _current_model_source(self) -> str:
        return "session" if self._session_model_override is not None else "config"

    def _current_model_selection(self) -> ModelSettingsSelection:
        return ModelSettingsSelection(
            base_url=normalize_ollama_base_url(self.runtime.config.models.base_url),
            chat_model=self.runtime.config.models.chat_model,
            embedding_model=self.runtime.config.models.embedding_model,
            profile=self.runtime.config.models.profile,
        )

    def _config_model_selection(self) -> ModelSettingsSelection:
        config = self.config_loader(self.config_path)
        return ModelSettingsSelection(
            base_url=normalize_ollama_base_url(config.models.base_url),
            chat_model=config.models.chat_model,
            embedding_model=config.models.embedding_model,
            profile=config.models.profile,
        )

    def _selection_to_session_override(self, selection: ModelSettingsSelection) -> ModelSettingsSelection | None:
        config_selection = self._config_model_selection()
        if selection.matches(config_selection):
            return None
        return ModelSettingsSelection(
            base_url=selection.base_url,
            chat_model=selection.chat_model,
            embedding_model=selection.embedding_model,
            profile="custom",
        )

    def _pending_model_payload(self) -> dict[str, object] | None:
        if self._pending_model_selection is None:
            return None
        target_source = "config" if self._selection_to_session_override(self._pending_model_selection) is None else "session"
        return self._pending_model_selection.to_dict(source=target_source)

    def _validated_model_selection(
        self,
        *,
        base_url: str,
        chat_model: str,
        embedding_model: str,
    ) -> ModelSettingsSelection:
        discovery = self._discover_ollama(base_url)
        if not discovery.reachable:
            raise ValueError(discovery.error or f"Could not reach Ollama at {discovery.base_url}.")
        available_models = set(discovery.models)
        if not chat_model.strip():
            raise ValueError("Choose a chat model before applying session settings.")
        if not embedding_model.strip():
            raise ValueError("Choose an embedding model before applying session settings.")
        if chat_model not in available_models:
            raise ValueError(f"Chat model `{chat_model}` is not installed on {discovery.base_url}.")
        if embedding_model not in available_models:
            raise ValueError(f"Embedding model `{embedding_model}` is not installed on {discovery.base_url}.")
        profile = self._config_model_selection().profile
        return ModelSettingsSelection(
            base_url=discovery.base_url,
            chat_model=chat_model,
            embedding_model=embedding_model,
            profile=profile,
        )

    def _suggested_prompts(self, *, principals: list[str]) -> list[PromptSuggestion]:
        seeds = self.runtime.store.list_prompt_seed_chunks(
            permissions_enabled=self.runtime.config.permissions.enabled,
            active_principals=principals,
        )
        if not seeds:
            return default_prompt_suggestions()
        return build_prompt_suggestions(seeds)


def create_web_app(
    *,
    config_path: str | Path | None = None,
    manager: WebRuntimeManager | None = None,
):
    try:
        from starlette.applications import Starlette
        from starlette.datastructures import UploadFile
        from starlette.requests import Request
        from starlette.responses import HTMLResponse, JSONResponse
        from starlette.routing import Mount, Route
        from starlette.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("The web UI extra is not installed. Install `.[web]` to serve the shadcn web app.") from exc

    runtime_manager = manager or WebRuntimeManager(config_path=Path(config_path or "config.yaml").resolve())
    dist_dir = runtime_manager.config_path.parent / "web" / "dist"
    assets_dir = dist_dir / "assets"
    index_path = dist_dir / "index.html"

    async def status_endpoint(_request: Request):
        return JSONResponse(runtime_manager.status_payload())

    async def reload_endpoint(_request: Request):
        return JSONResponse(runtime_manager.reload())

    async def ingest_bridge_health_endpoint(_request: Request):
        return JSONResponse(runtime_manager.bridge_health_payload())

    async def discover_model_settings_endpoint(request: Request):
        payload = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        return JSONResponse(runtime_manager.discover_models(base_url=str(payload.get("base_url", "")).strip() or None))

    async def apply_model_settings_endpoint(request: Request):
        payload = await request.json()
        try:
            result = runtime_manager.apply_model_settings(
                base_url=str(payload.get("base_url", "")).strip(),
                chat_model=str(payload.get("chat_model", "")).strip(),
                embedding_model=str(payload.get("embedding_model", "")).strip(),
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(result)

    async def reindex_model_settings_endpoint(_request: Request):
        try:
            result = runtime_manager.reindex_pending_model_settings()
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(result)

    async def cancel_model_settings_endpoint(_request: Request):
        return JSONResponse(runtime_manager.cancel_pending_model_settings())

    async def ingest_endpoint(request: Request):
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            uploaded_files = [item for item in form.getlist("files") if isinstance(item, UploadFile)]
            if not uploaded_files:
                return JSONResponse({"error": "Choose a folder with at least one supported document."}, status_code=400)
            try:
                result = runtime_manager.ingest_uploaded_files(
                    uploaded_files=uploaded_files,
                    folder_name=str(form.get("folder_name", "")).strip() or None,
                    force_embeddings=str(form.get("force_embeddings", "")).strip().lower() in {"1", "true", "yes", "on"},
                )
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            return JSONResponse(result)

        payload = await request.json() if content_type.startswith("application/json") else {}
        try:
            result = runtime_manager.ingest(
                documents_path=payload.get("documents_path"),
                force_embeddings=bool(payload.get("force_embeddings", False)),
            )
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(result)

    async def ask_endpoint(request: Request):
        payload = await request.json()
        question = str(payload.get("question", "")).strip()
        principals = [str(item).strip() for item in payload.get("principals", []) if str(item).strip()]
        if not question:
            return JSONResponse({"error": "Question is required."}, status_code=400)
        if not principals:
            principals = list(runtime_manager.runtime.config.permissions.active_principals)
        try:
            result = runtime_manager.ask(question=question, principals=principals)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        return JSONResponse(result)

    async def suggested_prompts_endpoint(request: Request):
        payload = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        principals = [str(item).strip() for item in payload.get("principals", []) if str(item).strip()]
        return JSONResponse(runtime_manager.suggested_prompts_payload(principals=principals or None))

    async def spa_entry(_request: Request):
        if not index_path.exists():
            return HTMLResponse(
                (
                    "<h1>Web UI build missing</h1>"
                    "<p>Run <code>cd web && npm install && npm run build</code> "
                    "before starting <code>local-rag serve-web</code>.</p>"
                ),
                status_code=503,
            )
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    routes = [
        Route("/api/status", status_endpoint),
        Route("/api/reload", reload_endpoint, methods=["POST"]),
        Route("/api/ingest/bridge-health", ingest_bridge_health_endpoint),
        Route("/api/model-settings/discover", discover_model_settings_endpoint, methods=["POST"]),
        Route("/api/model-settings/apply", apply_model_settings_endpoint, methods=["POST"]),
        Route("/api/model-settings/reindex", reindex_model_settings_endpoint, methods=["POST"]),
        Route("/api/model-settings/cancel", cancel_model_settings_endpoint, methods=["POST"]),
        Route("/api/ingest", ingest_endpoint, methods=["POST"]),
        Route("/api/suggested-prompts", suggested_prompts_endpoint, methods=["POST"]),
        Route("/api/ask", ask_endpoint, methods=["POST"]),
    ]
    if assets_dir.exists():
        routes.append(Mount("/assets", app=StaticFiles(directory=assets_dir), name="assets"))
    routes.extend(
        [
            Route("/", spa_entry),
            Route("/{path:path}", spa_entry),
        ]
    )
    return Starlette(debug=False, routes=routes)


def run_web_server(config_path: str | Path | None = None) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("The web UI extra is not installed. Install `.[web]` to serve the shadcn web app.") from exc

    resolved_config_path = Path(config_path or "config.yaml").resolve()
    config = load_config(resolved_config_path)
    app = create_web_app(config_path=resolved_config_path)
    uvicorn.run(app, host=config.web.host, port=config.web.port, log_level="info")


def _resolve_documents_override(documents_path: str | Path | None) -> Path | None:
    if documents_path is None:
        return None
    candidate = Path(documents_path).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Documents path does not exist: {candidate}")
    if not candidate.is_dir():
        raise NotADirectoryError(f"Documents path is not a directory: {candidate}")
    return candidate


def _stage_uploaded_files(target_root: Path, uploaded_files: list[object]) -> list[Path]:
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    supported_count = 0
    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "filename", "") or ""
        relative_path = _normalize_relative_upload_path(filename)
        if relative_path is None or not _is_supported_upload_path(relative_path):
            continue
        destination = target_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_handle = getattr(uploaded_file, "file", None)
        if file_handle is None:
            continue
        file_handle.seek(0)
        with destination.open("wb") as output_handle:
            shutil.copyfileobj(file_handle, output_handle)
        written_paths.append(relative_path)
        if not any(str(relative_path).lower().endswith(suffix) for suffix in SIDECAR_SUFFIXES):
            supported_count += 1

    if supported_count == 0:
        shutil.rmtree(target_root, ignore_errors=True)
        raise ValueError("The selected folder did not contain any supported documents (PDF, Markdown, text, or DOCX).")

    return written_paths


def _normalize_relative_upload_path(raw_filename: str) -> Path | None:
    normalized = raw_filename.replace("\\", "/").strip("/")
    if not normalized:
        return None
    relative_path = Path(normalized)
    if relative_path.is_absolute():
        return None
    if any(part in {"", ".", ".."} for part in relative_path.parts):
        return None
    if any(part.startswith(".") for part in relative_path.parts):
        return None
    return relative_path


def _is_supported_upload_path(path: Path) -> bool:
    lowered = str(path).lower()
    if any(lowered.endswith(suffix) for suffix in SIDECAR_SUFFIXES):
        return True
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _infer_folder_label(staged_paths: list[Path]) -> str | None:
    if not staged_paths:
        return None
    first_parts = staged_paths[0].parts
    if not first_parts:
        return None
    return first_parts[0]
