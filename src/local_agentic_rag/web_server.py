from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable

from .clients import ChatClient, EmbeddingClient
from .config import AppConfig, load_config
from .service import AppRuntime, build_runtime_from_config


@dataclass(slots=True)
class WebRuntimeManager:
    config_path: Path
    runtime_builder: Callable[..., AppRuntime] = build_runtime_from_config
    config_loader: Callable[[str | Path | None], AppConfig] = load_config
    embedding_client: EmbeddingClient | None = None
    chat_client: ChatClient | None = None
    _lock: Lock = field(init=False, repr=False)
    _runtime: AppRuntime = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = Lock()
        self._runtime = self._build_runtime()

    @property
    def runtime(self) -> AppRuntime:
        return self._runtime

    def status_payload(self) -> dict[str, object]:
        corpus = self.runtime.store.get_corpus_summary()
        return {
            "project_name": self.runtime.config.project_name,
            "config_path": str(self.config_path),
            "documents_path": str(self.runtime.config.paths.documents),
            "models": {
                "profile": self.runtime.config.models.profile,
                "chat_model": self.runtime.config.models.chat_model,
                "embedding_model": self.runtime.config.models.embedding_model,
            },
            "permissions_enabled": self.runtime.config.permissions.enabled,
            "default_principals": list(self.runtime.config.permissions.active_principals),
            "corpus": corpus.to_dict(),
        }

    def reload(self) -> dict[str, object]:
        with self._lock:
            self._runtime = self._build_runtime(documents_path=self.runtime.config.paths.documents)
            return self.status_payload()

    def ingest(
        self,
        *,
        documents_path: str | None = None,
        force_embeddings: bool = False,
    ) -> dict[str, object]:
        with self._lock:
            runtime = self._build_runtime(documents_path=_resolve_documents_override(documents_path))
            report = runtime.ingestion.ingest(prune_missing=True, force_embeddings=force_embeddings)
            self._runtime = runtime
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

    def _build_runtime(self, documents_path: str | Path | None = None) -> AppRuntime:
        config = self.config_loader(self.config_path)
        if documents_path is not None:
            candidate = _resolve_documents_override(documents_path)
            config.paths.documents = candidate
        return self.runtime_builder(
            config=config,
            embedding_client=self.embedding_client,
            chat_client=self.chat_client,
        )


def create_web_app(
    *,
    config_path: str | Path | None = None,
    manager: WebRuntimeManager | None = None,
):
    try:
        from starlette.applications import Starlette
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

    async def ingest_endpoint(request: Request):
        payload = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
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
        Route("/api/ingest", ingest_endpoint, methods=["POST"]),
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
