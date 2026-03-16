from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .bootstrap import run_bootstrap_checks
from .config import load_config
from .mcp_server import run_mcp_server
from .service import build_runtime
from .web_server import run_web_server

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _split_principals(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


@app.command()
def bootstrap(
    config: str = typer.Option("config.yaml", help="Path to the main config file."),
    pull_missing: bool = typer.Option(False, help="Attempt to pull missing Ollama models."),
) -> None:
    app_config = load_config(config)
    report = run_bootstrap_checks(app_config, pull_missing=pull_missing)
    table = Table(title="Bootstrap checks")
    table.add_column("Check")
    table.add_column("Status")
    table.add_row("Python >= 3.11", "ok" if report.python_ok else "missing")
    table.add_row("Ollama installed", "ok" if report.ollama_installed else "missing")
    table.add_row(f"Chat model `{app_config.models.chat_model}`", "ok" if report.chat_model_available else "missing")
    table.add_row(
        f"Embedding model `{app_config.models.embedding_model}`",
        "ok" if report.embedding_model_available else "missing",
    )
    if report.ingest_mode == "bridge":
        table.add_row(
            f"Ingest bridge `{app_config.ingest.bridge_model}`",
            "ok" if report.bridge_reachable else "missing",
        )
    else:
        table.add_row("Ingest strategy", "local")
    table.add_row(f"Vector backend `{app_config.retrieval.vector_backend}`", "ok" if report.vector_backend_ready else "missing")
    console.print(table)
    if report.notes:
        console.print(Panel("\n".join(report.notes), title="Notes"))


@app.command()
def ingest(
    config: str = typer.Option("config.yaml", help="Path to the main config file."),
) -> None:
    runtime = build_runtime(config_path=config)
    report = runtime.ingestion.ingest(prune_missing=False)
    _print_ingestion_report(
        report,
        runtime.config.ingest.mode,
        runtime.config.ingest.bridge_model,
        runtime.agent.runtime_status().to_dict(),
    )


@app.command()
def reindex(
    config: str = typer.Option("config.yaml", help="Path to the main config file."),
    force_embeddings: bool = typer.Option(False, help="Re-embed unchanged files with the current embedding model."),
) -> None:
    runtime = build_runtime(config_path=config)
    report = runtime.ingestion.ingest(prune_missing=True, force_embeddings=force_embeddings)
    _print_ingestion_report(
        report,
        runtime.config.ingest.mode,
        runtime.config.ingest.bridge_model,
        runtime.agent.runtime_status().to_dict(),
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask against the indexed corpus."),
    config: str = typer.Option("config.yaml", help="Path to the main config file."),
    principals: str = typer.Option("", help="Comma-separated principals for permission-aware retrieval."),
    trace: bool = typer.Option(False, help="Print the structured retrieval trace."),
) -> None:
    runtime = build_runtime(config_path=config)
    active_principals = _split_principals(principals) or list(runtime.config.permissions.active_principals)
    try:
        result = runtime.agent.answer(question, active_principals=active_principals)
    except RuntimeError as exc:
        console.print(Panel(str(exc), title="Generation failed"))
        raise SystemExit(1)
    console.print(Panel(result.answer, title="Answer"))
    console.print(
        Panel(
            f"status={result.status}\n"
            f"task_mode={result.task_mode}\n"
            f"active_mode={result.active_mode}\n"
            f"failure_reason={result.failure_reason or 'none'}",
            title="Answer status",
        )
    )
    if result.citations:
        table = Table(title="Citations")
        table.add_column("Chunk")
        table.add_column("Citation")
        for citation in result.citations:
            table.add_row(citation.chunk_id, citation.citation)
        console.print(table)
    else:
        console.print("[yellow]No citations available.[/yellow]")
    if trace:
        console.print(Panel(json.dumps(result.trace.to_dict(), indent=2), title="Trace"))


@app.command("serve-ui")
def serve_ui(
    config: str = typer.Option("config.yaml", help="Path to the main config file."),
) -> None:
    app_config = load_config(config)
    script_path = app_config.root_dir / "streamlit_app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
        "--server.address",
        app_config.ui.host,
        "--server.port",
        str(app_config.ui.port),
        "--",
        "--config",
        str(Path(config).resolve()),
    ]
    raise SystemExit(subprocess.call(command))


@app.command("serve-mcp")
def serve_mcp(
    config: str = typer.Option("config.yaml", help="Path to the main config file."),
) -> None:
    run_mcp_server(config)


@app.command("serve-web")
def serve_web(
    config: str = typer.Option("config.yaml", help="Path to the main config file."),
) -> None:
    run_web_server(config)


def _print_ingestion_report(report, ingest_mode: str, bridge_model: str, agent_status: dict[str, object]) -> None:
    if ingest_mode == "bridge":
        console.print(Panel(f"Bridge enrichment enabled via `{bridge_model}`.", title="Ingest mode"))
    else:
        console.print(Panel("Local heuristic ingest is active.", title="Ingest mode"))
    console.print(
        Panel(
            f"configured={agent_status['configured_mode']} | active={agent_status['active_mode']}"
            + (
                f"\n{agent_status['downgrade_reason']}"
                if agent_status.get("downgrade_reason")
                else ""
            ),
            title="Agent mode",
        )
    )
    table = Table(title="Ingestion report")
    table.add_column("Category")
    table.add_column("Count")
    table.add_row("Processed", str(len(report.processed)))
    table.add_row("Skipped", str(len(report.skipped)))
    table.add_row("Deleted", str(len(report.deleted)))
    table.add_row("Errors", str(len(report.errors)))
    console.print(table)
    if report.errors:
        console.print(Panel(json.dumps(report.errors, indent=2), title="Errors"))


def main() -> None:
    app()
