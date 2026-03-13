from __future__ import annotations

from .mcp_tools import MCPToolset
from .service import build_runtime


def run_mcp_server(config_path: str | None = None) -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("The MCP extra is not installed. Install `.[mcp]` to serve MCP tools.") from exc

    runtime = build_runtime(config_path=config_path)
    tools = MCPToolset(runtime)
    server = FastMCP("local-agentic-rag")

    @server.tool()
    def get_runtime_status() -> dict[str, object]:
        return tools.get_runtime_status()

    @server.tool()
    def ingest_path(
        path: str,
        prune_missing: bool = True,
        force_embeddings: bool = False,
    ) -> dict[str, object]:
        return tools.ingest_path(path, prune_missing=prune_missing, force_embeddings=force_embeddings)

    @server.tool()
    def search_documents(query: str, top_k: int = 5, principals: list[str] | None = None) -> dict[str, object]:
        return tools.search_documents(query, top_k=top_k, principals=principals)

    @server.tool()
    def get_chunk_context(chunk_id: str) -> dict[str, object]:
        return tools.get_chunk_context(chunk_id)

    @server.tool()
    def ask_with_citations(
        question: str,
        principals: list[str] | None = None,
        debug: bool = False,
    ) -> dict[str, object]:
        return tools.ask_with_citations(question, principals=principals, debug=debug)

    server.run()
