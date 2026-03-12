from tests.conftest import build_test_runtime

from local_agentic_rag.mcp_tools import MCPToolset


def test_mcp_tool_responses(tmp_path) -> None:
    runtime, _config_path, _docs_path = build_test_runtime(tmp_path)
    runtime.ingestion.ingest(prune_missing=False)
    toolset = MCPToolset(runtime)

    search_result = toolset.search_documents("support escalation path", principals=["*"])
    assert search_result["hits"]

    chunk_id = search_result["hits"][0]["chunk_id"]
    context_result = toolset.get_chunk_context(chunk_id)
    assert context_result["chunk_id"] == chunk_id

    answer_result = toolset.ask_with_citations("Who owns support escalations?", principals=["*"], debug=True)
    assert answer_result["citations"]
    assert "trace" in answer_result
