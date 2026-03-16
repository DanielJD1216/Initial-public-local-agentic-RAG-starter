from tests.conftest import build_test_runtime

from local_agentic_rag.mcp_tools import MCPToolset


def test_mcp_tool_responses(tmp_path) -> None:
    runtime, _config_path, docs_path = build_test_runtime(tmp_path)
    toolset = MCPToolset(runtime)

    ingest_result = toolset.ingest_path(str(docs_path))
    assert len(ingest_result["report"]["processed"]) == 4
    assert ingest_result["status"]["ingest"]["corpus"]["mode"] == "local"

    status_result = toolset.get_runtime_status()
    assert status_result["local_models"]["chat_model"] == "fake-chat"
    assert status_result["ingest"]["mode"] == "local"
    assert status_result["agent"]["configured_mode"] == "middleweight"

    search_result = toolset.search_documents("support escalation path", principals=["*"])
    assert search_result["hits"]

    chunk_id = search_result["hits"][0]["chunk_id"]
    context_result = toolset.get_chunk_context(chunk_id)
    assert context_result["chunk_id"] == chunk_id

    answer_result = toolset.ask_with_citations("Who owns support escalations?", principals=["*"], debug=True)
    assert answer_result["citations"]
    assert answer_result["task_mode"] in {"simple_lookup", "ownership_policy", "cross_document_analysis"}
    assert "trace" in answer_result
