from __future__ import annotations

import os
from pathlib import Path

from local_agentic_rag.service import build_runtime


DEMO_QUESTIONS = [
    ("Simple lookup", "What is the standard support first response time?"),
    (
        "Cross-document",
        "Who owns support escalations and when is the postmortem due for a customer-facing incident?",
    ),
    ("Permission check", "When is the salary adjustment review window planned?"),
]


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Streamlit is not installed. Install the ui extra to use the web app.") from exc

    st.set_page_config(page_title="Local Agentic RAG", layout="wide")
    config_path = _resolve_config_path()

    @st.cache_resource(show_spinner=False)
    def _load_runtime(active_config_path: str):
        return build_runtime(config_path=active_config_path)

    runtime = _load_runtime(config_path)
    corpus = runtime.store.get_corpus_summary()

    st.session_state.setdefault("question_input", "")
    st.title(runtime.config.project_name)
    st.caption("Local hybrid retrieval, citations, permission-aware demos, and transparent trace inspection.")

    _render_overview(st, runtime, corpus)

    with st.sidebar:
        debug, principals = _render_sidebar(st, runtime, corpus, config_path, _load_runtime)

    if corpus.document_count == 0:
        st.warning("No indexed documents were found. Run `local-rag ingest`, then use Reload runtime.")

    _render_demo_questions(st)

    with st.form("ask-form", clear_on_submit=False):
        question = st.text_area(
            "Ask a grounded question",
            key="question_input",
            placeholder="Example: What is the support escalation policy and who owns it?",
            height=140,
        )
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("Enter a question before running the demo.")
        else:
            try:
                result = runtime.agent.answer(question.strip(), active_principals=principals)
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                st.session_state["latest_result"] = result
                st.session_state["latest_principals"] = principals

    result = st.session_state.get("latest_result")
    if result is not None:
        _render_result(
            st,
            result=result,
            active_principals=st.session_state.get("latest_principals", principals),
            debug=debug,
        )


def _render_overview(st, runtime, corpus) -> None:
    metrics = st.columns(4)
    metrics[0].metric("Documents", corpus.document_count)
    metrics[1].metric("Chunks", corpus.chunk_count)
    metrics[2].metric("Restricted docs", corpus.restricted_document_count)
    metrics[3].metric("Permissions", "On" if runtime.config.permissions.enabled else "Off")

    st.caption(
        " | ".join(
            [
                f"Profile: {runtime.config.models.profile}",
                f"Chat: {runtime.config.models.chat_model}",
                f"Embeddings: {runtime.config.models.embedding_model}",
                f"Vector backend: {runtime.config.retrieval.vector_backend}",
            ]
        )
    )


def _render_sidebar(st, runtime, corpus, config_path: str, runtime_loader) -> tuple[bool, list[str]]:
    st.subheader("Runtime")
    st.write(f"Config: `{config_path}`")
    st.write(f"Documents: `{runtime.config.paths.documents}`")
    if st.button("Reload runtime", use_container_width=True):
        runtime_loader.clear()
        st.session_state.pop("latest_result", None)
        st.rerun()

    debug = st.toggle("Show raw retrieval trace", value=False)

    st.divider()
    st.subheader("Access view")
    if not runtime.config.permissions.enabled:
        st.info("Permission enforcement is disabled in this config. All indexed chunks are searchable.")
        return debug, list(runtime.config.permissions.active_principals)

    mode = st.radio(
        "Choose the principal set",
        options=["Config default", "Public only", "Custom principals"],
        index=0,
    )

    if mode == "Config default":
        principals = list(runtime.config.permissions.active_principals)
    elif mode == "Public only":
        principals = []
    else:
        default_principals = [item for item in runtime.config.permissions.active_principals if item != "*"]
        selected = st.multiselect(
            "Principals",
            options=corpus.principals,
            default=default_principals,
            help="These values are discovered from indexed document metadata.",
        )
        extra_principals = st.text_input(
            "Extra principals",
            placeholder="owners, finance",
            help="Optional comma-separated principals for corpora that use values not shown above.",
        )
        principals = sorted({*selected, *_parse_principals(extra_principals)})

    st.caption(f"Effective principals: {_format_principals(principals)}")
    if corpus.principals:
        st.write("Indexed principals")
        st.write(", ".join(corpus.principals))
    else:
        st.write("No restricted principals were discovered in the indexed corpus yet.")
    return debug, principals


def _render_demo_questions(st) -> None:
    st.subheader("Demo flow")
    st.caption("Start with a sample question, then switch the access view in the sidebar to demonstrate permission changes.")
    columns = st.columns(len(DEMO_QUESTIONS))
    for index, (label, query) in enumerate(DEMO_QUESTIONS):
        if columns[index].button(label, use_container_width=True, key=f"demo-question-{index}"):
            st.session_state["question_input"] = query


def _render_result(st, *, result, active_principals: list[str], debug: bool) -> None:
    st.divider()
    st.subheader("Answer")
    if result.grounded:
        st.success("Grounded answer with citations.")
    else:
        st.warning("The assistant withheld a definitive answer because grounding was insufficient.")
    st.caption(f"Answered with principals: {_format_principals(active_principals)}")
    st.write(result.answer)

    tabs = st.tabs(["Citations", "Retrieved chunks", "Trace"])

    with tabs[0]:
        if result.citations:
            for citation in result.citations:
                st.markdown(f"**{citation.citation}**")
                st.caption(f"Chunk: `{citation.chunk_id}`")
                if citation.reason:
                    st.write(citation.reason)
        else:
            st.info("No citations are shown for this answer.")

    with tabs[1]:
        if not result.retrieved_chunks:
            st.info("No chunks were retrieved.")
        cited_ids = {citation.chunk_id for citation in result.citations}
        for chunk in result.retrieved_chunks:
            label = "Cited" if chunk.chunk_id in cited_ids else "Retrieved"
            with st.expander(f"{label} · {chunk.title} · {chunk.location_label}"):
                st.caption(
                    f"chunk_id={chunk.chunk_id} | access_scope={chunk.access_scope} | principals={', '.join(chunk.access_principals)}"
                )
                st.write(chunk.text)

    with tabs[2]:
        best_attempt = max(result.trace.attempts, key=lambda attempt: attempt.evidence_score, default=None)
        trace_metrics = st.columns(4)
        trace_metrics[0].metric("Attempts", len(result.trace.attempts))
        trace_metrics[1].metric("Query type", result.trace.query_type)
        trace_metrics[2].metric(
            "Best evidence",
            f"{best_attempt.evidence_score:.3f}" if best_attempt is not None else "n/a",
        )
        trace_metrics[3].metric("Rewrite used", "Yes" if result.trace.rewritten_query else "No")
        if result.trace.rewritten_query:
            st.write(f"Rewritten query: `{result.trace.rewritten_query}`")
        if result.trace.verification_notes:
            for note in result.trace.verification_notes:
                st.caption(note)
        if debug:
            st.json(result.trace.to_dict(), expanded=False)
        else:
            st.info("Turn on 'Show raw retrieval trace' in the sidebar to inspect the full structured trace.")


def _parse_principals(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _format_principals(principals: list[str]) -> str:
    return ", ".join(principals) if principals else "public-only"


def _resolve_config_path() -> str:
    args = [value for value in os.sys.argv[1:] if value != "--"]
    if "--config" in args:
        index = args.index("--config")
        return str(Path(args[index + 1]).resolve())
    return str(Path(os.environ.get("RAG_CONFIG_PATH", "config.yaml")).resolve())
