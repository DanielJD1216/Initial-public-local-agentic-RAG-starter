# V1 Scope

## Goal

Ship a cloneable local RAG starter that can ingest a practical document corpus, answer with grounded citations, enforce document-level permissions, and expose the same runtime through CLI, Streamlit, MCP, and a browser UI.

## In Scope

- Local document ingestion for `PDF`, `Markdown`, `TXT`, and `DOCX`
- Local embeddings and local chat generation through Ollama
- SQLite metadata storage plus hybrid keyword/vector retrieval
- Grounded answers with citations and explicit abstain behavior
- Permission-aware retrieval with restricted-document filtering
- Auto-restriction from configured sensitivity markers such as `confidential`
- Web UI support for:
  - browser folder picking
  - direct path indexing
  - session-only local model switching
  - access-view simulation (`Config`, `Public`, `Custom`)
  - explicit restricted-access responses
- CLI, Streamlit, and MCP access to the same shared runtime
- Optional localhost ingest bridge for ingest-time enrichment only

## Out of Scope

- Vision understanding of CAD drawings, image-heavy PDFs, or scanned sheets
- Audio or video ingestion
- Cloud-hosted answer generation as the default runtime path
- Enterprise auth, SSO, user management, or audited access logs
- Long-horizon autonomous planning agents
- Multi-tenant deployment or hosted SaaS operations
- Raw CAD entity parsing from DWG/DXF or geometry-aware drawing intelligence

## V1 Acceptance Gates

- Fresh clone can bootstrap without hand-editing repo code
- Sample corpus ingests successfully
- A simple grounded answer returns with citations
- A restricted sample query is blocked in `Public`
- The same restricted query is allowed in `Custom` with the correct principal
- The web UI can switch folders and re-index from the browser
- The web UI can switch local models in-session without editing YAML
- `pytest` passes
- `npm run build` passes for the web app
- README and shipped config defaults do not contradict each other

## Release Notes

This scope defines a lightweight, production-friendly local RAG starter. It does not claim to be a full engineering-drawing intelligence system or a middle-weight autonomous agent platform.
