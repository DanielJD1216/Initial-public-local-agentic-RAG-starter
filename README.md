# Local Agentic RAG Starter

An opinionated public starter for building a production-quality local agentic RAG workflow with:

- mandatory metadata and provenance
- hybrid retrieval with SQLite FTS5 + FAISS
- grounded answers with citations
- a transparent plain-Python agent loop
- local Ollama runtime only
- CLI, Streamlit, MCP, and a shadcn-style web UI

This repo is intentionally **not** an architecture chooser. It ships one clean default path so people can clone it, run it, and learn from it without re-deciding the stack every time.

## What V1 Supports

- PDF
- Markdown
- Text
- DOCX

Audio and video are intentionally out of scope for the first release.

## Architecture

- Metadata store: SQLite
- Keyword search: SQLite FTS5 / BM25
- Vector search: FAISS by default
- Embeddings: local Ollama embedding model
- Generation: local Ollama chat model
- Agent behavior: classify -> retrieve -> optionally rewrite once -> retrieve again -> answer only from cited evidence -> abstain when grounding is weak
- Interfaces: CLI, Streamlit, shadcn-style web UI, MCP

## Quickstart

### Recommended bootstrap

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
cd web && npm install && npm run build && cd ..
local-rag bootstrap
local-rag ingest
local-rag ask "What is the standard support first response time?" --trace
```

### Manual fallback

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
local-rag bootstrap
local-rag ingest
```

## Default config

The main config surface is [`config.yaml`](./config.yaml).
There is also a laptop-friendlier [`config.small.yaml`](./config.small.yaml) that uses `qwen3:4b`.
If you want to test a newer small model without editing YAML, use [`config.qwen35.small.yaml`](./config.qwen35.small.yaml) for `qwen3.5:4b`.
For permission-enforcement demos, use [`config.small.permissions.yaml`](./config.small.permissions.yaml).

Important defaults:

- documents path starts at `./sample_corpus`
- model profile defaults to `balanced`
- default chat model is `qwen3:8b`
- default embedding model is `nomic-embed-text`
- default chat calls disable thinking mode and allow a 300 second read timeout
- permissions are stored on every record, but enforcement starts disabled

You can override key settings through environment variables in `.env`.
For a public repo, keep the checked-in configs as safe defaults and put your real document paths or private overrides in an untracked `config.local.yaml`.

## Metadata contract

Every document and chunk must carry provenance and permission metadata.

The ingestion pipeline always stores:

- `doc_id`
- `source_path`
- `content_type`
- `checksum`
- `parser_version`
- `title`
- `ingested_at`
- `access_scope`
- `access_principals`
- chunk index and citation-ready location fields

If required metadata is missing or invalid, ingestion fails closed.

## Optional sidecar metadata

You can override title and permission fields with a sidecar file next to a document:

```yaml
# handbook.pdf.meta.yaml
title: "Customer Support Handbook"
access_scope: "restricted"
access_principals:
  - "owners"
  - "finance"
```

Supported sidecar names:

- `my_doc.pdf.meta.yaml`
- `my_doc.pdf.meta.yml`
- `my_doc.pdf.meta.json`

## Commands

```bash
local-rag bootstrap
local-rag ingest
local-rag reindex
local-rag reindex --force-embeddings
local-rag ask "Who owns support escalations?" --trace
local-rag serve-ui
local-rag serve-web
local-rag serve-mcp
```

## Permission demo

The sample corpus includes one restricted document that only `owners` and `finance` should see.

Test the blocked path:

```bash
local-rag ask "When is the salary adjustment review window planned?" --config config.small.permissions.yaml --trace
```

Test the allowed path:

```bash
local-rag ask "When is the salary adjustment review window planned?" --config config.small.permissions.yaml --principals owners --trace
```

## Streamlit UI

```bash
local-rag serve-ui
```

For the permission-aware demo flow, launch the UI with:

```bash
local-rag serve-ui --config config.small.permissions.yaml
```

The UI uses the same shared runtime as the CLI and now includes:

- corpus overview metrics
- sample demo questions
- a principal selector with `Config default`, `Public only`, and `Custom principals`
- separate panes for citations, retrieved chunks, and the structured trace
- a reload control so the app picks up fresh ingest or reindex runs without restarting Streamlit

## Web UI

Build the frontend once:

```bash
cd web
npm install
npm run build
cd ..
```

Then launch the web app:

```bash
local-rag serve-web --config config.small.permissions.yaml
```

The web UI is a shadcn-style React shell with:

- a Codex-like three-pane layout
- live runtime and corpus metrics
- document-path switching and indexing from the browser
- permission-aware principal controls
- answer, citations, chunks, and trace inspection in separate panels

To try the newer small model in the web UI later:

```bash
ollama pull qwen3.5:4b
local-rag serve-web --config config.qwen35.small.yaml
```

Suggested demo flow:

1. Start on `sample_corpus`.
2. Run `Permission check` with the default principals to show the abstain path.
3. Switch to `Custom`, enable `owners`, and ask again to show the restricted citation.
4. Paste a real folder path into `Documents folder`, click `Index folder`, and rerun a simple question.

## MCP tools

```bash
local-rag serve-mcp
```

Built-in tools:

- `search_documents`
- `get_chunk_context`
- `ask_with_citations`

## Guided Codex onboarding

The repo includes a guided Codex prompt at [`prompts/guided_codex_prompt.md`](./prompts/guided_codex_prompt.md).

Use that prompt when you want Codex to walk you through:

- environment checks
- document path setup
- indexing
- test queries
- Streamlit
- MCP

The prompt is intentionally opinionated around the shipped architecture so tutorial viewers get the same experience.

## Sample corpus

The `sample_corpus/` folder is synthetic and safe to publish. It gives you:

- a simple support-policy lookup case
- a cross-document multi-hop case
- a vague support question that benefits from query rewriting
- a restricted document for permission-filter tests

## Running tests

```bash
source .venv/bin/activate
pytest
```

Current repo test coverage includes:

- parser behavior
- chunking behavior
- metadata validation
- permission filtering
- citation formatting
- ingestion and reindex flow
- MCP tool responses
- agent grounding checks

## Troubleshooting

### Ollama is missing

Install Ollama first, then run:

```bash
local-rag bootstrap
```

### The configured models are missing

Pull them with Ollama:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

Or switch to the `small` profile in `config.yaml`.

### The chat step times out

This usually means the selected Ollama chat model is too slow for the machine or is spending too long in reasoning mode.

Try these in order:

1. keep `disable_thinking: true` in `config.yaml`
2. increase `models.request_timeout_seconds`
3. switch to a smaller chat model

Example:

```yaml
models:
  profile: "small"
  chat_model: "qwen3:4b"
  embedding_model: "nomic-embed-text"
  request_timeout_seconds: 300
  disable_thinking: true
```

Then pull the smaller model and rebuild:

```bash
ollama pull qwen3:4b
ollama pull qwen3.5:4b
local-rag ask "What is the support escalation path?" --trace
```

### FAISS is unavailable in your environment

Set this in `config.yaml` for a portable fallback:

```yaml
retrieval:
  vector_backend: "numpy"
```

That fallback is slower and mainly intended for testing or unsupported environments.

### I changed the embedding model

Rebuild the stored embeddings:

```bash
local-rag reindex --force-embeddings
```

## Runtime promise

After setup, the app is designed to run locally through Ollama with no ongoing API calls. The automated tests in this repo run against fake local model clients so they stay deterministic and do not require network access.
