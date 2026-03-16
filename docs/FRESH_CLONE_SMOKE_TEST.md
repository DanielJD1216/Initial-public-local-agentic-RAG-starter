# Fresh Clone Smoke Test

Use this checklist to verify that a brand-new clone behaves like the reference repo.

## 1. Bootstrap the environment

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
cd web && npm install && npm run build && cd ..
```

## 2. Verify local model prerequisites

```bash
local-rag bootstrap
```

Expected:

- Ollama is installed
- the configured chat model is available
- the configured embedding model is available
- the selected vector backend is ready

## 3. Ingest the sample corpus

```bash
local-rag ingest
```

Expected:

- no ingest errors
- a local SQLite store and vector index appear under `./.rag_local/`

## 4. Run a grounded CLI query

```bash
local-rag ask "What is the standard support first response time?" --trace
```

Expected:

- answer contains `4 business hours`
- answer includes citations

## 5. Run a permission check

Blocked path:

```bash
local-rag ask "When is the salary adjustment review window planned?" --trace
```

Allowed path:

```bash
local-rag ask "When is the salary adjustment review window planned?" --principals owners --trace
```

Expected:

- blocked path returns a restricted or abstained result without leaking restricted text
- allowed path returns the grounded answer

## 6. Launch the web UI

```bash
local-rag serve-web
```

Open `http://127.0.0.1:3000`.

Expected:

- runtime status loads successfully
- sample corpus metrics appear
- question flow works
- `Public` blocks restricted documents
- `Custom > owners` unlocks the restricted demo query

## 7. If port 3000 is already in use

```bash
lsof -n -P -iTCP:3000 -sTCP:LISTEN
```

Stop the stale process, then restart from this repo:

```bash
source .venv/bin/activate
local-rag serve-web
```

## 8. Optional browser folder check

- Use `Choose folder` in the web UI
- Select a local test corpus
- Re-run a simple grounded question

Expected:

- corpus counts change
- the retrieved citations reflect the new folder
