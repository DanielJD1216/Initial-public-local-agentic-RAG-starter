#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PREFERRED_PYTHON="${PYTHON_BIN:-python3.12}"

if command -v "$PREFERRED_PYTHON" >/dev/null 2>&1; then
  PYTHON_BIN="$PREFERRED_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "No compatible Python interpreter was found. Install Python 3.11+ and retry." >&2
  exit 1
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
then
  echo "Python 3.11+ is required. Found: $($PYTHON_BIN --version 2>&1)" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PYTHON_BIN" .venv
  source .venv/bin/activate
  uv pip install --reinstall -r requirements-dev.txt
else
  "$PYTHON_BIN" -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install --force-reinstall -r requirements-dev.txt
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

echo "Bootstrap complete."
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  local-rag bootstrap"
echo "  local-rag ingest"
echo "  local-rag ask \"What is the support escalation path?\" --trace"
