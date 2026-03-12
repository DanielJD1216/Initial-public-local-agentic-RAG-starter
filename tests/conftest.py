from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import yaml

from local_agentic_rag.service import build_runtime


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "do",
    "for",
    "how",
    "i",
    "is",
    "me",
    "of",
    "should",
    "tell",
    "the",
    "to",
    "what",
    "who",
}


class FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * 32
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = sum(ord(char) for char in token)
                vector[digest % 32] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class FakeChatClient:
    def chat_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, object]:
        if "rewrite weak retrieval queries" in system_prompt:
            question = _extract_question(user_prompt)
            if "support" in question.lower():
                return {
                    "rewritten_query": "support coverage escalation path customer communication",
                    "reason": "Expand the topic into the specific support concepts stored in the corpus.",
                }
            return {"rewritten_query": question, "reason": "The original query is already specific enough."}

        question = _extract_question(user_prompt)
        evidence = _extract_evidence(user_prompt)
        selected = _select_supporting_evidence(question, evidence)
        if not selected:
            return {"answer": "", "grounded": False, "citations": []}
        answer = " ".join(item["sentence"] for item in selected)
        citations = [
            {"chunk_id": item["chunk_id"], "reason": "Source sentence used in the grounded answer."}
            for item in selected
        ]
        return {"answer": answer, "grounded": True, "citations": citations}


def _extract_question(user_prompt: str) -> str:
    match = re.search(r"Question:\n(.*?)\n\n", user_prompt, re.S)
    return match.group(1).strip() if match else user_prompt.strip()


def _extract_evidence(user_prompt: str) -> list[dict[str, str]]:
    match = re.search(r"Evidence:\n(.*?)\n\nAnswer", user_prompt, re.S)
    if not match:
        return []
    return json.loads(match.group(1))


def _select_supporting_evidence(question: str, evidence: list[dict[str, str]]) -> list[dict[str, str]]:
    keywords = [token for token in re.findall(r"[a-z0-9]+", question.lower()) if token not in STOPWORDS]
    question_lower = question.lower()
    selected: list[dict[str, str]] = []
    seen_chunk_ids: set[str] = set()
    for entry in evidence:
        sentences = re.split(r"(?<=[.!?])\s+", entry["text"])
        best_sentence = ""
        best_score = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for keyword in keywords if keyword in sentence_lower)
            if "response" in question_lower and "response" in sentence_lower:
                score += 2
            if any(token in question_lower for token in ["when", "time", "deadline", "due"]) and re.search(r"\d", sentence_lower):
                score += 2
            if score > best_score:
                best_sentence = sentence.strip()
                best_score = score
        if best_score > 0 and entry["chunk_id"] not in seen_chunk_ids:
            selected.append({"chunk_id": entry["chunk_id"], "sentence": best_sentence, "score": best_score})
            seen_chunk_ids.add(entry["chunk_id"])
    selected.sort(key=lambda item: item["score"], reverse=True)
    return [{"chunk_id": item["chunk_id"], "sentence": item["sentence"]} for item in selected[:3]]


def write_test_config(tmp_path: Path, docs_path: Path, *, permissions_enabled: bool = True) -> Path:
    config_path = tmp_path / "config.yaml"
    payload = {
        "version": 1,
        "project_name": "Test Local Agentic RAG",
        "paths": {
            "documents": str(docs_path),
            "database": str(tmp_path / ".rag_local" / "rag.sqlite3"),
            "vector_index": str(tmp_path / ".rag_local" / "vector.index"),
            "vector_metadata": str(tmp_path / ".rag_local" / "vector.meta.json"),
            "cache_dir": str(tmp_path / ".rag_local" / "cache"),
        },
        "models": {
            "profile": "small",
            "chat_model": "fake-chat",
            "embedding_model": "fake-embed",
            "base_url": "http://127.0.0.1:11434",
            "request_timeout_seconds": 30,
        },
        "retrieval": {
            "top_k": 4,
            "keyword_k": 6,
            "vector_k": 6,
            "max_chunk_tokens": 120,
            "overlap_tokens": 20,
            "min_evidence_score": 0.34,
            "rrf_k": 60,
            "vector_backend": "numpy",
        },
        "permissions": {
            "enabled": permissions_enabled,
            "default_access_scope": "public",
            "default_access_principals": ["*"],
            "active_principals": ["*"],
        },
        "ui": {"host": "127.0.0.1", "port": 8501},
        "web": {"host": "127.0.0.1", "port": 3000},
        "mcp": {"host": "127.0.0.1", "port": 8000},
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def copy_sample_corpus(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "sample_corpus"
    destination = tmp_path / "sample_corpus"
    shutil.copytree(source, destination)
    return destination


def build_test_runtime(tmp_path: Path, *, permissions_enabled: bool = True):
    docs_path = copy_sample_corpus(tmp_path)
    config_path = write_test_config(tmp_path, docs_path, permissions_enabled=permissions_enabled)
    runtime = build_runtime(
        config_path=config_path,
        embedding_client=FakeEmbeddingClient(),
        chat_client=FakeChatClient(),
    )
    return runtime, config_path, docs_path
