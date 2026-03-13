from __future__ import annotations

import json
import re
from typing import Any, Protocol

import httpx


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class ChatClient(Protocol):
    def chat_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        ...


class OllamaEmbeddingClient:
    def __init__(self, *, base_url: str, model: str, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": self.model,
            "input": texts,
        }
        try:
            with httpx.Client(timeout=_timeout(self.timeout_seconds)) as client:
                response = client.post(f"{self.base_url}/api/embed", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama embedding request to `{self.model}` timed out after {self.timeout_seconds}s. "
                "Increase models.request_timeout_seconds in config.yaml if this machine is slow."
            ) from exc
        return data["embeddings"]


class OllamaChatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: int = 120,
        disable_thinking: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.disable_thinking = disable_thinking

    def chat_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0.2},
        }
        if self.disable_thinking:
            payload["think"] = False
        try:
            with httpx.Client(timeout=_timeout(self.timeout_seconds)) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Ollama chat request to `{self.model}` timed out after {self.timeout_seconds}s. "
                "Try a smaller chat model, disable thinking, or increase models.request_timeout_seconds in config.yaml."
            ) from exc
        content = str(data.get("message", {}).get("content") or data.get("response") or "").strip()
        return _parse_json_response(content, model=self.model)


def _timeout(total_seconds: int) -> httpx.Timeout:
    return httpx.Timeout(connect=10.0, read=float(total_seconds), write=30.0, pool=10.0)


def _parse_json_response(content: str, *, model: str) -> dict[str, Any]:
    normalized = content.strip()
    if not normalized:
        raise RuntimeError(f"Ollama model `{model}` returned an empty response when JSON was required.")

    candidates: list[str] = [normalized]

    # Models sometimes wrap the JSON in fences or think tags even when format=json is requested.
    without_think = re.sub(r"<think>.*?</think>", "", normalized, flags=re.S | re.I).strip()
    if without_think and without_think != normalized:
        candidates.append(without_think)

    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", without_think, flags=re.S | re.I).strip()
    if fenced and fenced not in candidates:
        candidates.append(fenced)

    extracted = _extract_json_object(fenced)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            parsed, trailing_index = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if candidate[trailing_index:].strip():
            continue
        if isinstance(parsed, dict):
            return parsed

    snippet = normalized[:240].replace("\n", "\\n")
    raise RuntimeError(
        f"Ollama model `{model}` returned content that was not valid JSON. "
        f"Response started with: {snippet}"
    )


def _extract_json_object(content: str) -> str | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", content):
        try:
            parsed, trailing_index = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return content[match.start() : match.start() + trailing_index]
    return None
