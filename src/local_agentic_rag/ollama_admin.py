from __future__ import annotations

from dataclasses import dataclass

import httpx


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


@dataclass(slots=True)
class OllamaDiscoveryResult:
    base_url: str
    reachable: bool
    models: list[str]
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "base_url": self.base_url,
            "reachable": self.reachable,
            "models": self.models,
            "error": self.error,
        }


def normalize_ollama_base_url(raw_base_url: str | None) -> str:
    value = (raw_base_url or "").strip()
    if not value:
        return DEFAULT_OLLAMA_BASE_URL
    if "://" not in value:
        value = f"http://{value}"
    return value.rstrip("/")


def resolve_ollama_model_name(requested_model: str, available_models: list[str]) -> str:
    normalized_requested = requested_model.strip()
    if not normalized_requested:
        return normalized_requested
    if normalized_requested in available_models:
        return normalized_requested
    if normalized_requested.endswith(":latest"):
        base_name = normalized_requested[: -len(":latest")]
        if base_name in available_models:
            return base_name
        return normalized_requested
    latest_name = f"{normalized_requested}:latest"
    if latest_name in available_models:
        return latest_name
    return normalized_requested


def ollama_models_equivalent(left: str, right: str) -> bool:
    left_value = left.strip()
    right_value = right.strip()
    if left_value == right_value:
        return True
    return resolve_ollama_model_name(left_value, [right_value]) == right_value or resolve_ollama_model_name(
        right_value,
        [left_value],
    ) == left_value


def discover_ollama_models(base_url: str | None, *, timeout_seconds: int = 5) -> OllamaDiscoveryResult:
    normalized_base_url = normalize_ollama_base_url(base_url)
    try:
        with httpx.Client(timeout=_timeout(timeout_seconds)) as client:
            response = client.get(f"{normalized_base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        return OllamaDiscoveryResult(
            base_url=normalized_base_url,
            reachable=False,
            models=[],
            error=f"Ollama responded with HTTP {exc.response.status_code} at {normalized_base_url}.",
        )
    except httpx.HTTPError as exc:
        return OllamaDiscoveryResult(
            base_url=normalized_base_url,
            reachable=False,
            models=[],
            error=f"Could not reach Ollama at {normalized_base_url}: {exc}.",
        )
    except ValueError:
        return OllamaDiscoveryResult(
            base_url=normalized_base_url,
            reachable=False,
            models=[],
            error=f"Ollama returned an unreadable response at {normalized_base_url}.",
        )

    raw_models = payload.get("models", [])
    models = sorted(
        {
            str(item.get("name", "")).strip()
            for item in raw_models
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        }
    )
    if models:
        return OllamaDiscoveryResult(
            base_url=normalized_base_url,
            reachable=True,
            models=models,
        )
    return OllamaDiscoveryResult(
        base_url=normalized_base_url,
        reachable=True,
        models=[],
        error=f"Connected to Ollama at {normalized_base_url}, but no models are installed yet.",
    )


def _timeout(total_seconds: int) -> httpx.Timeout:
    connect_timeout = min(float(total_seconds), 5.0)
    return httpx.Timeout(connect=connect_timeout, read=float(total_seconds), write=10.0, pool=5.0)
