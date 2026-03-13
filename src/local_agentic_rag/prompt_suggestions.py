from __future__ import annotations

import re

from .models import ChunkRecord, PromptSuggestion

DEFAULT_PROMPT_SUGGESTIONS = [
    PromptSuggestion(label="Simple lookup", prompt="What is the standard support first response time?"),
    PromptSuggestion(
        label="Cross-document",
        prompt="Who owns support escalations and when is the postmortem due for a customer-facing incident?",
    ),
    PromptSuggestion(label="Permission check", prompt="When is the salary adjustment review window planned?"),
]


def default_prompt_suggestions() -> list[PromptSuggestion]:
    return [PromptSuggestion(label=item.label, prompt=item.prompt) for item in DEFAULT_PROMPT_SUGGESTIONS]


def build_prompt_suggestions(chunks: list[ChunkRecord], *, max_prompts: int = 4) -> list[PromptSuggestion]:
    suggestions: list[PromptSuggestion] = []
    seen_prompts: set[str] = set()
    seen_labels: set[str] = set()

    for chunk in chunks:
        suggestion = _suggestion_from_chunk(chunk)
        if suggestion is None or suggestion.prompt in seen_prompts:
            continue
        suggestion = _dedupe_label(suggestion, seen_labels)
        suggestions.append(suggestion)
        seen_prompts.add(suggestion.prompt)
        seen_labels.add(suggestion.label)
        if len(suggestions) >= max_prompts:
            return suggestions

    for chunk in chunks:
        fallback = _fallback_suggestion(chunk)
        if fallback.prompt in seen_prompts:
            continue
        fallback = _dedupe_label(fallback, seen_labels)
        suggestions.append(fallback)
        seen_prompts.add(fallback.prompt)
        seen_labels.add(fallback.label)
        if len(suggestions) >= max_prompts:
            return suggestions

    return suggestions or default_prompt_suggestions()


def _suggestion_from_chunk(chunk: ChunkRecord) -> PromptSuggestion | None:
    topic = _topic_from_section_path(chunk.section_path, chunk.title)
    topic_lower = topic.lower()
    text_lower = chunk.text.lower()
    title = chunk.title

    if "salary adjustment" in text_lower or "salary adjustment" in topic_lower:
        return PromptSuggestion(
            label="Review window",
            prompt=f'In "{title}", when is the salary adjustment review window planned?',
        )

    if "first response" in text_lower or ("support" in text_lower and "business hour" in text_lower):
        return PromptSuggestion(
            label="Support SLA",
            prompt=f'In "{title}", what is the standard support first response time?',
        )

    if "postmortem" in text_lower or "postmortem" in topic_lower:
        return PromptSuggestion(
            label="Postmortem due",
            prompt=f'In "{title}", when is a postmortem due for a customer-facing incident?',
        )

    if "escalation" in text_lower or "escalation" in topic_lower:
        return PromptSuggestion(
            label="Escalation path",
            prompt=f'In "{title}", what is the escalation path?',
        )

    if _looks_like_pricing_chunk(text_lower):
        return PromptSuggestion(
            label="Pricing terms",
            prompt=f'What does "{title}" say about pricing, discounts, and trial terms?',
        )

    if _looks_like_ownership_chunk(text_lower, topic_lower):
        return PromptSuggestion(
            label=_short_label(f"{topic} owners"),
            prompt=f'In "{title}", who owns the responsibilities described in "{topic}"?',
        )

    if _looks_like_steps_chunk(text_lower):
        return PromptSuggestion(
            label="Main steps",
            prompt=f'What are the main steps described in "{title}" for "{topic}"?',
        )

    if _looks_like_timeline_chunk(text_lower):
        return PromptSuggestion(
            label=_short_label(f"{topic} timeline"),
            prompt=f'In "{title}", what timeline or deadline is described in "{topic}"?',
        )

    return None


def _fallback_suggestion(chunk: ChunkRecord) -> PromptSuggestion:
    topic = _topic_from_section_path(chunk.section_path, chunk.title)
    return PromptSuggestion(
        label=_short_label(topic),
        prompt=f'What are the key points in "{chunk.title}" about "{topic}"?',
    )


def _topic_from_section_path(section_path: str, title: str) -> str:
    parts = [part.strip() for part in section_path.split(">") if part.strip()]
    if parts:
        tail = parts[-1]
        if tail.lower() != title.lower():
            return tail
    return title


def _looks_like_pricing_chunk(text_lower: str) -> bool:
    return any(token in text_lower for token in ["pricing", "monthly plan", "annual plan", "discount", "trial", "invoice"])


def _looks_like_ownership_chunk(text_lower: str, topic_lower: str) -> bool:
    owner_tokens = [" owner ", " owner is", " owners ", "owns ", "responsible", "communication owner", "technical owner"]
    return any(token in text_lower for token in owner_tokens) or "owner" in topic_lower


def _looks_like_steps_chunk(text_lower: str) -> bool:
    return bool(re.search(r"\b1\.\s|\b2\.\s|\b3\.\s", text_lower)) or any(
        token in text_lower for token in ["step", "procedure", "process", "workflow"]
    )


def _looks_like_timeline_chunk(text_lower: str) -> bool:
    return any(
        token in text_lower
        for token in [
            "day",
            "days",
            "week",
            "weeks",
            "month",
            "months",
            "hour",
            "hours",
            "deadline",
            "window",
            "planned for",
            "due",
        ]
    )


def _short_label(raw_label: str) -> str:
    words = [word for word in re.findall(r"[A-Za-z0-9]+", raw_label) if word]
    compact = " ".join(words[:3]).strip() or "Key points"
    return compact[:24].rstrip()


def _dedupe_label(suggestion: PromptSuggestion, seen_labels: set[str]) -> PromptSuggestion:
    if suggestion.label not in seen_labels:
        return suggestion
    base_label = suggestion.label
    suffix = 2
    while f"{base_label} {suffix}" in seen_labels:
        suffix += 1
    return PromptSuggestion(label=f"{base_label} {suffix}", prompt=suggestion.prompt)
