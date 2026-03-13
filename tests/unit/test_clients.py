import pytest

from local_agentic_rag.clients import _parse_json_response


def test_parse_json_response_accepts_plain_json() -> None:
    payload = _parse_json_response('{"answer":"4","grounded":true}', model="test-model")
    assert payload["answer"] == "4"
    assert payload["grounded"] is True


def test_parse_json_response_accepts_fenced_json() -> None:
    payload = _parse_json_response(
        '```json\n{"answer":"4","grounded":true}\n```',
        model="test-model",
    )
    assert payload["answer"] == "4"


def test_parse_json_response_extracts_json_from_mixed_text() -> None:
    payload = _parse_json_response(
        'Here is the result:\n{"answer":"4","grounded":true,"citations":[]}\nDone.',
        model="test-model",
    )
    assert payload["grounded"] is True
    assert payload["citations"] == []


def test_parse_json_response_rejects_non_json() -> None:
    with pytest.raises(RuntimeError, match="not valid JSON"):
        _parse_json_response("not json at all", model="test-model")
