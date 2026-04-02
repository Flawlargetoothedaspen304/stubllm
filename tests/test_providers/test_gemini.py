"""Tests for the Gemini provider."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stubllm.fixtures.models import Fixture, MatchCriteria, MockResponse, Provider, ToolCallResponse
from stubllm.server import create_app


@pytest.fixture()
def client() -> TestClient:
    fixtures = [
        Fixture(
            name="gemini_response",
            match=MatchCriteria(provider=Provider.GEMINI),
            response=MockResponse(content="Hi from Gemini mock!"),
        )
    ]
    return TestClient(create_app(fixtures=fixtures))


def test_generate_content_structure(client: TestClient) -> None:
    resp = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "candidates" in data
    assert data["candidates"][0]["content"]["role"] == "model"
    assert data["candidates"][0]["content"]["parts"][0]["text"] == "Hi from Gemini mock!"


def test_usage_metadata(client: TestClient) -> None:
    resp = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )
    meta = resp.json()["usageMetadata"]
    assert "promptTokenCount" in meta
    assert "candidatesTokenCount" in meta
    assert "totalTokenCount" in meta


def test_function_call_response() -> None:
    fixtures = [
        Fixture(
            name="func_call",
            match=MatchCriteria(provider=Provider.GEMINI),
            response=MockResponse(
                tool_calls=[
                    ToolCallResponse(
                        id="fc_001",
                        function={"name": "search", "arguments": '{"query": "test"}'},
                    )
                ]
            ),
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))
    resp = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "search for test"}]}]},
    )
    data = resp.json()
    part = data["candidates"][0]["content"]["parts"][0]
    assert "functionCall" in part
    assert part["functionCall"]["name"] == "search"


def test_error_response_rate_limit() -> None:
    fixtures = [
        Fixture(
            name="rate_limit",
            match=MatchCriteria(provider=Provider.GEMINI),
            response=MockResponse(
                http_status=429,
                error_message="Rate limit exceeded.",
                error_code="RESOURCE_EXHAUSTED",
            ),
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))
    resp = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )
    assert resp.status_code == 429
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == 429
    assert data["error"]["message"] == "Rate limit exceeded."
    assert data["error"]["status"] == "RESOURCE_EXHAUSTED"


def test_error_response_500() -> None:
    fixtures = [
        Fixture(
            name="internal_error",
            match=MatchCriteria(provider=Provider.GEMINI),
            response=MockResponse(
                http_status=500,
                error_message="Internal error.",
            ),
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))
    resp = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )
    assert resp.status_code == 500
    data = resp.json()
    assert "error" in data
    assert data["error"]["message"] == "Internal error."


def test_stream_generate_content(client: TestClient) -> None:
    resp = client.post(
        "/v1beta/models/gemini-pro:streamGenerateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "stream this"}]}]},
    )
    assert resp.status_code == 200
    assert "data: " in resp.text


def test_stream_generate_content_with_tool_call() -> None:
    fixtures = [
        Fixture(
            name="stream_tool",
            match=MatchCriteria(provider=Provider.GEMINI),
            response=MockResponse(
                tool_calls=[
                    ToolCallResponse(
                        id="fc_stream",
                        function={"name": "lookup", "arguments": '{"q": "x"}'},
                    )
                ]
            ),
        )
    ]
    c = TestClient(create_app(fixtures=fixtures))
    resp = c.post(
        "/v1beta/models/gemini-pro:streamGenerateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "lookup x"}]}]},
    )
    assert resp.status_code == 200
    assert "functionCall" in resp.text


def test_generate_content_with_latency(client: TestClient) -> None:
    import time

    fixtures = [
        Fixture(
            name="slow",
            match=MatchCriteria(provider=Provider.GEMINI),
            response=MockResponse(content="ok", latency_ms=50),
        )
    ]
    c = TestClient(create_app(fixtures=fixtures))
    start = time.monotonic()
    resp = c.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )
    elapsed_ms = (time.monotonic() - start) * 1000
    assert resp.status_code == 200
    assert elapsed_ms >= 40  # allow slight timing slack


def test_format_stream_chunk_invalid_json_args() -> None:
    """format_stream_chunk with unparseable JSON args falls back to empty dict."""
    from stubllm.fixtures.matcher import FixtureMatcher
    from stubllm.providers.gemini import GeminiProvider

    provider = GeminiProvider(FixtureMatcher([]))
    chunk = provider.format_stream_chunk(
        delta="",
        model="gemini-pro",
        request_id="req-1",
        finish=False,
        tool_call_chunk={
            "id": "tc1",
            "type": "function",
            "function": {"name": "broken", "arguments": "NOT_VALID_JSON"},
        },
    )
    # Should not raise; args fall back to {}
    import json as _json
    data = _json.loads(chunk.removeprefix("data: "))
    part = data["candidates"][0]["content"]["parts"][0]
    assert part["functionCall"]["args"] == {}


def test_stream_generate_content_error_response() -> None:
    """streamGenerateContent must return HTTP error for error fixtures, not stream them."""
    fixtures = [
        Fixture(
            name="stream_err",
            match=MatchCriteria(provider=Provider.GEMINI),
            response=MockResponse(
                http_status=429,
                error_message="Rate limited.",
                error_code="RESOURCE_EXHAUSTED",
            ),
        )
    ]
    c = TestClient(create_app(fixtures=fixtures))
    resp = c.post(
        "/v1beta/models/gemini-pro:streamGenerateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )
    assert resp.status_code == 429
    data = resp.json()
    assert data["error"]["code"] == 429
    assert data["error"]["status"] == "RESOURCE_EXHAUSTED"


def test_stream_generate_content_latency() -> None:
    """streamGenerateContent respects latency_ms."""
    import time

    fixtures = [
        Fixture(
            name="slow_stream",
            match=MatchCriteria(provider=Provider.GEMINI),
            response=MockResponse(content="slow", latency_ms=50),
        )
    ]
    c = TestClient(create_app(fixtures=fixtures))
    start = time.monotonic()
    resp = c.post(
        "/v1beta/models/gemini-pro:streamGenerateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )
    elapsed_ms = (time.monotonic() - start) * 1000
    assert resp.status_code == 200
    assert elapsed_ms >= 40


def test_generate_content_tool_call_dict_arguments() -> None:
    """format_response handles tool arguments passed as a dict (not a JSON string)."""
    fixtures = [
        Fixture(
            name="dict_args",
            match=MatchCriteria(provider=Provider.GEMINI),
            response=MockResponse(
                tool_calls=[
                    ToolCallResponse(
                        id="fc_002",
                        function={"name": "lookup", "arguments": {"q": "test"}},
                    )
                ]
            ),
        )
    ]
    c = TestClient(create_app(fixtures=fixtures))
    resp = c.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "lookup"}]}]},
    )
    assert resp.status_code == 200
    part = resp.json()["candidates"][0]["content"]["parts"][0]
    assert part["functionCall"]["args"] == {"q": "test"}


def test_generate_content_tool_call_invalid_json_arguments() -> None:
    """format_response falls back to empty dict for unparseable JSON arguments."""
    fixtures = [
        Fixture(
            name="bad_json",
            match=MatchCriteria(provider=Provider.GEMINI),
            response=MockResponse(
                tool_calls=[
                    ToolCallResponse(
                        id="fc_003",
                        function={"name": "broken", "arguments": "NOT_VALID_JSON"},
                    )
                ]
            ),
        )
    ]
    c = TestClient(create_app(fixtures=fixtures))
    resp = c.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "broken"}]}]},
    )
    assert resp.status_code == 200
    part = resp.json()["candidates"][0]["content"]["parts"][0]
    assert part["functionCall"]["args"] == {}


def test_model_role_normalization() -> None:
    """Gemini 'model' role should be normalized to 'assistant' for matching."""
    from stubllm.fixtures.models import ContentMatch, MessageMatch

    fixtures = [
        Fixture(
            name="multi_turn",
            match=MatchCriteria(
                provider=Provider.GEMINI,
                messages=[MessageMatch(role="user", content=ContentMatch(contains="follow up"))],
            ),
            response=MockResponse(content="Follow-up matched!"),
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))
    resp = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={
            "contents": [
                {"role": "user", "parts": [{"text": "first question"}]},
                {"role": "model", "parts": [{"text": "first answer"}]},
                {"role": "user", "parts": [{"text": "follow up question"}]},
            ]
        },
    )
    data = resp.json()
    assert data["candidates"][0]["content"]["parts"][0]["text"] == "Follow-up matched!"
