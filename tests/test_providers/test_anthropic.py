"""Tests for the Anthropic provider."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stubllm.fixtures.models import Fixture, MatchCriteria, MockResponse, Provider, ToolCallResponse
from stubllm.server import create_app


@pytest.fixture()
def client() -> TestClient:
    fixtures = [
        Fixture(
            name="claude_response",
            match=MatchCriteria(provider=Provider.ANTHROPIC),
            response=MockResponse(content="Hi from Anthropic mock!"),
        )
    ]
    return TestClient(create_app(fixtures=fixtures))


def test_messages_response_structure(client: TestClient) -> None:
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100,
        },
        headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert isinstance(data["content"], list)
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "Hi from Anthropic mock!"


def test_messages_usage_format(client: TestClient) -> None:
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "hi"}]},
    )
    usage = resp.json()["usage"]
    assert "input_tokens" in usage
    assert "output_tokens" in usage


def test_tool_use_response() -> None:
    fixtures = [
        Fixture(
            name="tool_use",
            match=MatchCriteria(provider=Provider.ANTHROPIC),
            response=MockResponse(
                tool_calls=[
                    ToolCallResponse(
                        id="toolu_01",
                        function={"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    )
                ]
            ),
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "weather?"}]},
    )
    data = resp.json()
    assert data["content"][0]["type"] == "tool_use"
    assert data["content"][0]["name"] == "get_weather"
    assert data["stop_reason"] == "tool_use"


def test_system_prompt_normalization() -> None:
    """System prompt should be included in matching."""
    fixtures = [
        Fixture(
            name="system_aware",
            match=MatchCriteria(
                provider=Provider.ANTHROPIC,
                messages=[
                    {"role": "system", "content": {"contains": "helpful"}},  # type: ignore[dict-item]
                ],
            ),
            response=MockResponse(content="System matched!"),
        )
    ]
    from stubllm.fixtures.models import ContentMatch, MessageMatch
    fixtures[0].match.messages = [
        MessageMatch(role="system", content=ContentMatch(contains="helpful"))
    ]

    client = TestClient(create_app(fixtures=fixtures))
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-6",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    data = resp.json()
    assert data["content"][0]["text"] == "System matched!"


def test_error_response_rate_limit() -> None:
    fixtures = [
        Fixture(
            name="rate_limit",
            match=MatchCriteria(provider=Provider.ANTHROPIC),
            response=MockResponse(
                http_status=429,
                error_message="Rate limit exceeded.",
                error_code="rate_limit_error",
            ),
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 429
    data = resp.json()
    assert data["type"] == "error"
    assert data["error"]["type"] == "rate_limit_error"
    assert data["error"]["message"] == "Rate limit exceeded."


def test_error_response_overloaded() -> None:
    fixtures = [
        Fixture(
            name="overloaded",
            match=MatchCriteria(provider=Provider.ANTHROPIC),
            response=MockResponse(
                http_status=529,
                error_message="Overloaded.",
            ),
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 529
    data = resp.json()
    assert data["type"] == "error"
    assert data["error"]["message"] == "Overloaded."


def test_latency_non_streaming() -> None:
    import time

    fixtures = [
        Fixture(
            name="slow",
            match=MatchCriteria(provider=Provider.ANTHROPIC),
            response=MockResponse(content="ok", latency_ms=50),
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))
    start = time.monotonic()
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "hi"}]},
    )
    elapsed_ms = (time.monotonic() - start) * 1000
    assert resp.status_code == 200
    assert elapsed_ms >= 40


def test_tool_use_dict_arguments() -> None:
    """Tool arguments passed as a dict (not a JSON string) should be handled directly."""
    fixtures = [
        Fixture(
            name="dict_args",
            match=MatchCriteria(provider=Provider.ANTHROPIC),
            response=MockResponse(
                tool_calls=[
                    ToolCallResponse(
                        id="toolu_02",
                        function={"name": "search", "arguments": {"query": "test"}},  # dict, not str
                    )
                ]
            ),
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "search"}]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"][0]["type"] == "tool_use"
    assert data["content"][0]["input"] == {"query": "test"}


def test_tool_use_invalid_json_arguments() -> None:
    """Invalid JSON string in tool arguments falls back to empty dict."""
    fixtures = [
        Fixture(
            name="bad_args",
            match=MatchCriteria(provider=Provider.ANTHROPIC),
            response=MockResponse(
                tool_calls=[
                    ToolCallResponse(
                        id="toolu_03",
                        function={"name": "lookup", "arguments": "NOT_VALID_JSON"},
                    )
                ]
            ),
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "lookup"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["content"][0]["input"] == {}


def test_streaming_with_tool_call() -> None:
    """Streaming with tool calls emits content_block_start events."""
    fixtures = [
        Fixture(
            name="stream_tool",
            match=MatchCriteria(provider=Provider.ANTHROPIC),
            response=MockResponse(
                tool_calls=[
                    ToolCallResponse(
                        id="toolu_stream",
                        function={"name": "weather", "arguments": "{}"},
                    )
                ]
            ),
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "weather?"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "content_block_start" in resp.text


def test_content_blocks_normalization() -> None:
    """Anthropic content blocks (list format) are normalized for matching."""
    from stubllm.fixtures.models import ContentMatch, MessageMatch

    fixtures = [
        Fixture(
            name="block_match",
            match=MatchCriteria(
                provider=Provider.ANTHROPIC,
                messages=[MessageMatch(content=ContentMatch(contains="block content"))],
            ),
            response=MockResponse(content="block matched"),
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "block content here"}],
                }
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "block matched"


def test_streaming_anthropic() -> None:
    fixtures = [
        Fixture(
            name="stream_test",
            match=MatchCriteria(provider=Provider.ANTHROPIC),
            response=MockResponse(content="Hello!", stream_chunk_delay_ms=0),
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))
    resp = client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "message_stop" in resp.text
