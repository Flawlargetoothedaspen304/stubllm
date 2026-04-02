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
    from stubllm.fixtures.models import MessageMatch, ContentMatch
    fixtures[0].match.messages = [MessageMatch(role="system", content=ContentMatch(contains="helpful"))]

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
