"""Tests for the OpenAI provider."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stubllm.fixtures.models import Fixture, MatchCriteria, MockResponse, Provider, ToolCallResponse
from stubllm.server import create_app


@pytest.fixture()
def client() -> TestClient:
    fixtures = [
        Fixture(
            name="hello",
            match=MatchCriteria(provider=Provider.OPENAI),
            response=MockResponse(content="Hi from OpenAI mock!"),
        )
    ]
    return TestClient(create_app(fixtures=fixtures))


def test_chat_completion_structure(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    data = resp.json()
    assert "id" in data
    assert data["object"] == "chat.completion"
    assert "created" in data
    assert "choices" in data
    assert "usage" in data
    assert data["system_fingerprint"] == "stubllm"


def test_chat_completion_message_content(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    msg = resp.json()["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Hi from OpenAI mock!"


def test_tool_call_format() -> None:
    fixtures = [
        Fixture(
            name="tool",
            match=MatchCriteria(provider=Provider.OPENAI),
            response=MockResponse(
                tool_calls=[
                    ToolCallResponse(
                        id="call_001",
                        function={"name": "search", "arguments": '{"q": "test"}'},
                    )
                ]
            ),
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "search"}]},
    )
    data = resp.json()
    tc = data["choices"][0]["message"]["tool_calls"][0]
    assert tc["id"] == "call_001"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "search"
    assert data["choices"][0]["message"]["content"] is None
    assert data["choices"][0]["finish_reason"] == "tool_calls"


def test_usage_fields(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    usage = resp.json()["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage


def test_embeddings_endpoint(client: TestClient) -> None:
    resp = client.post(
        "/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "Hello world"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert data["data"][0]["object"] == "embedding"
    assert isinstance(data["data"][0]["embedding"], list)
    assert data["model"] == "text-embedding-3-small"


def test_models_endpoint(client: TestClient) -> None:
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert all("id" in m for m in data["data"])
