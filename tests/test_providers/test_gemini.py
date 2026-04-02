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


def test_model_role_normalization() -> None:
    """Gemini 'model' role should be normalized to 'assistant' for matching."""
    from stubllm.fixtures.models import MessageMatch, ContentMatch

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
