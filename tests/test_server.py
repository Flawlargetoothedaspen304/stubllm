"""Tests for the FastAPI server."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stubllm.fixtures.models import (
    ContentMatch,
    Fixture,
    MatchCriteria,
    MessageMatch,
    MockResponse,
    Provider,
    ToolCallResponse,
)
from stubllm.server import create_app


@pytest.fixture()
def greeting_fixture() -> Fixture:
    return Fixture(
        name="greeting",
        match=MatchCriteria(
            provider=Provider.OPENAI,
            messages=[MessageMatch(content=ContentMatch(contains="hello"))],
        ),
        response=MockResponse(content="Hello! How can I help?"),
    )


@pytest.fixture()
def tool_fixture() -> Fixture:
    return Fixture(
        name="weather_tool",
        match=MatchCriteria(
            provider=Provider.OPENAI,
            messages=[MessageMatch(content=ContentMatch(contains="weather"))],
            tools_present=True,
        ),
        response=MockResponse(
            tool_calls=[
                ToolCallResponse(
                    id="call_abc",
                    function={"name": "get_weather", "arguments": '{"location": "Amsterdam"}'},
                )
            ]
        ),
    )


@pytest.fixture()
def client(greeting_fixture: Fixture, tool_fixture: Fixture) -> TestClient:
    app = create_app(fixtures=[greeting_fixture, tool_fixture])
    return TestClient(app)


class TestHealthEndpoints:
    def test_root(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "stubllm"
        assert data["fixtures_loaded"] == 2

    def test_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_fixtures_listing(self, client: TestClient) -> None:
        resp = client.get("/_fixtures")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2


class TestOpenAIChatCompletions:
    def test_matching_fixture(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello there"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert "Hello" in data["choices"][0]["message"]["content"]

    def test_no_matching_fixture_returns_fallback(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "something with no fixture"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "No fixture matched" in data["choices"][0]["message"]["content"]

    def test_tool_call_response(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "what's the weather?"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        tool_calls = data["choices"][0]["message"]["tool_calls"]
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert data["choices"][0]["finish_reason"] == "tool_calls"

    def test_response_format_json(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
                "response_format": {"type": "json_schema"},
            },
        )
        assert resp.status_code == 200
        import json
        content = resp.json()["choices"][0]["message"]["content"]
        # Should be valid JSON
        json.loads(content)


class TestOpenAIEmbeddings:
    def test_embeddings(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/embeddings",
            json={"model": "text-embedding-ada-002", "input": ["hello", "world"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert len(data["data"][0]["embedding"]) == 1536


class TestOpenAIModels:
    def test_list_models(self, client: TestClient) -> None:
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert any(m["id"] == "gpt-4o" for m in data["data"])


class TestStatsEndpoint:
    def test_stats_initially_empty(self, client: TestClient) -> None:
        resp = client.get("/_stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "fixture_call_counts" in data
        assert isinstance(data["fixture_call_counts"], dict)

    def test_stats_increments_after_call(self) -> None:
        from stubllm.fixtures.models import Fixture, MatchCriteria, MockResponse, Provider

        fixtures = [
            Fixture(
                name="stat_test",
                match=MatchCriteria(provider=Provider.OPENAI),
                response=MockResponse(content="counted"),
            )
        ]
        app = create_app(fixtures=fixtures)
        c = TestClient(app)
        c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        resp = c.get("/_stats")
        assert resp.status_code == 200
        assert resp.json()["fixture_call_counts"].get("stat_test", 0) == 1

    def test_fixture_files_loading(self, tmp_path: object) -> None:
        """create_app accepts fixture_files= to load individual YAML files."""
        from pathlib import Path

        import yaml

        assert isinstance(tmp_path, Path)
        f = tmp_path / "test.yaml"
        f.write_text(
            yaml.dump([{"name": "loaded", "response": {"content": "from file"}}])
        )

        app = create_app(fixture_files=[f])
        c = TestClient(app)
        resp = c.get("/_fixtures")
        assert resp.status_code == 200
        assert any(fx["name"] == "loaded" for fx in resp.json()["fixtures"])

    def test_fixture_dirs_loading(self, tmp_path: object) -> None:
        """create_app accepts fixture_dirs= to scan a directory for YAML files."""
        from pathlib import Path

        import yaml

        assert isinstance(tmp_path, Path)
        d = tmp_path / "fixtures"
        d.mkdir()
        (d / "test.yaml").write_text(
            yaml.dump([{"name": "from_dir", "response": {"content": "dir loaded"}}])
        )

        app = create_app(fixture_dirs=[d])
        c = TestClient(app)
        resp = c.get("/_fixtures")
        assert resp.status_code == 200
        assert any(fx["name"] == "from_dir" for fx in resp.json()["fixtures"])


class TestOpenAIStreaming:
    def test_streaming_response(self, client: TestClient) -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        content = resp.text
        assert "data: " in content
        assert "[DONE]" in content
