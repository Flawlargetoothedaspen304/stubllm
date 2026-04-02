"""Tests for MockLLMServer (live threaded server)."""

from __future__ import annotations

import time

import httpx
import pytest

from stubllm.fixtures.models import Fixture, MatchCriteria, ContentMatch, MessageMatch, MockResponse, Provider
from stubllm.server import MockLLMServer


@pytest.fixture(scope="module")
def live_server() -> MockLLMServer:
    """Start a real in-process server for integration tests."""
    fixtures = [
        Fixture(
            name="hi",
            match=MatchCriteria(provider=Provider.OPENAI),
            response=MockResponse(content="Hello from live server!"),
        )
    ]
    server = MockLLMServer(host="127.0.0.1", port=0, fixtures=fixtures)
    server.start()
    yield server
    server.stop()


class TestMockLLMServerLifecycle:
    def test_server_starts_and_has_url(self, live_server: MockLLMServer) -> None:
        assert live_server.url.startswith("http://127.0.0.1:")

    def test_server_responds_to_health(self, live_server: MockLLMServer) -> None:
        with httpx.Client() as client:
            resp = client.get(f"{live_server.url}/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_server_handles_chat_completion(self, live_server: MockLLMServer) -> None:
        with httpx.Client() as client:
            resp = client.post(
                f"{live_server.url}/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer test-key"},
            )
        assert resp.status_code == 200
        assert "Hello from live server!" in resp.json()["choices"][0]["message"]["content"]

    def test_call_log_records_requests(self, live_server: MockLLMServer) -> None:
        live_server.reset_calls()
        with httpx.Client() as client:
            client.post(
                f"{live_server.url}/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]},
            )
        assert live_server.call_count == 1
        assert live_server.calls[0]["path"] == "/v1/chat/completions"

    def test_reset_clears_call_log(self, live_server: MockLLMServer) -> None:
        live_server.reset_calls()
        assert live_server.call_count == 0

    def test_add_fixtures_at_runtime(self, live_server: MockLLMServer) -> None:
        new_fixture = Fixture(
            name="dynamic",
            match=MatchCriteria(
                messages=[MessageMatch(content=ContentMatch(contains="dynamic_keyword_xyz"))]
            ),
            response=MockResponse(content="Dynamically added fixture!"),
        )
        live_server.add_fixtures([new_fixture])

        with httpx.Client() as client:
            resp = client.post(
                f"{live_server.url}/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "dynamic_keyword_xyz"}],
                },
            )
        assert "Dynamically added fixture!" in resp.json()["choices"][0]["message"]["content"]
