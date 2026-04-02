"""Tests for the pytest plugin (fixtures and helpers)."""

from __future__ import annotations

import pytest

from stubllm.fixtures.models import Fixture, MatchCriteria, MockResponse, Provider
from stubllm.pytest_plugin.plugin import MockLLMServerFixture, use_fixtures
from stubllm.server import MockLLMServer


@pytest.fixture(scope="module")
def plugin_server() -> MockLLMServer:
    fixtures = [
        Fixture(
            name="plugin_test",
            match=MatchCriteria(provider=Provider.OPENAI),
            response=MockResponse(content="Plugin test response"),
        )
    ]
    server = MockLLMServer(host="127.0.0.1", port=0, fixtures=fixtures)
    server.start()
    wrapper = MockLLMServerFixture(server)
    yield wrapper
    server.stop()


class TestMockLLMServerFixture:
    def test_url_is_accessible(self, plugin_server: MockLLMServerFixture) -> None:
        import httpx

        resp = httpx.get(f"{plugin_server.url}/health")
        assert resp.status_code == 200

    def test_openai_url_has_v1_prefix(self, plugin_server: MockLLMServerFixture) -> None:
        assert plugin_server.openai_url == f"{plugin_server.url}/v1/"
        assert "/v1/" in plugin_server.openai_url

    def test_call_count_increments(self, plugin_server: MockLLMServerFixture) -> None:
        import httpx

        plugin_server.reset()
        initial = plugin_server.call_count
        httpx.post(
            f"{plugin_server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "test"}]},
        )
        assert plugin_server.call_count == initial + 1

    def test_assert_called_with_prompt(self, plugin_server: MockLLMServerFixture) -> None:
        import httpx

        plugin_server.reset()
        httpx.post(
            f"{plugin_server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [
                {"role": "user", "content": "unique_test_prompt"}
            ]},
        )
        plugin_server.assert_called_with_prompt("unique_test_prompt")

    def test_assert_called_with_prompt_fails_when_not_sent(
        self, plugin_server: MockLLMServerFixture
    ) -> None:
        plugin_server.reset()
        with pytest.raises(AssertionError):
            plugin_server.assert_called_with_prompt("this_was_never_sent")

    def test_assert_called_once(self, plugin_server: MockLLMServerFixture) -> None:
        import httpx

        plugin_server.reset()
        httpx.post(
            f"{plugin_server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "once"}]},
        )
        plugin_server.assert_called_once()

    def test_assert_called_once_fails_on_two_calls(
        self, plugin_server: MockLLMServerFixture
    ) -> None:
        import httpx

        plugin_server.reset()
        for _ in range(2):
            httpx.post(
                f"{plugin_server.url}/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]},
            )
        with pytest.raises(AssertionError):
            plugin_server.assert_called_once()

    def test_assert_last_call_path(self, plugin_server: MockLLMServerFixture) -> None:
        import httpx

        plugin_server.reset()
        httpx.post(
            f"{plugin_server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]},
        )
        plugin_server.assert_last_call_path("/v1/chat/completions")

    def test_calls_list(self, plugin_server: MockLLMServerFixture) -> None:
        import httpx

        plugin_server.reset()
        httpx.post(
            f"{plugin_server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "list test"}]},
        )
        calls = plugin_server.calls
        assert len(calls) == 1
        assert calls[0]["path"] == "/v1/chat/completions"


class TestUseFixturesDecorator:
    def test_decorator_preserves_function_name(self) -> None:
        @use_fixtures("some_file.yaml")
        def my_test() -> None:
            pass

        assert my_test.__name__ == "my_test"

    def test_decorator_stores_paths(self) -> None:
        @use_fixtures("a.yaml", "b.yaml")
        def my_test() -> None:
            pass

        assert hasattr(my_test, "_use_fixtures_paths")
        assert len(my_test._use_fixtures_paths) == 2

    def test_decorator_calls_wrapped_function(self) -> None:
        called = []

        @use_fixtures()
        def my_test() -> None:
            called.append(True)

        my_test()
        assert called == [True]
