"""Tests for the pytest plugin (fixtures and helpers)."""

from __future__ import annotations

import pytest

from stubllm.fixtures.models import Fixture, MatchCriteria, MockResponse, Provider
from stubllm.pytest_plugin._helpers import MockLLMServerFixture, use_fixtures
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

    def test_assert_not_called_passes_when_no_calls(
        self, plugin_server: MockLLMServerFixture
    ) -> None:
        plugin_server.reset()
        plugin_server.assert_not_called()

    def test_assert_not_called_fails_when_called(
        self, plugin_server: MockLLMServerFixture
    ) -> None:
        import httpx

        plugin_server.reset()
        httpx.post(
            f"{plugin_server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]},
        )
        with pytest.raises(AssertionError):
            plugin_server.assert_not_called()

    def test_assert_called_n_times(self, plugin_server: MockLLMServerFixture) -> None:
        import httpx

        plugin_server.reset()
        for _ in range(3):
            httpx.post(
                f"{plugin_server.url}/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]},
            )
        plugin_server.assert_called_n_times(3)

    def test_assert_called_n_times_fails_on_wrong_count(
        self, plugin_server: MockLLMServerFixture
    ) -> None:
        import httpx

        plugin_server.reset()
        httpx.post(
            f"{plugin_server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]},
        )
        with pytest.raises(AssertionError):
            plugin_server.assert_called_n_times(5)

    def test_assert_model_was(self, plugin_server: MockLLMServerFixture) -> None:
        import httpx

        plugin_server.reset()
        httpx.post(
            f"{plugin_server.url}/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "x"}]},
        )
        plugin_server.assert_model_was("gpt-4o-mini")

    def test_assert_model_was_fails_on_wrong_model(
        self, plugin_server: MockLLMServerFixture
    ) -> None:
        import httpx

        plugin_server.reset()
        httpx.post(
            f"{plugin_server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]},
        )
        with pytest.raises(AssertionError):
            plugin_server.assert_model_was("claude-3-opus")

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


    def test_fixture_isolation_between_decorator_calls(
        self, plugin_server: MockLLMServerFixture
    ) -> None:
        """Fixtures from a previous @use_fixtures call must not bleed into the next."""
        import httpx

        from stubllm.fixtures.models import (
            ContentMatch,
            Fixture,
            MatchCriteria,
            MessageMatch,
            MockResponse,
        )

        # Load fixture A: matches "alpha" → "response_alpha"
        plugin_server.replace_fixtures([
            Fixture(
                name="alpha",
                match=MatchCriteria(
                    messages=[MessageMatch(content=ContentMatch(contains="alpha"))]
                ),
                response=MockResponse(content="response_alpha"),
            )
        ])
        r1 = httpx.post(
            f"{plugin_server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "alpha"}]},
        )
        assert "response_alpha" in r1.json()["choices"][0]["message"]["content"]

        # Now replace with fixture B: matches "beta" → "response_beta"
        plugin_server.replace_fixtures([
            Fixture(
                name="beta",
                match=MatchCriteria(
                    messages=[MessageMatch(content=ContentMatch(contains="beta"))]
                ),
                response=MockResponse(content="response_beta"),
            )
        ])
        # "alpha" fixture must be gone — should get fallback, NOT response_alpha
        r2 = httpx.post(
            f"{plugin_server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "alpha"}]},
        )
        assert "response_alpha" not in r2.json()["choices"][0]["message"]["content"]


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

    def test_decorator_finds_positional_server_arg(self, plugin_server: MockLLMServerFixture) -> None:
        """Server passed as a positional arg (not a kwarg) must still be detected."""
        called = []

        @use_fixtures()
        def my_test(stubllm_server: MockLLMServerFixture) -> None:
            called.append(True)

        # Pass the server as a positional argument
        my_test(plugin_server)
        assert called == [True]

    def test_decorator_loads_fixture_file(self, tmp_path: pytest.TempPathFactory, plugin_server: MockLLMServerFixture) -> None:
        """Fixture files are loaded and pushed to the server when the test runs."""
        yaml_file = tmp_path / "seq_test.yaml"  # type: ignore[operator]
        yaml_file.write_text(
            "fixtures:\n"
            "  - name: file_fixture\n"
            "    response:\n"
            "      content: loaded_from_file\n"
        )
        loaded_server: list[MockLLMServerFixture] = []

        @use_fixtures(str(yaml_file))
        def my_test(stubllm_server: MockLLMServerFixture) -> None:
            loaded_server.append(stubllm_server)

        my_test(stubllm_server=plugin_server)

        # server fixture list must have the file fixture
        assert any(f.name == "file_fixture" for f in plugin_server._server.app.state.fixtures)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Tests for MockLLMServerFixture gaps
# ---------------------------------------------------------------------------


class TestMockLLMServerFixtureGaps:
    def test_assert_called_with_prompt_case_sensitive_matches(
        self, plugin_server: MockLLMServerFixture
    ) -> None:
        import httpx

        plugin_server.reset()
        httpx.post(
            f"{plugin_server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "CaseSensitive"}]},
        )
        plugin_server.assert_called_with_prompt("CaseSensitive", case_sensitive=True)

    def test_assert_called_with_prompt_case_sensitive_no_match(
        self, plugin_server: MockLLMServerFixture
    ) -> None:
        import httpx

        plugin_server.reset()
        httpx.post(
            f"{plugin_server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "lowercase"}]},
        )
        with pytest.raises(AssertionError):
            plugin_server.assert_called_with_prompt("LOWERCASE", case_sensitive=True)

    def test_assert_last_call_path_raises_when_no_calls(
        self, plugin_server: MockLLMServerFixture
    ) -> None:
        plugin_server.reset()
        with pytest.raises(AssertionError, match="No calls recorded"):
            plugin_server.assert_last_call_path("/v1/chat/completions")

    def test_assert_last_call_path_raises_on_wrong_path(
        self, plugin_server: MockLLMServerFixture
    ) -> None:
        import httpx

        plugin_server.reset()
        httpx.post(
            f"{plugin_server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]},
        )
        with pytest.raises(AssertionError):
            plugin_server.assert_last_call_path("/v1/wrong/path")

    def test_add_fixtures_appends_without_replacing(
        self, plugin_server: MockLLMServerFixture
    ) -> None:
        """add_fixtures must append — existing fixtures must still match."""
        plugin_server.replace_fixtures(
            [Fixture(name="base", response=MockResponse(content="base_response"))]
        )
        plugin_server.add_fixtures(
            [Fixture(name="extra", response=MockResponse(content="extra_response"))]
        )
        names = [f.name for f in plugin_server._server.app.state.fixtures]  # type: ignore[union-attr]
        assert "base" in names
        assert "extra" in names


# ---------------------------------------------------------------------------
# Test for the session-scoped stubllm_server pytest fixture
# ---------------------------------------------------------------------------


def test_session_stubllm_server_fixture(stubllm_server: MockLLMServerFixture) -> None:
    """The session-scoped stubllm_server fixture from the plugin must be usable."""
    import httpx

    resp = httpx.get(f"{stubllm_server.url}/health")
    assert resp.status_code == 200
    assert stubllm_server.openai_url.endswith("/v1/")
