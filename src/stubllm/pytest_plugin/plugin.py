"""pytest plugin: provides stubllm_server fixture and @use_fixtures decorator."""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from stubllm.fixtures.models import Fixture
    from stubllm.server import MockLLMServer


class MockLLMServerFixture:
    """Thin wrapper around MockLLMServer with assertion helpers."""

    def __init__(self, server: MockLLMServer) -> None:
        self._server = server

    @property
    def url(self) -> str:
        return self._server.url

    @property
    def openai_url(self) -> str:
        """Base URL for the OpenAI SDK (includes /v1/ prefix).

        The OpenAI Python SDK does not add /v1/ itself, so use this property:

            client = openai.OpenAI(base_url=stubllm_server.openai_url, api_key="test")
        """
        return f"{self._server.url}/v1/"

    @property
    def call_count(self) -> int:
        return self._server.call_count

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._server.calls

    def reset(self) -> None:
        """Reset call log between tests."""
        self._server.reset_calls()

    def add_fixtures(self, fixtures: list[Fixture]) -> None:
        self._server.add_fixtures(fixtures)

    def assert_called_once(self) -> None:
        assert self.call_count == 1, f"Expected 1 call, got {self.call_count}"

    def assert_called_with_prompt(self, expected: str, *, case_sensitive: bool = False) -> None:
        """Assert that at least one call contained the given prompt text."""
        for call in self.calls:
            messages = call.get("body", {}).get("messages", [])
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    haystack = content if case_sensitive else content.lower()
                    needle = expected if case_sensitive else expected.lower()
                    if needle in haystack:
                        return
        raise AssertionError(
            f"No call found containing prompt {expected!r}. "
            f"Calls made: {[c.get('body', {}).get('messages') for c in self.calls]}"
        )

    def assert_last_call_path(self, path: str) -> None:
        assert self.calls, "No calls recorded"
        last = self.calls[-1]
        assert last["path"] == path, f"Expected path {path!r}, got {last['path']!r}"


# -----------------------------------------------------------------------
# Session-scoped server (shared across the test session)
# -----------------------------------------------------------------------


@pytest.fixture(scope="session")
def stubllm_server(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Session-scoped stubllm server fixture.

    Starts a server once per test session. Tests can call .reset() between runs.
    Use the @use_fixtures decorator to load fixtures per test.
    """
    # Lazy import to prevent early module loading before coverage starts
    from stubllm.server import MockLLMServer  # noqa: PLC0415

    server = MockLLMServer(host="127.0.0.1", port=0)
    server.start()
    wrapper = MockLLMServerFixture(server)
    yield wrapper
    server.stop()


# -----------------------------------------------------------------------
# @use_fixtures decorator
# -----------------------------------------------------------------------


def use_fixtures(*fixture_paths: str | Path) -> Callable[..., Any]:
    """Decorator: load fixture files for a specific test.

    Usage::

        @use_fixtures("fixtures/chat.yaml")
        def test_greeting(stubllm_server):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            server_fixture: MockLLMServerFixture | None = kwargs.get("stubllm_server")  # type: ignore[assignment]
            if server_fixture is None:
                for arg in args:
                    if isinstance(arg, MockLLMServerFixture):
                        server_fixture = arg
                        break

            if server_fixture is not None:
                # Lazy import to prevent early loading before coverage starts
                from stubllm.fixtures.loader import FixtureLoader  # noqa: PLC0415

                server_fixture.reset()
                loader = FixtureLoader()
                fixtures: list[Any] = []
                for path in fixture_paths:
                    fixtures.extend(loader.load_file(Path(path)))
                server_fixture.add_fixtures(fixtures)

            return func(*args, **kwargs)

        wrapper._use_fixtures_paths = list(fixture_paths)  # type: ignore[attr-defined]
        return wrapper

    return decorator
