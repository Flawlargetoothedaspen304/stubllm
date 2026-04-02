"""MockLLMServerFixture and use_fixtures — imported after coverage starts."""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
        """Add fixtures without replacing existing ones.

        Call counts for existing fixtures are preserved — only the new fixture
        names start at zero. Use replace_fixtures() to reset everything.
        """
        self._server.add_fixtures(fixtures)

    def replace_fixtures(self, fixtures: list[Fixture]) -> None:
        """Replace all current fixtures and reset sequence call counts."""
        self._server.replace_fixtures(fixtures)

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

    def assert_not_called(self) -> None:
        assert self.call_count == 0, f"Expected no calls, got {self.call_count}"

    def assert_called_n_times(self, n: int) -> None:
        assert self.call_count == n, f"Expected {n} call(s), got {self.call_count}"

    def assert_model_was(self, model: str) -> None:
        """Assert that at least one call used the given model name."""
        models_used = [c.get("body", {}).get("model") for c in self.calls]
        assert model in models_used, (
            f"Model {model!r} was not used. Models seen: {models_used}"
        )

    def assert_last_call_path(self, path: str) -> None:
        assert self.calls, "No calls recorded"
        last = self.calls[-1]
        assert last["path"] == path, f"Expected path {path!r}, got {last['path']!r}"

    def assert_fixture_hit(self, name: str, times: int | None = None) -> None:
        """Assert that the named fixture was matched (optionally exactly `times` times)."""
        counts: dict[str, int] = {}
        if self._server.app is not None:
            counts = dict(self._server.app.state.fixture_call_counts)
        hit_count = counts.get(name, 0)
        if times is None:
            assert hit_count > 0, (
                f"Fixture {name!r} was never hit. Counts: {counts}"
            )
        else:
            assert hit_count == times, (
                f"Expected fixture {name!r} to be hit {times} time(s), got {hit_count}. "
                f"Counts: {counts}"
            )


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
                from stubllm.fixtures.loader import FixtureLoader  # noqa: PLC0415

                server_fixture.reset()
                loader = FixtureLoader()
                fixtures: list[Any] = []
                for path in fixture_paths:
                    fixtures.extend(loader.load_file(Path(path)))
                server_fixture.replace_fixtures(fixtures)

            return func(*args, **kwargs)

        wrapper._use_fixtures_paths = list(fixture_paths)  # type: ignore[attr-defined]
        return wrapper

    return decorator
