"""pytest plugin: registers the session-scoped stubllm_server fixture."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture(scope="session")
def stubllm_server(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Session-scoped stubllm server fixture.

    Starts a server once per test session. Tests can call .reset() between runs.
    Use the @use_fixtures decorator to load fixtures per test.
    """
    # Lazy imports keep this module light when loaded early by pytest11
    from stubllm.pytest_plugin._helpers import MockLLMServerFixture  # noqa: PLC0415
    from stubllm.server import MockLLMServer  # noqa: PLC0415

    server = MockLLMServer(host="127.0.0.1", port=0)
    server.start()
    wrapper = MockLLMServerFixture(server)
    yield wrapper
    server.stop()
