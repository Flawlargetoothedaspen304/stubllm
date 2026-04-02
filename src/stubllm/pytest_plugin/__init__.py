"""pytest plugin for stubllm."""

from __future__ import annotations

from typing import Any

__all__ = ["MockLLMServerFixture", "use_fixtures"]


def __getattr__(name: str) -> Any:
    """Lazy re-export so _helpers.py is not loaded at plugin registration time."""
    if name in ("MockLLMServerFixture", "use_fixtures"):
        from stubllm.pytest_plugin._helpers import (  # noqa: PLC0415
            MockLLMServerFixture,
            use_fixtures,
        )

        _globals = globals()
        _globals["MockLLMServerFixture"] = MockLLMServerFixture
        _globals["use_fixtures"] = use_fixtures
        return _globals[name]
    raise AttributeError(f"module 'stubllm.pytest_plugin' has no attribute {name!r}")
