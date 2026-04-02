"""FastAPI server combining all provider routers."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from stubllm import __version__
from stubllm.fixtures.loader import FixtureLoader
from stubllm.fixtures.matcher import FixtureMatcher
from stubllm.fixtures.models import Fixture, MockResponse
from stubllm.providers.anthropic import AnthropicProvider
from stubllm.providers.gemini import GeminiProvider
from stubllm.providers.openai import OpenAIProvider


def create_app(
    fixtures: list[Fixture] | None = None,
    fixture_dirs: list[Path] | None = None,
    fixture_files: list[Path] | None = None,
    fallback_response: MockResponse | None = None,
) -> FastAPI:
    """Create and configure the mockllm FastAPI application.

    Args:
        fixtures: Pre-loaded fixture objects.
        fixture_dirs: Directories to scan for fixture files.
        fixture_files: Individual fixture files to load.
        fallback_response: Response when no fixture matches.
    """
    all_fixtures: list[Fixture] = list(fixtures or [])
    loader = FixtureLoader()

    if fixture_dirs:
        for d in fixture_dirs:
            all_fixtures.extend(loader.load_directory(d))

    if fixture_files:
        for f in fixture_files:
            all_fixtures.extend(loader.load_file(f))

    matcher = FixtureMatcher(all_fixtures, fallback_response)

    app = FastAPI(
        title="stubllm",
        description="Deterministic mock server for LLM APIs",
        version=__version__,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount provider routers
    openai_provider = OpenAIProvider(matcher)
    anthropic_provider = AnthropicProvider(matcher)
    gemini_provider = GeminiProvider(matcher)

    app.include_router(openai_provider.router())
    app.include_router(anthropic_provider.router())
    app.include_router(gemini_provider.router())

    # Health/meta endpoints
    @app.get("/")
    async def root() -> dict[str, Any]:
        return {
            "name": "stubllm",
            "version": __version__,
            "fixtures_loaded": len(all_fixtures),
            "providers": ["openai", "anthropic", "gemini"],
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/_fixtures")
    async def list_fixtures() -> dict[str, Any]:
        return {
            "count": len(all_fixtures),
            "fixtures": [
                {
                    "name": f.name,
                    "provider": f.match.provider.value if f.match.provider else "any",
                    "endpoint": f.match.endpoint,
                    "model": f.match.model,
                }
                for f in all_fixtures
            ],
        }

    # Store matcher reference so pytest plugin can access it
    app.state.matcher = matcher
    app.state.fixtures = all_fixtures

    return app


class MockLLMServer:
    """Manages a running stubllm server instance (used by pytest plugin and CLI)."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        **app_kwargs: Any,
    ) -> None:
        self._host = host
        self._port = port
        self._app_kwargs = app_kwargs
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._actual_port: int = port
        self._call_log: list[dict[str, Any]] = []
        self.app: FastAPI | None = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._actual_port}"

    @property
    def call_count(self) -> int:
        return len(self._call_log)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return list(self._call_log)

    def reset_calls(self) -> None:
        self._call_log.clear()

    def add_fixtures(self, fixtures: list[Fixture]) -> None:
        """Dynamically add fixtures at runtime."""
        if self.app is not None:
            existing = self.app.state.fixtures
            existing.extend(fixtures)
            self.app.state.matcher._fixtures = existing

    def replace_fixtures(self, fixtures: list[Fixture]) -> None:
        """Replace all current fixtures with the given list."""
        if self.app is not None:
            self.app.state.fixtures.clear()
            self.app.state.fixtures.extend(fixtures)
            self.app.state.matcher._fixtures = self.app.state.fixtures

    def start(self) -> None:
        """Start server in a background thread."""
        import socket

        self.app = create_app(**self._app_kwargs)
        _attach_call_logger(self.app, self._call_log)

        # If port=0, find a free port now (before starting the thread)
        if self._port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._actual_port = s.getsockname()[1]
        else:
            self._actual_port = self._port

        self._thread = threading.Thread(
            target=self._run_server,
            daemon=True,
        )
        self._thread.start()

        # Wait until the port is accepting connections
        deadline = 10.0
        step = 0.05
        waited = 0.0
        while waited < deadline:
            try:
                with socket.create_connection((self._host, self._actual_port), timeout=0.1):
                    return  # server is up
            except OSError:
                import time as _time
                _time.sleep(step)
                waited += step

        raise RuntimeError("stubllm server failed to start within 10 seconds")

    def stop(self) -> None:
        """Stop the running server."""
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)

    def _run_server(self) -> None:
        config = uvicorn.Config(
            app=self.app,
            host=self._host,
            port=self._actual_port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        asyncio.run(self._server.serve())


def _attach_call_logger(app: FastAPI, call_log: list[dict[str, Any]]) -> None:
    """Middleware that records every request for assertion helpers."""
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response

    class CallLoggerMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Any) -> Response:
            body_bytes = await request.body()

            # Re-inject body so downstream handlers can read it
            async def receive() -> dict[str, Any]:
                return {"type": "http.request", "body": body_bytes, "more_body": False}

            request._receive = receive  # type: ignore[assignment]

            import json

            try:
                body = json.loads(body_bytes) if body_bytes else {}
            except Exception:
                body = {}

            response = await call_next(request)
            call_log.append(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "body": body,
                    "status_code": response.status_code,
                }
            )
            return response

    app.add_middleware(CallLoggerMiddleware)
