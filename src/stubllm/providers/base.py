"""Abstract base class for provider endpoint handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from stubllm.fixtures.matcher import FixtureMatcher
from stubllm.fixtures.models import MockResponse, Provider


class BaseProvider(ABC):
    """Base class for all LLM provider implementations."""

    provider: Provider

    def __init__(self, matcher: FixtureMatcher) -> None:
        self._matcher = matcher

    @abstractmethod
    def router(self) -> APIRouter:
        """Return the FastAPI router for this provider's endpoints."""
        ...

    @abstractmethod
    def format_response(self, response: MockResponse, model: str, request_id: str) -> dict[str, Any]:
        """Convert a normalized MockResponse into the provider's wire format."""
        ...

    @abstractmethod
    def format_stream_chunk(
        self,
        delta: str,
        model: str,
        request_id: str,
        finish: bool,
        tool_call_chunk: dict[str, Any] | None,
    ) -> str:
        """Format a single SSE chunk in the provider's wire format."""
        ...

    @abstractmethod
    def format_stream_final(self, model: str, request_id: str) -> str:
        """Format the final SSE event (e.g. [DONE] for OpenAI)."""
        ...

    def make_streaming_response(
        self,
        response: MockResponse,
        model: str,
        request_id: str,
        media_type: str = "text/event-stream",
    ) -> StreamingResponse:
        """Build a StreamingResponse using provider-specific SSE formatting."""
        from stubllm.streaming.sse import stream_response

        return StreamingResponse(
            stream_response(self, response, model, request_id),
            media_type=media_type,
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
