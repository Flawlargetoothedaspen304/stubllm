"""Anthropic-compatible endpoint handlers."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from stubllm.fixtures.models import MockResponse, Provider
from stubllm.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Implements Anthropic-compatible API endpoints."""

    provider = Provider.ANTHROPIC

    def router(self) -> APIRouter:
        router = APIRouter()

        @router.post("/v1/messages")
        async def messages(request: Request) -> Any:
            body = await request.json()
            raw_messages = body.get("messages", [])
            model = body.get("model", "claude-opus-4-6")
            system = body.get("system")
            tools = body.get("tools")
            stream = body.get("stream", False)

            # Normalize Anthropic messages to common format
            normalized = _normalize_messages(raw_messages, system)

            mock_resp, fixture_name = self._matcher.match(
                provider=Provider.ANTHROPIC,
                endpoint="/v1/messages",
                messages=normalized,
                model=model,
                tools=tools,
                headers=dict(request.headers),
            )

            request_id = f"msg_{uuid.uuid4().hex[:24]}"

            if mock_resp.latency_ms > 0 and not stream:
                await asyncio.sleep(mock_resp.latency_ms / 1000.0)

            if stream:
                return self.make_streaming_response(mock_resp, model, request_id)

            payload = self.format_response(mock_resp, model, request_id)
            return JSONResponse(content=payload, status_code=mock_resp.http_status)

        return router

    def format_response(
        self, response: MockResponse, model: str, request_id: str
    ) -> dict[str, Any]:
        """Format a MockResponse as an Anthropic messages response."""
        content: list[dict[str, Any]] = []

        if response.tool_calls:
            for tc in response.tool_calls:
                import json as _json

                args = tc.function.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except _json.JSONDecodeError:
                        args = {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.get("name", "unknown"),
                        "input": args,
                    }
                )
        else:
            content.append({"type": "text", "text": response.content or ""})

        usage = response.usage
        return {
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": model,
            "stop_reason": "tool_use" if response.tool_calls else response.finish_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 10),
                "output_tokens": usage.get("completion_tokens", 20),
            },
        }

    def format_stream_chunk(
        self,
        delta: str,
        model: str,
        request_id: str,
        finish: bool,
        tool_call_chunk: dict[str, Any] | None,
    ) -> str:
        """Format an Anthropic SSE streaming chunk."""
        if tool_call_chunk:
            event_type = "content_block_start"
            event_data: dict[str, Any] = {
                "type": event_type,
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_call_chunk["id"],
                    "name": tool_call_chunk["function"].get("name", ""),
                    "input": {},
                },
            }
        elif finish:
            event_type = "message_delta"
            event_data = {
                "type": event_type,
                "delta": {"type": "end_turn", "stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 20},
            }
        else:
            event_type = "content_block_delta"
            event_data = {
                "type": event_type,
                "index": 0,
                "delta": {"type": "text_delta", "text": delta},
            }

        return f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"

    def format_stream_final(self, model: str, request_id: str) -> str:
        """Anthropic SSE stream ends with message_stop."""
        data = {"type": "message_stop"}
        return f"event: message_stop\ndata: {json.dumps(data)}\n\n"


def _normalize_messages(
    messages: list[dict[str, Any]],
    system: str | None,
) -> list[dict[str, Any]]:
    """Convert Anthropic message format to common format for matching."""
    normalized: list[dict[str, Any]] = []
    if system:
        normalized.append({"role": "system", "content": system})
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # Content blocks: extract text
            text = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            normalized.append({"role": role, "content": text})
        else:
            normalized.append({"role": role, "content": content})
    return normalized
