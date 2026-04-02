"""Google Gemini-compatible endpoint handlers."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from stubllm.fixtures.models import MockResponse, Provider
from stubllm.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Implements Google Gemini-compatible API endpoints."""

    provider = Provider.GEMINI

    def router(self) -> APIRouter:
        router = APIRouter()

        @router.post("/v1beta/models/{model_id}:generateContent")
        async def generate_content(model_id: str, request: Request) -> Any:
            body = await request.json()
            contents = body.get("contents", [])
            tools = body.get("tools")

            messages = _normalize_contents(contents)

            mock_resp, fixture_name = self._matcher.match(
                provider=Provider.GEMINI,
                endpoint=f"/v1beta/models/{model_id}:generateContent",
                messages=messages,
                model=model_id,
                tools=tools,
                headers=dict(request.headers),
            )

            if mock_resp.latency_ms > 0:
                await asyncio.sleep(mock_resp.latency_ms / 1000.0)

            payload = self.format_response(mock_resp, model_id, str(uuid.uuid4()))
            return JSONResponse(content=payload, status_code=mock_resp.http_status)

        @router.post("/v1beta/models/{model_id}:streamGenerateContent")
        async def stream_generate_content(model_id: str, request: Request) -> Any:
            body = await request.json()
            contents = body.get("contents", [])
            tools = body.get("tools")

            messages = _normalize_contents(contents)
            request_id = str(uuid.uuid4())

            mock_resp, fixture_name = self._matcher.match(
                provider=Provider.GEMINI,
                endpoint=f"/v1beta/models/{model_id}:streamGenerateContent",
                messages=messages,
                model=model_id,
                tools=tools,
                headers=dict(request.headers),
            )

            return self.make_streaming_response(
                mock_resp, model_id, request_id, media_type="text/event-stream"
            )

        return router

    def format_response(
        self, response: MockResponse, model: str, request_id: str
    ) -> dict[str, Any]:
        """Format a MockResponse as a Gemini generateContent response."""
        parts: list[dict[str, Any]] = []

        if response.tool_calls:
            for tc in response.tool_calls:
                import json as _json

                args = tc.function.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except _json.JSONDecodeError:
                        args = {}
                parts.append(
                    {
                        "functionCall": {
                            "name": tc.function.get("name", "unknown"),
                            "args": args,
                        }
                    }
                )
        else:
            parts.append({"text": response.content or ""})

        usage = response.usage
        return {
            "candidates": [
                {
                    "content": {"parts": parts, "role": "model"},
                    "finishReason": (
                        "STOP" if response.finish_reason == "stop"
                        else response.finish_reason.upper()
                    ),
                    "index": 0,
                    "safetyRatings": [],
                }
            ],
            "usageMetadata": {
                "promptTokenCount": usage.get("prompt_tokens", 10),
                "candidatesTokenCount": usage.get("completion_tokens", 20),
                "totalTokenCount": usage.get("total_tokens", 30),
            },
            "modelVersion": model,
        }

    def format_stream_chunk(
        self,
        delta: str,
        model: str,
        request_id: str,
        finish: bool,
        tool_call_chunk: dict[str, Any] | None,
    ) -> str:
        if tool_call_chunk:
            import json as _json

            args = tool_call_chunk["function"].get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except _json.JSONDecodeError:
                    args = {}
            parts = [{"functionCall": {
                "name": tool_call_chunk["function"].get("name", ""),
                "args": args,
            }}]
        else:
            parts = [{"text": delta}]

        data: dict[str, Any] = {
            "candidates": [
                {
                    "content": {"parts": parts, "role": "model"},
                    "finishReason": "STOP" if finish else None,
                    "index": 0,
                }
            ],
        }
        return f"data: {json.dumps(data)}\n\n"

    def format_stream_final(self, model: str, request_id: str) -> str:
        return ""  # Gemini SSE doesn't have a separate DONE marker


def _normalize_contents(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Gemini contents format to common message format."""
    messages: list[dict[str, Any]] = []
    for item in contents:
        role = item.get("role", "user")
        # Gemini uses "user" and "model" roles; normalize "model" → "assistant"
        if role == "model":
            role = "assistant"
        parts = item.get("parts", [])
        text = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
        messages.append({"role": role, "content": text})
    return messages
