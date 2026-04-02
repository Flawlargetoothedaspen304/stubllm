"""OpenAI-compatible endpoint handlers."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from stubllm.fixtures.models import MockResponse, Provider
from stubllm.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Implements OpenAI-compatible API endpoints."""

    provider = Provider.OPENAI

    def router(self) -> APIRouter:
        router = APIRouter()

        @router.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> Any:
            body = await request.json()
            messages = body.get("messages", [])
            model = body.get("model", "gpt-4o")
            tools = body.get("tools") or body.get("functions")
            stream = body.get("stream", False)
            response_format = body.get("response_format")

            fixture, fixture_name = self._matcher.match(
                provider=Provider.OPENAI,
                endpoint="/v1/chat/completions",
                messages=messages,
                model=model,
                tools=tools,
                headers=dict(request.headers),
            )
            count = request.app.state.fixture_call_counts.get(fixture_name, 0)
            mock_resp = fixture.get_response(count)
            request.app.state.fixture_call_counts[fixture_name] = count + 1

            request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

            # Validate JSON schema if response_format specified
            if response_format and response_format.get("type") == "json_schema":
                mock_resp = _ensure_json_content(mock_resp)

            if mock_resp.latency_ms > 0 and not stream:
                await asyncio.sleep(mock_resp.latency_ms / 1000.0)

            if mock_resp.http_status >= 400:
                return JSONResponse(
                    content=_format_error(mock_resp),
                    status_code=mock_resp.http_status,
                )

            if stream:
                return self.make_streaming_response(mock_resp, model, request_id)

            payload = self.format_response(mock_resp, model, request_id)
            return JSONResponse(content=payload, status_code=mock_resp.http_status)

        @router.post("/v1/embeddings")
        async def embeddings(request: Request) -> Any:
            body = await request.json()
            model = body.get("model", "text-embedding-ada-002")
            inputs = body.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]

            data = [
                {
                    "object": "embedding",
                    "embedding": [0.0] * 1536,
                    "index": i,
                }
                for i in range(len(inputs))
            ]
            return JSONResponse(
                content={
                    "object": "list",
                    "data": data,
                    "model": model,
                    "usage": {"prompt_tokens": len(inputs) * 5, "total_tokens": len(inputs) * 5},
                }
            )

        @router.get("/v1/models")
        async def list_models() -> Any:
            models = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "text-embedding-ada-002",
            ]
            now = int(time.time())
            return JSONResponse(
                content={
                    "object": "list",
                    "data": [
                        {
                            "id": m,
                            "object": "model",
                            "created": now,
                            "owned_by": "stubllm",
                        }
                        for m in models
                    ],
                }
            )

        return router

    def format_response(
        self, response: MockResponse, model: str, request_id: str
    ) -> dict[str, Any]:
        """Format a MockResponse as an OpenAI chat completion response."""
        message: dict[str, Any] = {"role": "assistant"}

        if response.tool_calls:
            message["content"] = None
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function,
                }
                for tc in response.tool_calls
            ]
        else:
            message["content"] = response.content

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": (
                        "tool_calls" if response.tool_calls else response.finish_reason
                    ),
                    "logprobs": None,
                }
            ],
            "usage": response.usage,
            "system_fingerprint": "stubllm",
        }

    def format_stream_chunk(
        self,
        delta: str,
        model: str,
        request_id: str,
        finish: bool,
        tool_call_chunk: dict[str, Any] | None,
    ) -> str:
        import json

        chunk: dict[str, Any] = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
        }

        if tool_call_chunk:
            chunk["choices"][0]["delta"] = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": tool_call_chunk["id"],
                        "type": tool_call_chunk["type"],
                        "function": tool_call_chunk["function"],
                    }
                ]
            }
        else:
            chunk["choices"][0]["delta"] = {"content": delta}
            if finish:
                chunk["choices"][0]["finish_reason"] = "stop"

        return f"data: {json.dumps(chunk)}\n\n"

    def format_stream_final(self, model: str, request_id: str) -> str:
        return "data: [DONE]\n\n"


def _format_error(response: MockResponse) -> dict[str, Any]:
    """Format a MockResponse as an OpenAI-style error body."""
    error_type = response.error_code or _default_error_type(response.http_status)
    return {
        "error": {
            "message": response.error_message or "An error occurred.",
            "type": error_type,
            "param": None,
            "code": response.error_code,
        }
    }


def _default_error_type(http_status: int) -> str:
    mapping = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        429: "rate_limit_error",
        500: "server_error",
        503: "server_error",
    }
    return mapping.get(http_status, "api_error")


def _ensure_json_content(response: MockResponse) -> MockResponse:
    """Ensure the response content is valid JSON (for response_format=json_schema)."""
    import json

    content = response.content or ""
    try:
        json.loads(content)
        return response
    except json.JSONDecodeError:
        # Wrap non-JSON content in a simple JSON object
        return response.model_copy(
            update={"content": json.dumps({"result": content})}
        )
