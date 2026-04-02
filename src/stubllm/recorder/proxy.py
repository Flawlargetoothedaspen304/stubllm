"""Record-and-replay proxy: forwards requests to a real LLM API and saves responses as fixtures."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import Response


def create_recording_app(target_url: str, fixture_dir: Path) -> FastAPI:
    """Create a FastAPI app that proxies requests to target_url and records them.

    Args:
        target_url: The real API base URL (e.g. https://api.openai.com).
        fixture_dir: Directory where fixture files will be saved.
    """
    app = FastAPI(title="stubllm recorder")
    client = httpx.AsyncClient(base_url=target_url, timeout=60.0)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(path: str, request: Request) -> Response:
        # Build forwarded headers (strip host, sanitize auth)
        headers = _sanitize_headers(dict(request.headers))
        body_bytes = await request.body()

        try:
            body = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            body = {}

        # Forward to real API
        upstream_response = await client.request(
            method=request.method,
            url=f"/{path}",
            headers=headers,
            content=body_bytes,
            params=dict(request.query_params),
        )

        response_body = upstream_response.content
        try:
            response_json = json.loads(response_body)
        except json.JSONDecodeError:
            response_json = None

        # Record as fixture if it's a successful chat/messages endpoint
        if (
            upstream_response.status_code == 200
            and response_json is not None
            and request.method == "POST"
        ):
            _maybe_record_fixture(
                path=f"/{path}",
                request_body=body,
                response_body=response_json,
                fixture_dir=fixture_dir,
            )

        return Response(
            content=response_body,
            status_code=upstream_response.status_code,
            headers=dict(upstream_response.headers),
            media_type=upstream_response.headers.get("content-type"),
        )

    return app


def _sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    """Remove or sanitize sensitive headers before forwarding."""
    sanitized = {}
    skip = {"host", "content-length", "transfer-encoding"}
    for k, v in headers.items():
        if k.lower() in skip:
            continue
        sanitized[k] = v
    return sanitized


def _maybe_record_fixture(
    path: str,
    request_body: dict[str, Any],
    response_body: dict[str, Any],
    fixture_dir: Path,
) -> None:
    """Convert a recorded request/response pair into a fixture file."""
    fixture = _build_fixture(path, request_body, response_body)
    if fixture is None:
        return

    timestamp = int(time.time())
    safe_name = re.sub(r"[^a-z0-9_]", "_", fixture["name"].lower())
    filename = fixture_dir / f"{safe_name}_{timestamp}.yaml"

    with open(filename, "w", encoding="utf-8") as fh:
        yaml.dump({"fixtures": [fixture]}, fh, allow_unicode=True, default_flow_style=False)


def _build_fixture(
    path: str,
    request_body: dict[str, Any],
    response_body: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a fixture dict from a request/response pair."""
    provider = _detect_provider(path)
    if not provider:
        return None

    name = f"recorded_{uuid.uuid4().hex[:8]}"
    messages = request_body.get("messages", [])
    model = request_body.get("model", "")

    # Extract last user message for the match criteria
    last_user = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        None,
    )

    match: dict[str, Any] = {"provider": provider}
    if model:
        match["model"] = model
    if last_user and isinstance(last_user, str):
        match["messages"] = [{"role": "user", "content": {"contains": last_user[:100]}}]
        name = f"recorded_{last_user[:20].replace(' ', '_').lower()}"

    response = _extract_response(provider, response_body, model)

    return {"name": name, "match": match, "response": response}


def _detect_provider(path: str) -> str | None:
    if (
        path.startswith("/v1/chat")
        or path.startswith("/v1/embeddings")
        or path.startswith("/v1/models")
    ):
        return "openai"
    if path.startswith("/v1/messages"):
        return "anthropic"
    if path.startswith("/v1beta/models"):
        return "gemini"
    return None


def _extract_response(
    provider: str,
    response_body: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Extract normalized response fields from the provider's response format."""
    response: dict[str, Any] = {"model": model or "mock-model"}

    if provider == "openai":
        choices = response_body.get("choices", [{}])
        choice = choices[0] if choices else {}
        msg = choice.get("message", {})
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        if content:
            response["content"] = content
        if tool_calls:
            response["tool_calls"] = tool_calls
        if "usage" in response_body:
            response["usage"] = response_body["usage"]

    elif provider == "anthropic":
        content_blocks = response_body.get("content", [])
        texts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
        if texts:
            response["content"] = " ".join(texts)
        usage = response_body.get("usage", {})
        if usage:
            response["usage"] = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            }

    elif provider == "gemini":
        candidates = response_body.get("candidates", [{}])
        candidate = candidates[0] if candidates else {}
        parts = candidate.get("content", {}).get("parts", [])
        texts = [p.get("text", "") for p in parts if "text" in p]
        if texts:
            response["content"] = " ".join(texts)

    return response
