"""Server-Sent Events streaming simulation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from stubllm.fixtures.models import MockResponse

if TYPE_CHECKING:
    from stubllm.providers.base import BaseProvider


async def stream_response(
    provider: BaseProvider,
    response: MockResponse,
    model: str,
    request_id: str,
) -> AsyncIterator[str]:
    """Yield SSE chunks from a MockResponse, simulating token-by-token streaming."""
    delay = response.stream_chunk_delay_ms / 1000.0

    # Simulate initial latency
    if response.latency_ms > 0:
        await asyncio.sleep(response.latency_ms / 1000.0)

    if response.tool_calls:
        # Stream tool calls as a single chunk (they're not easily split)
        for tc in response.tool_calls:
            chunk = provider.format_stream_chunk(
                delta="",
                model=model,
                request_id=request_id,
                finish=False,
                tool_call_chunk={
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function,
                },
            )
            yield chunk
            if delay > 0:
                await asyncio.sleep(delay)
    else:
        content = response.content or ""
        # Split into words for realistic token simulation
        tokens = _tokenize(content)
        for i, token in enumerate(tokens):
            is_last = i == len(tokens) - 1
            chunk = provider.format_stream_chunk(
                delta=token,
                model=model,
                request_id=request_id,
                finish=is_last,
                tool_call_chunk=None,
            )
            yield chunk
            if delay > 0 and not is_last:
                await asyncio.sleep(delay)

    final = provider.format_stream_final(model=model, request_id=request_id)
    if final:
        yield final


def _tokenize(text: str) -> list[str]:
    """Split text into tokens (words + spaces) for streaming simulation."""
    if not text:
        return [""]
    tokens: list[str] = []
    current = ""
    for char in text:
        current += char
        if char in (" ", "\n", "\t") or len(current) >= 4:
            tokens.append(current)
            current = ""
    if current:
        tokens.append(current)
    return tokens
