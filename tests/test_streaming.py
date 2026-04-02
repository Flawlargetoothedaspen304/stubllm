"""Tests for SSE streaming simulation."""

from __future__ import annotations

import json

import pytest

from stubllm.fixtures.models import MockResponse, ToolCallResponse
from stubllm.providers.openai import OpenAIProvider
from stubllm.providers.anthropic import AnthropicProvider
from stubllm.fixtures.matcher import FixtureMatcher
from stubllm.streaming.sse import _tokenize, stream_response


class TestTokenize:
    def test_empty_string(self) -> None:
        tokens = _tokenize("")
        assert tokens == [""]

    def test_splits_on_spaces(self) -> None:
        tokens = _tokenize("hello world")
        assert "".join(tokens) == "hello world"
        assert len(tokens) > 1

    def test_short_content_single_token(self) -> None:
        tokens = _tokenize("Hi!")
        assert "".join(tokens) == "Hi!"


class TestOpenAIStreaming:
    def setup_method(self) -> None:
        self.matcher = FixtureMatcher([])
        self.provider = OpenAIProvider(self.matcher)

    @pytest.mark.asyncio
    async def test_streams_text_chunks(self) -> None:
        response = MockResponse(content="Hello world!", stream_chunk_delay_ms=0)
        chunks = []
        async for chunk in stream_response(self.provider, response, "gpt-4o", "test-id"):
            chunks.append(chunk)
        assert len(chunks) > 0
        # Last chunk should be [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"
        # Content chunks should be valid JSON
        for c in chunks[:-1]:
            assert c.startswith("data: ")
            data = json.loads(c[6:])
            assert "choices" in data

    @pytest.mark.asyncio
    async def test_streams_tool_calls(self) -> None:
        tc = ToolCallResponse(
            id="call_123",
            function={"name": "get_weather", "arguments": '{"city": "NYC"}'},
        )
        response = MockResponse(tool_calls=[tc], stream_chunk_delay_ms=0)
        chunks = []
        async for chunk in stream_response(self.provider, response, "gpt-4o", "test-id"):
            chunks.append(chunk)
        # Should have at least one tool call chunk + [DONE]
        assert any("tool_calls" in c for c in chunks[:-1])
        assert chunks[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_finish_reason_in_last_content_chunk(self) -> None:
        response = MockResponse(content="Hi", stream_chunk_delay_ms=0)
        chunks = []
        async for chunk in stream_response(self.provider, response, "gpt-4o", "test-id"):
            chunks.append(chunk)
        # Find last content chunk (before [DONE])
        content_chunks = [c for c in chunks if c != "data: [DONE]\n\n"]
        last_content = json.loads(content_chunks[-1][6:])
        assert last_content["choices"][0]["finish_reason"] == "stop"


class TestAnthropicStreaming:
    def setup_method(self) -> None:
        self.matcher = FixtureMatcher([])
        self.provider = AnthropicProvider(self.matcher)

    @pytest.mark.asyncio
    async def test_streams_anthropic_format(self) -> None:
        response = MockResponse(content="Hello!", stream_chunk_delay_ms=0)
        chunks = []
        async for chunk in stream_response(self.provider, response, "claude-3", "msg_123"):
            chunks.append(chunk)
        assert len(chunks) > 0
        # Should end with message_stop
        assert "message_stop" in chunks[-1]
        # Each chunk should have event: prefix
        for c in chunks:
            assert c.startswith("event: ")
