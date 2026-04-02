"""Tests for the record-and-replay proxy."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from stubllm.recorder.proxy import (
    _build_fixture,
    _detect_provider,
    _extract_response,
    _sanitize_headers,
    create_recording_app,
)


class TestDetectProvider:
    def test_openai_chat(self) -> None:
        assert _detect_provider("/v1/chat/completions") == "openai"

    def test_openai_embeddings(self) -> None:
        assert _detect_provider("/v1/embeddings") == "openai"

    def test_anthropic(self) -> None:
        assert _detect_provider("/v1/messages") == "anthropic"

    def test_gemini(self) -> None:
        assert _detect_provider("/v1beta/models/gemini-pro:generateContent") == "gemini"

    def test_unknown(self) -> None:
        assert _detect_provider("/unknown/path") is None


class TestSanitizeHeaders:
    def test_removes_host(self) -> None:
        headers = {"host": "localhost:8765", "authorization": "Bearer sk-123", "content-type": "application/json"}
        sanitized = _sanitize_headers(headers)
        assert "host" not in sanitized
        assert "authorization" in sanitized  # We keep auth (user may want to forward)

    def test_removes_content_length(self) -> None:
        headers = {"content-length": "100", "x-custom": "value"}
        sanitized = _sanitize_headers(headers)
        assert "content-length" not in sanitized
        assert "x-custom" in sanitized


class TestExtractResponse:
    def test_openai_text_response(self) -> None:
        body = {
            "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        result = _extract_response("openai", body, "gpt-4o")
        assert result["content"] == "Hello!"
        assert result["usage"]["prompt_tokens"] == 5

    def test_anthropic_text_response(self) -> None:
        body = {
            "content": [{"type": "text", "text": "Hi there!"}],
            "usage": {"input_tokens": 5, "output_tokens": 8},
        }
        result = _extract_response("anthropic", body, "claude-opus-4-6")
        assert result["content"] == "Hi there!"
        assert result["usage"]["prompt_tokens"] == 5

    def test_gemini_text_response(self) -> None:
        body = {
            "candidates": [
                {"content": {"parts": [{"text": "Gemini response"}], "role": "model"}}
            ]
        }
        result = _extract_response("gemini", body, "gemini-pro")
        assert result["content"] == "Gemini response"


class TestBuildFixture:
    def test_openai_fixture(self) -> None:
        fixture = _build_fixture(
            "/v1/chat/completions",
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
            {"choices": [{"message": {"content": "Hi!"}}]},
        )
        assert fixture is not None
        assert fixture["match"]["provider"] == "openai"
        assert fixture["match"]["model"] == "gpt-4o"
        assert "hello" in str(fixture["match"]["messages"])

    def test_unknown_path_returns_none(self) -> None:
        fixture = _build_fixture("/unknown", {}, {})
        assert fixture is None
