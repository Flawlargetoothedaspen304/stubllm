"""Tests for the record-and-replay proxy."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stubllm.recorder.proxy import (
    _build_fixture,
    _detect_provider,
    _extract_response,
    _maybe_record_fixture,
    _sanitize_headers,
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
        headers = {
            "host": "localhost:8765",
            "authorization": "Bearer sk-123",
            "content-type": "application/json",
        }
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


class TestCreateRecordingApp:
    def _make_upstream(self, body: dict, status: int = 200) -> MagicMock:
        mock = MagicMock()
        mock.status_code = status
        mock.content = json.dumps(body).encode()
        mock.headers = {"content-type": "application/json"}
        return mock

    def _make_app(self, upstream_mock: MagicMock, tmp_path: Path):
        from fastapi.testclient import TestClient
        from stubllm.recorder.proxy import create_recording_app

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=upstream_mock)

        with patch("stubllm.recorder.proxy.httpx.AsyncClient") as MockAsyncClient:
            MockAsyncClient.return_value = mock_client
            app = create_recording_app("https://api.openai.com", tmp_path)

        return TestClient(app), mock_client

    def test_proxy_forwards_request_and_returns_upstream_body(self, tmp_path: Path) -> None:
        response_body = {"choices": [{"message": {"role": "assistant", "content": "Hi!"}}]}
        upstream = self._make_upstream(response_body)
        client, mock_client = self._make_app(upstream, tmp_path)

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "Hi!"
        mock_client.request.assert_awaited_once()

    def test_proxy_records_fixture_on_200(self, tmp_path: Path) -> None:
        response_body = {"choices": [{"message": {"role": "assistant", "content": "Hi!"}}]}
        upstream = self._make_upstream(response_body)
        client, _ = self._make_app(upstream, tmp_path)

        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
        )
        yaml_files = list(tmp_path.glob("*.yaml"))
        assert len(yaml_files) == 1

    def test_proxy_does_not_record_on_error_status(self, tmp_path: Path) -> None:
        upstream = self._make_upstream({"error": "rate limit"}, status=429)
        client, _ = self._make_app(upstream, tmp_path)

        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert list(tmp_path.glob("*.yaml")) == []

    def test_proxy_does_not_record_for_unknown_path(self, tmp_path: Path) -> None:
        upstream = self._make_upstream({"ok": True})
        client, _ = self._make_app(upstream, tmp_path)

        client.post("/unknown/endpoint", json={"data": "value"})
        assert list(tmp_path.glob("*.yaml")) == []

    def test_proxy_handles_non_json_upstream(self, tmp_path: Path) -> None:
        upstream = MagicMock()
        upstream.status_code = 200
        upstream.content = b"plain text response"
        upstream.headers = {"content-type": "text/plain"}

        client, _ = self._make_app(upstream, tmp_path)
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_proxy_get_request(self, tmp_path: Path) -> None:
        upstream = self._make_upstream({"data": []})
        client, mock_client = self._make_app(upstream, tmp_path)

        client.get("/v1/models")
        # GET request should not record (POST check fails)
        assert list(tmp_path.glob("*.yaml")) == []


class TestMaybeRecordFixture:
    def test_writes_yaml_for_openai(self, tmp_path: Path) -> None:
        _maybe_record_fixture(
            path="/v1/chat/completions",
            request_body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "tell me a joke"}],
            },
            response_body={"choices": [{"message": {"content": "Why did the..."}}]},
            fixture_dir=tmp_path,
        )
        files = list(tmp_path.glob("*.yaml"))
        assert len(files) == 1

        import yaml
        data = yaml.safe_load(files[0].read_text())
        assert "fixtures" in data
        assert data["fixtures"][0]["match"]["provider"] == "openai"

    def test_skips_unknown_path(self, tmp_path: Path) -> None:
        _maybe_record_fixture(
            path="/unknown/path",
            request_body={},
            response_body={},
            fixture_dir=tmp_path,
        )
        assert list(tmp_path.glob("*.yaml")) == []


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

    def test_no_model_no_messages(self) -> None:
        """When model and messages are absent, fixture is still built without them."""
        fixture = _build_fixture(
            "/v1/chat/completions",
            {},  # no model, no messages
            {"choices": [{"message": {"content": "Hi"}}]},
        )
        assert fixture is not None
        assert "model" not in fixture["match"]
        assert "messages" not in fixture["match"]

    def test_non_string_last_user_content(self) -> None:
        """Non-string user content (e.g. list) skips message match criteria."""
        fixture = _build_fixture(
            "/v1/chat/completions",
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            },
            {"choices": [{"message": {"content": "Hi"}}]},
        )
        assert fixture is not None
        # last_user is a list, so isinstance(last_user, str) is False → no message match
        assert "messages" not in fixture["match"]


class TestExtractResponseBranches:
    def test_openai_with_tool_calls(self) -> None:
        body = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "foo"}}],
                }
            }]
        }
        result = _extract_response("openai", body, "gpt-4o")
        assert "tool_calls" in result

    def test_openai_no_content_no_tool_calls(self) -> None:
        body = {"choices": [{"message": {"role": "assistant", "content": None}}]}
        result = _extract_response("openai", body, "gpt-4o")
        assert "content" not in result
        assert "tool_calls" not in result

    def test_openai_no_usage(self) -> None:
        body = {"choices": [{"message": {"content": "hi"}}]}
        result = _extract_response("openai", body, "gpt-4o")
        assert "usage" not in result

    def test_anthropic_no_text_blocks(self) -> None:
        body = {"content": [{"type": "tool_use", "name": "foo"}]}
        result = _extract_response("anthropic", body, "claude-opus-4-6")
        assert "content" not in result

    def test_anthropic_no_usage(self) -> None:
        body = {"content": [{"type": "text", "text": "hi"}]}
        result = _extract_response("anthropic", body, "claude-opus-4-6")
        assert "usage" not in result

    def test_gemini_no_candidates(self) -> None:
        body = {}
        result = _extract_response("gemini", body, "gemini-pro")
        assert "content" not in result
