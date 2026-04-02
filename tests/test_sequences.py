"""Tests for response sequences — successive calls return different responses."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stubllm.fixtures.models import Fixture, MatchCriteria, MockResponse, Provider
from stubllm.server import MockLLMServer, create_app


# ---------------------------------------------------------------------------
# Unit tests: Fixture.get_response()
# ---------------------------------------------------------------------------


def test_get_response_no_sequence_returns_response() -> None:
    fix = Fixture(
        name="simple",
        response=MockResponse(content="hello"),
    )
    assert fix.get_response(0).content == "hello"
    assert fix.get_response(5).content == "hello"


def test_get_response_sequence_returns_in_order() -> None:
    fix = Fixture(
        name="seq",
        sequence=[
            MockResponse(http_status=429, error_message="Rate limited"),
            MockResponse(content="Success!"),
        ],
    )
    r0 = fix.get_response(0)
    assert r0.http_status == 429

    r1 = fix.get_response(1)
    assert r1.http_status == 200
    assert r1.content == "Success!"


def test_get_response_sequence_clamps_to_last() -> None:
    fix = Fixture(
        name="seq",
        sequence=[
            MockResponse(http_status=429, error_message="Rate limited"),
            MockResponse(content="Final"),
        ],
    )
    assert fix.get_response(2).content == "Final"
    assert fix.get_response(99).content == "Final"


def test_fixture_rejects_both_response_and_sequence() -> None:
    with pytest.raises(ValueError, match="sequence"):
        Fixture(
            name="bad",
            response=MockResponse(content="explicit"),
            sequence=[MockResponse(content="a"), MockResponse(content="b")],
        )


def test_fixture_sequence_without_explicit_response_ok() -> None:
    """sequence alone (no explicit response=) must not raise."""
    fix = Fixture(
        name="ok",
        sequence=[MockResponse(content="a")],
    )
    assert fix.sequence is not None


# ---------------------------------------------------------------------------
# Integration tests: OpenAI provider with sequences
# ---------------------------------------------------------------------------


def _openai_post(client: TestClient) -> dict:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    return resp


def test_sequence_openai_returns_different_responses_on_successive_calls() -> None:
    fixtures = [
        Fixture(
            name="retry",
            match=MatchCriteria(provider=Provider.OPENAI),
            sequence=[
                MockResponse(http_status=429, error_message="Rate limit exceeded."),
                MockResponse(http_status=429, error_message="Rate limit exceeded."),
                MockResponse(content="Success after retry!"),
            ],
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))

    r1 = _openai_post(client)
    assert r1.status_code == 429

    r2 = _openai_post(client)
    assert r2.status_code == 429

    r3 = _openai_post(client)
    assert r3.status_code == 200
    assert r3.json()["choices"][0]["message"]["content"] == "Success after retry!"


def test_sequence_openai_sticks_on_last_response() -> None:
    fixtures = [
        Fixture(
            name="retry",
            match=MatchCriteria(provider=Provider.OPENAI),
            sequence=[
                MockResponse(http_status=429, error_message="Rate limited"),
                MockResponse(content="Done"),
            ],
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))

    _openai_post(client)  # call 1 → 429
    r2 = _openai_post(client)  # call 2 → 200 "Done"
    r3 = _openai_post(client)  # call 3 → still "Done"
    r4 = _openai_post(client)  # call 4 → still "Done"

    assert r2.status_code == 200
    assert r3.status_code == 200
    assert r4.json()["choices"][0]["message"]["content"] == "Done"


def test_replace_fixtures_resets_sequence_counts() -> None:
    """replace_fixtures() should reset call counts so sequences restart."""
    server = MockLLMServer(
        fixtures=[
            Fixture(
                name="seq",
                match=MatchCriteria(provider=Provider.OPENAI),
                sequence=[
                    MockResponse(http_status=429, error_message="Rate limited"),
                    MockResponse(content="Done"),
                ],
            )
        ]
    )
    server.start()
    try:
        import httpx

        r1 = httpx.post(
            f"{server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r1.status_code == 429

        # Replace fixtures (same sequence) → counts should reset
        server.replace_fixtures(
            [
                Fixture(
                    name="seq",
                    match=MatchCriteria(provider=Provider.OPENAI),
                    sequence=[
                        MockResponse(http_status=429, error_message="Rate limited"),
                        MockResponse(content="Done"),
                    ],
                )
            ]
        )

        r2 = httpx.post(
            f"{server.url}/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        # Should be 429 again (counter reset)
        assert r2.status_code == 429
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Integration tests: Anthropic provider with sequences
# ---------------------------------------------------------------------------


def test_sequence_anthropic_returns_different_responses() -> None:
    fixtures = [
        Fixture(
            name="retry",
            match=MatchCriteria(provider=Provider.ANTHROPIC),
            sequence=[
                MockResponse(http_status=529, error_message="Overloaded"),
                MockResponse(content="OK"),
            ],
        )
    ]
    client = TestClient(create_app(fixtures=fixtures))

    r1 = client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r1.status_code == 529

    r2 = client.post(
        "/v1/messages",
        json={
            "model": "claude-opus-4-6",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r2.status_code == 200
    assert r2.json()["content"][0]["text"] == "OK"
