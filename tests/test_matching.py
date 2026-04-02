"""Tests for the fixture matching engine."""

from __future__ import annotations

from stubllm.fixtures.matcher import FixtureMatcher
from stubllm.fixtures.models import (
    ContentMatch,
    Fixture,
    MatchCriteria,
    MessageMatch,
    MockResponse,
    Provider,
)


def _make_fixture(name: str, **match_kwargs: object) -> Fixture:
    return Fixture(
        name=name,
        match=MatchCriteria(**match_kwargs),  # type: ignore[arg-type]
        response=MockResponse(content=f"response from {name}"),
    )


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


class TestFixtureMatcher:
    def test_exact_match_wins_over_contains(self) -> None:
        fixtures = [
            _make_fixture(
                "broad",
                messages=[MessageMatch(content=ContentMatch(contains="hello"))],
            ),
            _make_fixture(
                "exact",
                messages=[MessageMatch(content=ContentMatch(exact="hello world"))],
            ),
        ]
        matcher = FixtureMatcher(fixtures)
        resp, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[_msg("user", "hello world")],
        )
        assert name == "exact"

    def test_fallback_when_no_match(self) -> None:
        matcher = FixtureMatcher([])
        resp, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[_msg("user", "something unmatched")],
        )
        assert name == "__fallback__"
        assert "No fixture matched" in (resp.get_response(0).content or "")

    def test_provider_mismatch_skipped(self) -> None:
        fixtures = [
            _make_fixture("anthropic_only", provider=Provider.ANTHROPIC),
        ]
        matcher = FixtureMatcher(fixtures)
        resp, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[_msg("user", "hello")],
        )
        assert name == "__fallback__"

    def test_tools_present_match(self) -> None:
        fixtures = [
            _make_fixture("with_tools", tools_present=True),
            _make_fixture("without_tools", tools_present=False),
        ]
        matcher = FixtureMatcher(fixtures)
        resp, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[_msg("user", "weather")],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        assert name == "with_tools"

    def test_model_match(self) -> None:
        fixtures = [
            _make_fixture("gpt4", model="gpt-4o"),
            _make_fixture("gpt35", model="gpt-3.5-turbo"),
        ]
        matcher = FixtureMatcher(fixtures)
        resp, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[_msg("user", "hello")],
            model="gpt-3.5-turbo",
        )
        assert name == "gpt35"

    def test_most_specific_wins(self) -> None:
        fixtures = [
            _make_fixture(
                "specific",
                provider=Provider.OPENAI,
                model="gpt-4o",
                messages=[MessageMatch(content=ContentMatch(contains="weather"))],
                tools_present=True,
            ),
            _make_fixture(
                "generic",
                messages=[MessageMatch(content=ContentMatch(contains="weather"))],
            ),
        ]
        matcher = FixtureMatcher(fixtures)
        resp, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[_msg("user", "what's the weather?")],
            model="gpt-4o",
            tools=[{"type": "function"}],
        )
        assert name == "specific"

    def test_fallback_response_customizable(self) -> None:
        fallback = MockResponse(content="custom fallback", http_status=404)
        matcher = FixtureMatcher([], fallback_response=fallback)
        resp, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[_msg("user", "hello")],
        )
        assert name == "__fallback__"

    def test_regex_match(self) -> None:
        fixtures = [
            _make_fixture(
                "joke_fixture",
                messages=[MessageMatch(content=ContentMatch(regex=r"tell me.*joke"))],
            ),
        ]
        matcher = FixtureMatcher(fixtures)
        resp, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[_msg("user", "tell me a funny joke please")],
        )
        assert name == "joke_fixture"

    def test_endpoint_mismatch_skipped(self) -> None:
        """A fixture with the wrong endpoint is not matched."""
        fixtures = [_make_fixture("ep_only", endpoint="/v1/embeddings")]
        matcher = FixtureMatcher(fixtures)
        _, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[_msg("user", "hello")],
        )
        assert name == "__fallback__"

    def test_endpoint_match_scores(self) -> None:
        """Endpoint match adds to score and is preferred over no-endpoint fixture."""
        fixtures = [
            _make_fixture("with_endpoint", endpoint="/v1/chat/completions"),
            _make_fixture("any_endpoint"),
        ]
        matcher = FixtureMatcher(fixtures)
        _, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[_msg("user", "hello")],
        )
        assert name == "with_endpoint"

    def test_headers_match(self) -> None:
        """Headers criteria must all match."""
        fixtures = [
            _make_fixture("with_header", headers={"x-tenant": "acme"}),
        ]
        matcher = FixtureMatcher(fixtures)
        _, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[_msg("user", "hello")],
            headers={"x-tenant": "acme", "content-type": "application/json"},
        )
        assert name == "with_header"

    def test_headers_mismatch_skipped(self) -> None:
        fixtures = [_make_fixture("strict_header", headers={"x-tenant": "acme"})]
        matcher = FixtureMatcher(fixtures)
        _, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[_msg("user", "hello")],
            headers={"x-tenant": "other"},
        )
        assert name == "__fallback__"

    def test_headers_none_when_criteria_requires(self) -> None:
        """Passing headers=None when criteria specifies headers → no match."""
        fixtures = [_make_fixture("needs_header", headers={"x-api": "v1"})]
        matcher = FixtureMatcher(fixtures)
        _, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[_msg("user", "hello")],
            headers=None,
        )
        assert name == "__fallback__"

    def test_message_role_only_match(self) -> None:
        """MessageMatch with only a role (no content) still matches and scores +1."""
        fixtures = [
            Fixture(
                name="role_only",
                match=MatchCriteria(
                    messages=[MessageMatch(role="system")]
                ),
                response=MockResponse(content="role matched"),
            )
        ]
        matcher = FixtureMatcher(fixtures)
        _, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[
                _msg("system", "you are an assistant"),
                _msg("user", "hello"),
            ],
        )
        assert name == "role_only"

    def test_message_str_content_match(self) -> None:
        """MessageMatch with a plain string content matches and scores +3."""
        fixtures = [
            Fixture(
                name="str_content",
                match=MatchCriteria(messages=[MessageMatch(content="hello")]),
                response=MockResponse(content="str matched"),
            )
        ]
        matcher = FixtureMatcher(fixtures)
        _, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[_msg("user", "hello world")],
        )
        assert name == "str_content"

    def test_multiple_message_criteria(self) -> None:
        fixtures = [
            Fixture(
                name="multi",
                match=MatchCriteria(
                    messages=[
                        MessageMatch(role="system", content=ContentMatch(contains="assistant")),
                        MessageMatch(role="user", content=ContentMatch(contains="hello")),
                    ]
                ),
                response=MockResponse(content="matched multi"),
            )
        ]
        matcher = FixtureMatcher(fixtures)
        resp, name = matcher.match(
            provider=Provider.OPENAI,
            endpoint="/v1/chat/completions",
            messages=[
                _msg("system", "You are a helpful assistant"),
                _msg("user", "hello there"),
            ],
        )
        assert name == "multi"
