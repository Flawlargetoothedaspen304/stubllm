"""Request-to-fixture matching engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stubllm.fixtures.models import Fixture, MatchCriteria, MessageMatch, MockResponse, Provider


@dataclass
class MatchResult:
    """Result of a fixture match attempt."""

    fixture: Fixture
    score: int
    matched: bool = True


class FixtureMatcher:
    """Stateless matcher: given a request and a list of fixtures, returns the best match."""

    def __init__(
        self, fixtures: list[Fixture], fallback_response: MockResponse | None = None
    ) -> None:
        self._fixtures = fixtures
        self._fallback = fallback_response or MockResponse(
            content="No fixture matched this request.",
            http_status=200,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        *,
        provider: Provider,
        endpoint: str,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[MockResponse, str]:
        """Return (response, fixture_name) for the best-matching fixture.

        Falls back to the default fallback if nothing matches.
        """
        best: MatchResult | None = None

        for fixture in self._fixtures:
            score = self._score(
                fixture.match,
                provider=provider,
                endpoint=endpoint,
                messages=messages,
                model=model,
                tools=tools,
                headers=headers,
            )
            if score is None:
                continue  # did not match
            if best is None or score > best.score:
                best = MatchResult(fixture=fixture, score=score)

        if best is not None:
            return best.fixture.response, best.fixture.name

        available = ", ".join(repr(f.name) for f in self._fixtures) or "none"
        last_user = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "<no user message>",
        )
        self._fallback.content = (
            f"No fixture matched for prompt {str(last_user)[:80]!r}. "
            f"Available fixtures: {available}"
        )
        return self._fallback, "__fallback__"

    # ------------------------------------------------------------------
    # Private scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        criteria: MatchCriteria,
        *,
        provider: Provider,
        endpoint: str,
        messages: list[dict[str, Any]],
        model: str | None,
        tools: list[dict[str, Any]] | None,
        headers: dict[str, str] | None,
    ) -> int | None:
        """Return a match score or None if the criteria don't match."""
        score = 0

        # Provider check
        if criteria.provider and criteria.provider != Provider.ANY:
            if criteria.provider != provider:
                return None
            score += 1

        # Endpoint check
        if criteria.endpoint and criteria.endpoint != endpoint:
            return None
        if criteria.endpoint:
            score += 1

        # Model check
        if criteria.model:
            if model is None or model != criteria.model:
                return None
            score += 2

        # Messages check
        if criteria.messages:
            msg_score = self._match_messages(criteria.messages, messages)
            if msg_score is None:
                return None
            score += msg_score

        # Tools check
        if criteria.tools_present is not None:
            has_tools = bool(tools)
            if criteria.tools_present != has_tools:
                return None
            score += 2

        # Headers check
        if criteria.headers:
            if headers is None:
                return None
            for key, val in criteria.headers.items():
                if headers.get(key) != val:
                    return None
            score += len(criteria.headers)

        return score

    def _match_messages(
        self,
        criteria: list[MessageMatch],
        messages: list[dict[str, Any]],
    ) -> int | None:
        """Match message criteria against the conversation messages.

        Returns a score reflecting specificity, or None if no match.
        """
        score = 0
        for msg_crit in criteria:
            matched_any = False
            for msg in messages:
                if msg_crit.matches(msg):
                    matched_any = True
                    # Compute per-criterion score
                    if msg_crit.content is not None:
                        if hasattr(msg_crit.content, "exact") and msg_crit.content.exact:
                            score += 10
                        elif hasattr(msg_crit.content, "contains") and msg_crit.content.contains:
                            score += 5
                        elif hasattr(msg_crit.content, "regex") and msg_crit.content.regex:
                            score += 4
                        else:
                            score += 3
                    else:
                        score += 1
                    break
            if not matched_any:
                return None
        return score
