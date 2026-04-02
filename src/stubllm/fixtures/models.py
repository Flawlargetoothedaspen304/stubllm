"""Pydantic v2 models for fixture definitions."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Provider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    ANY = "any"


class ContentMatch(BaseModel):
    """Flexible content matching strategies."""

    exact: str | None = None
    contains: str | None = None
    regex: str | None = None

    @model_validator(mode="after")
    def at_least_one_strategy(self) -> ContentMatch:
        if self.exact is None and self.contains is None and self.regex is None:
            raise ValueError("ContentMatch requires at least one of: exact, contains, regex")
        return self

    def matches(self, text: str) -> bool:
        """Return True if text satisfies this match criterion."""
        if self.exact is not None:
            return text == self.exact
        if self.contains is not None:
            return self.contains.lower() in text.lower()
        if self.regex is not None:
            return bool(re.search(self.regex, text, re.IGNORECASE | re.DOTALL))
        return False  # pragma: no cover


class MessageMatch(BaseModel):
    """Match criteria for a single message in the conversation."""

    role: str | None = None
    content: ContentMatch | str | None = None

    def matches(self, message: dict[str, Any]) -> bool:
        """Return True if message satisfies this match criterion."""
        if self.role is not None and message.get("role") != self.role:
            return False
        if self.content is not None:
            text = message.get("content", "")
            if isinstance(text, list):
                # Handle Anthropic-style content blocks
                text = " ".join(
                    block.get("text", "") for block in text if isinstance(block, dict)
                )
            if isinstance(self.content, str):
                return self.content.lower() in str(text).lower()
            return self.content.matches(str(text))
        return True


class MatchCriteria(BaseModel):
    """Criteria used to match an incoming request to a fixture."""

    provider: Provider | None = None
    endpoint: str | None = None
    model: str | None = None
    messages: list[MessageMatch] | None = None
    tools_present: bool | None = None
    headers: dict[str, str] | None = None

    def specificity_score(self) -> int:
        """Higher score = more specific = higher priority in matching."""
        score = 0
        if self.provider and self.provider != Provider.ANY:
            score += 1
        if self.endpoint:
            score += 1
        if self.model:
            score += 2
        if self.messages:
            for msg in self.messages:
                if msg.content:
                    if isinstance(msg.content, ContentMatch) and msg.content.exact:
                        score += 10  # exact match is very specific
                    elif isinstance(msg.content, ContentMatch) and msg.content.contains:
                        score += 5
                    elif isinstance(msg.content, ContentMatch) and msg.content.regex:
                        score += 4
                    else:
                        score += 3
        if self.tools_present is not None:
            score += 2
        if self.headers:
            score += len(self.headers)
        return score


class ToolCallResponse(BaseModel):
    """Represents a tool call in the response."""

    id: str = "call_mock_001"
    type: str = "function"
    function: dict[str, Any] = Field(default_factory=dict)


class MockResponse(BaseModel):
    """Provider-agnostic normalized response."""

    content: str | None = None
    tool_calls: list[ToolCallResponse] | None = None
    model: str = "mock-model"
    finish_reason: str = "stop"
    usage: dict[str, int] = Field(
        default_factory=lambda: {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    )
    latency_ms: int = 0
    stream_chunk_delay_ms: int = 20
    http_status: int = 200

    @model_validator(mode="after")
    def content_or_tool_calls(self) -> MockResponse:
        if self.content is None and self.tool_calls is None:
            self.content = "Mock response."
        return self


class Fixture(BaseModel):
    """A single request→response mapping."""

    name: str = "unnamed"
    match: MatchCriteria = Field(default_factory=MatchCriteria)
    response: MockResponse = Field(default_factory=MockResponse)


class FixtureFile(BaseModel):
    """Top-level fixture file structure (supports both list and dict formats)."""

    fixtures: list[Fixture] = Field(default_factory=list)
