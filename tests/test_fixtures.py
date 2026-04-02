"""Tests for fixture loading and models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from stubllm.fixtures.loader import FixtureLoader
from stubllm.fixtures.models import (
    ContentMatch,
    MatchCriteria,
    MessageMatch,
    MockResponse,
    Provider,
)


class TestContentMatch:
    def test_exact_match(self) -> None:
        cm = ContentMatch(exact="hello world")
        assert cm.matches("hello world")
        assert not cm.matches("hello")

    def test_contains_match_case_insensitive(self) -> None:
        cm = ContentMatch(contains="weather")
        assert cm.matches("What's the weather today?")
        assert cm.matches("WEATHER forecast")
        assert not cm.matches("sunshine")

    def test_regex_match(self) -> None:
        cm = ContentMatch(regex=r"tell me.*joke")
        assert cm.matches("tell me a joke please")
        assert cm.matches("Tell Me A Joke")
        assert not cm.matches("tell me a story")

    def test_requires_at_least_one_field(self) -> None:
        with pytest.raises(ValueError):
            ContentMatch()


class TestMessageMatch:
    def test_role_match(self) -> None:
        mm = MessageMatch(role="user")
        assert mm.matches({"role": "user", "content": "hello"})
        assert not mm.matches({"role": "assistant", "content": "hello"})

    def test_content_string_match(self) -> None:
        mm = MessageMatch(content="hello")
        assert mm.matches({"role": "user", "content": "say hello world"})
        assert not mm.matches({"role": "user", "content": "goodbye"})

    def test_content_object_match(self) -> None:
        mm = MessageMatch(content=ContentMatch(contains="weather"))
        assert mm.matches({"role": "user", "content": "what is the weather?"})

    def test_list_content(self) -> None:
        mm = MessageMatch(content=ContentMatch(contains="hello"))
        msg = {"role": "user", "content": [{"type": "text", "text": "hello world"}]}
        assert mm.matches(msg)


class TestMatchCriteria:
    def test_specificity_score_increases_with_constraints(self) -> None:
        empty = MatchCriteria()
        with_provider = MatchCriteria(provider=Provider.OPENAI)
        with_exact_msg = MatchCriteria(
            provider=Provider.OPENAI,
            messages=[MessageMatch(content=ContentMatch(exact="hello"))],
        )
        assert empty.specificity_score() < with_provider.specificity_score()
        assert with_provider.specificity_score() < with_exact_msg.specificity_score()

    def test_specificity_score_endpoint_and_model(self) -> None:
        with_endpoint = MatchCriteria(endpoint="/v1/chat/completions")
        with_model = MatchCriteria(model="gpt-4o")
        base = MatchCriteria()
        assert with_endpoint.specificity_score() > base.specificity_score()
        assert with_model.specificity_score() > base.specificity_score()
        # model scores higher than endpoint
        assert with_model.specificity_score() > with_endpoint.specificity_score()

    def test_specificity_score_message_contains(self) -> None:
        c = MatchCriteria(messages=[MessageMatch(content=ContentMatch(contains="hi"))])
        assert c.specificity_score() > 0

    def test_specificity_score_message_regex(self) -> None:
        c = MatchCriteria(messages=[MessageMatch(content=ContentMatch(regex=r"hello.*"))])
        assert c.specificity_score() > 0

    def test_specificity_score_message_str_content(self) -> None:
        c = MatchCriteria(messages=[MessageMatch(content="hello")])
        assert c.specificity_score() > 0

    def test_specificity_score_tools_present(self) -> None:
        c = MatchCriteria(tools_present=True)
        assert c.specificity_score() == 2

    def test_specificity_score_headers(self) -> None:
        c = MatchCriteria(headers={"x-custom": "val1", "x-other": "val2"})
        assert c.specificity_score() == 2  # len(headers) == 2


class TestMockResponse:
    def test_default_content_when_none(self) -> None:
        r = MockResponse()
        assert r.content == "Mock response."

    def test_tool_calls_suppress_default_content(self) -> None:
        from stubllm.fixtures.models import ToolCallResponse

        r = MockResponse(tool_calls=[ToolCallResponse(function={"name": "foo", "arguments": "{}"})])
        assert r.content is None


class TestFixtureLoader:
    def test_load_yaml_list_format(self, tmp_path: Path) -> None:
        data = [
            {
                "name": "greet",
                "match": {"provider": "openai"},
                "response": {"content": "Hello!"},
            }
        ]
        f = tmp_path / "fixtures.yaml"
        f.write_text(yaml.dump(data))
        loader = FixtureLoader()
        fixtures = loader.load_file(f)
        assert len(fixtures) == 1
        assert fixtures[0].name == "greet"
        assert fixtures[0].response.content == "Hello!"

    def test_load_yaml_dict_format(self, tmp_path: Path) -> None:
        data = {
            "fixtures": [
                {"name": "greet", "response": {"content": "Hi"}},
                {"name": "bye", "response": {"content": "Bye"}},
            ]
        }
        f = tmp_path / "fixtures.yaml"
        f.write_text(yaml.dump(data))
        loader = FixtureLoader()
        fixtures = loader.load_file(f)
        assert len(fixtures) == 2

    def test_load_json(self, tmp_path: Path) -> None:
        data = {"fixtures": [{"name": "test", "response": {"content": "ok"}}]}
        f = tmp_path / "fixtures.json"
        f.write_text(json.dumps(data))
        loader = FixtureLoader()
        fixtures = loader.load_file(f)
        assert len(fixtures) == 1

    def test_load_directory(self, tmp_path: Path) -> None:
        for i in range(3):
            data = [{"name": f"fixture_{i}", "response": {"content": f"response {i}"}}]
            (tmp_path / f"fix_{i}.yaml").write_text(yaml.dump(data))
        loader = FixtureLoader()
        fixtures = loader.load_directory(tmp_path)
        assert len(fixtures) == 3

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "fixtures.txt"
        f.write_text("not a fixture")
        loader = FixtureLoader()
        with pytest.raises(ValueError, match="Unsupported fixture"):
            loader.load_file(f)

    def test_invalid_fixture_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text(yaml.dump([{"response": {"content": "ok", "http_status": "not_an_int"}}]))
        loader = FixtureLoader()
        with pytest.raises(ValueError):
            loader.load_file(f)
