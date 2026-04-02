"""Advanced examples: multi-provider, streaming, structured output.

Run with:
    cd examples/advanced
    pytest test_advanced.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from stubllm.pytest_plugin import use_fixtures

TOOLS_FIXTURES = Path(__file__).parent / "fixtures" / "tools.yaml"
STREAMING_FIXTURES = Path(__file__).parent / "fixtures" / "streaming.yaml"


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai package not installed")
@use_fixtures(TOOLS_FIXTURES)
def test_tool_with_regex_match(stubllm_server: object) -> None:
    from stubllm.pytest_plugin import MockLLMServerFixture

    assert isinstance(stubllm_server, MockLLMServerFixture)

    client = openai.OpenAI(base_url=stubllm_server.openai_url, api_key="test")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "please search for Python tutorials"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    assert response.choices[0].message.tool_calls is not None
    assert response.choices[0].message.tool_calls[0].function.name == "web_search"


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai package not installed")
@use_fixtures(STREAMING_FIXTURES)
def test_streaming_response(stubllm_server: object) -> None:
    from stubllm.pytest_plugin import MockLLMServerFixture

    assert isinstance(stubllm_server, MockLLMServerFixture)

    client = openai.OpenAI(base_url=stubllm_server.openai_url, api_key="test")
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "tell me a story"}],
        stream=True,
    )
    collected = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            collected += delta
    assert "Once upon a time" in collected


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai package not installed")
@use_fixtures(STREAMING_FIXTURES, TOOLS_FIXTURES)
def test_multiple_fixture_files_loaded(stubllm_server: object) -> None:
    """Multiple fixture files can be combined with @use_fixtures."""
    from stubllm.pytest_plugin import MockLLMServerFixture

    assert isinstance(stubllm_server, MockLLMServerFixture)

    client = openai.OpenAI(base_url=stubllm_server.openai_url, api_key="test")

    # Test story fixture (from streaming.yaml)
    r1 = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "tell me a story"}],
    )
    assert "Once upon a time" in r1.choices[0].message.content  # type: ignore[operator]

    # Test tool fixture (from tools.yaml)
    r2 = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "search for something"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    assert r2.choices[0].message.tool_calls is not None
