"""Basic example: using stubllm with the OpenAI client.

Run with:
    cd examples/basic
    pytest test_example.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

# These imports require openai to be installed:
#   pip install openai
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from stubllm.pytest_plugin import use_fixtures

FIXTURES = Path(__file__).parent / "fixtures" / "chat.yaml"


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai package not installed")
@use_fixtures(FIXTURES)
def test_greeting(stubllm_server: object) -> None:
    """Test that a greeting message returns the expected mock response."""
    from stubllm.pytest_plugin import MockLLMServerFixture

    assert isinstance(stubllm_server, MockLLMServerFixture)

    client = openai.OpenAI(
        base_url=stubllm_server.openai_url,
        api_key="test-key",
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello world"}],
    )
    assert "Hello" in response.choices[0].message.content  # type: ignore[operator]
    assert stubllm_server.call_count == 1


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai package not installed")
@use_fixtures(FIXTURES)
def test_tool_calls(stubllm_server: object) -> None:
    """Test that weather queries return the expected tool call mock."""
    from stubllm.pytest_plugin import MockLLMServerFixture

    assert isinstance(stubllm_server, MockLLMServerFixture)

    client = openai.OpenAI(
        base_url=stubllm_server.openai_url,
        api_key="test-key",
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather in Amsterdam?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ],
    )
    tool_call = response.choices[0].message.tool_calls[0]  # type: ignore[index]
    assert tool_call.function.name == "get_weather"
    assert stubllm_server.call_count == 1
