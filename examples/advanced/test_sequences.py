"""Example: testing retry logic with response sequences.

The fixture returns 429 twice, then succeeds on the third call — letting you
test exponential backoff and retry handlers without a real API.

Run with:
    cd examples/advanced
    pytest test_sequences.py -v
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from stubllm.pytest_plugin import use_fixtures

SEQUENCES_FIXTURES = Path(__file__).parent / "fixtures" / "sequences.yaml"


def _client_with_retry(base_url: str, max_retries: int = 3) -> openai.OpenAI:
    """OpenAI client configured with automatic retries on 429."""
    return openai.OpenAI(
        base_url=base_url,
        api_key="test-key",
        max_retries=max_retries,
    )


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai package not installed")
@use_fixtures(SEQUENCES_FIXTURES)
def test_client_retries_on_rate_limit(stubllm_server: object) -> None:
    """The OpenAI SDK's built-in retry logic recovers after two 429 responses."""
    from stubllm.pytest_plugin import MockLLMServerFixture

    assert isinstance(stubllm_server, MockLLMServerFixture)

    client = _client_with_retry(stubllm_server.openai_url, max_retries=3)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "please retry this"}],
    )

    assert response.choices[0].message.content == "Success after retry!"
    # 3 calls: two 429s + one 200
    assert stubllm_server.call_count == 3


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="openai package not installed")
@use_fixtures(SEQUENCES_FIXTURES)
def test_manual_retry_loop(stubllm_server: object) -> None:
    """Manual retry loop: fail fast client, user-space retry until success."""
    from stubllm.pytest_plugin import MockLLMServerFixture

    assert isinstance(stubllm_server, MockLLMServerFixture)

    # Client with no retries so we can drive the loop ourselves
    client = openai.OpenAI(
        base_url=stubllm_server.openai_url,
        api_key="test-key",
        max_retries=0,
    )

    last_response = None
    for attempt in range(5):
        try:
            last_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "please retry this"}],
            )
            break  # success
        except openai.RateLimitError:
            time.sleep(0.01)  # tiny delay in the example; real code would back off

    assert last_response is not None
    assert last_response.choices[0].message.content == "Success after retry!"
    assert stubllm_server.call_count == 3
