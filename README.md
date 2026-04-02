# stubllm

**Deterministic mock server for LLM APIs. Test your AI code without spending tokens.**

[![CI](https://github.com/airupt/stubllm/actions/workflows/ci.yml/badge.svg)](https://github.com/airupt/stubllm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/stubllm)](https://pypi.org/project/stubllm/)
[![Python](https://img.shields.io/pypi/pyversions/stubllm)](https://pypi.org/project/stubllm/)
[![GitHub Stars](https://img.shields.io/github/stars/airupt/stubllm?style=social)](https://github.com/airupt/stubllm/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Works with **OpenAI** · **Anthropic** · **Google Gemini**

---

## Why stubllm?

| | stubllm | Real API | Ollama |
|---|---|---|---|
| Cost | Free | Paid | Free |
| Speed | <1ms | 1–30s | 5–30s |
| Deterministic | ✅ | ❌ | ❌ |
| Works offline | ✅ | ❌ | ✅ |
| No GPU needed | ✅ | ✅ | ❌ |
| pytest integration | ✅ | ❌ | ❌ |
| Fixtures & record-replay | ✅ | ❌ | ❌ |
| Error injection | ✅ | ❌ | ❌ |
| Response sequences | ✅ | ❌ | ❌ |
| CI-friendly | ✅ | Slow/expensive | Heavy |

---

## Install

```bash
pip install stubllm               # server + CLI
pip install "stubllm[pytest]"     # + pytest plugin
```

---

## Quickstart

**1. Create a fixture file**

```yaml
# fixtures/chat.yaml
fixtures:
  - name: greeting
    match:
      provider: openai
      messages:
        - role: user
          content:
            contains: "hello"
    response:
      content: "Hello! How can I help you today?"
```

**2. Start the server**

```bash
stubllm serve --fixture-dir ./fixtures
```

**3. Point your app at it**

```bash
export OPENAI_BASE_URL=http://localhost:8765/v1/
python your_app.py   # no real API calls, no tokens spent
```

---

## Fixtures

### Basic response

```yaml
fixtures:
  - name: greeting
    match:
      provider: openai          # openai | anthropic | gemini | any
      model: "gpt-4o"           # optional
      messages:
        - role: user
          content:
            contains: "hello"   # exact | contains | regex
    response:
      content: "Hello! How can I help you today?"
      usage:
        prompt_tokens: 10
        completion_tokens: 12
        total_tokens: 22
```

### Match strategies

```yaml
content:
  exact: "Hello, world!"     # highest priority — case-sensitive full match
  contains: "weather"        # substring — case-insensitive
  regex: "tell me.*joke"     # regular expression
```

### Matching priority

When multiple fixtures match, the most specific wins:

| Criteria | Score |
|---|---|
| `exact` message content | +10 |
| `contains` message content | +5 |
| `regex` message content | +4 |
| `model` specified | +2 |
| `tools_present` specified | +2 |
| `provider` specified | +1 |
| Fallback (no criteria) | 0 |

### Tool call response

```yaml
fixtures:
  - name: weather_tool
    match:
      provider: openai
      messages:
        - role: user
          content:
            contains: "weather"
      tools_present: true        # only match when tools are provided
    response:
      tool_calls:
        - id: "call_abc123"
          type: function
          function:
            name: "get_weather"
            arguments: '{"location": "Amsterdam"}'
```

### Error injection

Simulate rate limits, 500s, or any HTTP error. Each provider gets its native error format automatically.

```yaml
fixtures:
  - name: rate_limit
    match:
      provider: openai
      messages:
        - role: user
          content:
            contains: "expensive query"
    response:
      http_status: 429
      error_message: "Rate limit exceeded. Please retry after 60 seconds."
      error_code: "rate_limit_exceeded"   # optional — defaults derived from http_status
```

| `http_status` | OpenAI `type` | Anthropic `type` | Gemini `status` |
|---|---|---|---|
| 429 | `rate_limit_error` | `rate_limit_error` | `RESOURCE_EXHAUSTED` |
| 500 | `server_error` | `api_error` | `INTERNAL` |
| 503 | `server_error` | `api_error` | `UNAVAILABLE` |
| 401 | `authentication_error` | `authentication_error` | `UNAUTHENTICATED` |

### Response sequences

Return different responses on successive calls to the same fixture. The last entry repeats once the sequence is exhausted.

```yaml
fixtures:
  - name: retry_scenario
    match:
      provider: openai
    sequence:
      - http_status: 429
        error_message: "Rate limit exceeded."
      - http_status: 429
        error_message: "Rate limit exceeded."
      - content: "Success after retry!"
```

Call 1 → 429, call 2 → 429, call 3+ → success. Sequence counters reset when `replace_fixtures()` or `reset()` is called.

### Streaming

Streaming is controlled by the `stream: true` parameter in the request — fixtures work identically for both streaming and non-streaming calls. Control chunk speed in the fixture:

```yaml
response:
  content: "A long streaming response..."
  stream_chunk_delay_ms: 20   # default: 20ms between chunks
```

### Structured output

When `response_format: { type: "json_schema" }` is set, stubllm validates that the fixture response is valid JSON:

```yaml
fixtures:
  - name: structured
    match:
      provider: openai
    response:
      content: '{"name": "Alice", "age": 30}'
```

---

## Providers

### OpenAI

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8765/v1/",  # /v1/ required — the SDK does not add it
    api_key="test-key"
)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "hello"}]
)
```

### Anthropic

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://localhost:8765",  # SDK adds /v1/ itself
    api_key="test-key"
)
message = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "hello"}]
)
```

### Google Gemini

```python
from google import genai  # pip install google-genai

client = genai.Client(
    api_key="test-key",
    http_options={"base_url": "http://localhost:8765"},
)
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="hello",
)
```

---

## pytest plugin

The plugin registers automatically after `pip install "stubllm[pytest]"` — no `conftest.py` needed.

### Basic test

```python
import openai
from stubllm.pytest_plugin import use_fixtures

@use_fixtures("fixtures/chat.yaml")
def test_greeting(stubllm_server):
    client = openai.OpenAI(
        base_url=stubllm_server.openai_url,  # includes /v1/
        api_key="test-key"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello world"}]
    )
    assert "Hello" in response.choices[0].message.content
    assert stubllm_server.call_count == 1
```

### Assertions

```python
# Prompt content
stubllm_server.assert_called_with_prompt("hello")
stubllm_server.assert_called_with_prompt("Hello", case_sensitive=True)

# Call counts
stubllm_server.assert_called_once()
stubllm_server.assert_called_n_times(3)
stubllm_server.assert_not_called()

# Model and path
stubllm_server.assert_model_was("gpt-4o")
stubllm_server.assert_last_call_path("/v1/chat/completions")

# Fixture hits
stubllm_server.assert_fixture_hit("my_fixture")
stubllm_server.assert_fixture_hit("my_fixture", times=2)

# Raw access
assert stubllm_server.call_count == 2
for call in stubllm_server.calls:
    print(call["path"], call["body"])
```

### Runtime fixture management

```python
from stubllm import Fixture, MockResponse

@use_fixtures("base.yaml")
def test_dynamic(stubllm_server):
    # Replace all fixtures (resets sequence counters)
    stubllm_server.replace_fixtures([
        Fixture(name="new", response=MockResponse(content="replaced"))
    ])

    # Append without replacing
    stubllm_server.add_fixtures([
        Fixture(name="extra", response=MockResponse(content="extra"))
    ])

    # Reset call log and sequence counters between steps
    stubllm_server.reset()
```

### Multiple fixture files

```python
@use_fixtures("fixtures/chat.yaml", "fixtures/tools.yaml")
def test_combined(stubllm_server):
    ...
```

---

## CLI

```bash
# Start server — auto-loads ./fixtures/ if it exists
stubllm serve

# Custom port and fixture directory
stubllm serve --port 9000 --fixture-dir ./my-fixtures

# Multiple fixture directories or individual files
stubllm serve --fixture-dir ./fixtures/openai --fixture-dir ./fixtures/anthropic
stubllm serve --fixture-file chat.yaml --fixture-file tools.yaml

# Record mode — proxy real API calls and save as fixtures
stubllm record --target https://api.openai.com --fixture-dir ./recorded

# Version
stubllm --version
```

---

## Record & replay

Record real API interactions once, replay them forever:

```bash
# 1. Start in record mode
stubllm record --target https://api.openai.com --fixture-dir ./recorded_fixtures

# 2. Run your app — calls are proxied to OpenAI and saved locally
OPENAI_BASE_URL=http://localhost:8765/v1/ python your_app.py

# 3. Commit the recorded fixtures (API keys are stripped automatically)
git add recorded_fixtures/
```

Future runs use the local fixtures — no network, no tokens.

---

## Server endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Returns `{"status": "ok"}` — for readiness checks |
| `GET /` | Server info: version, fixture count, available providers |
| `GET /_fixtures` | Lists all loaded fixtures (name, provider, model, endpoint) |
| `GET /_stats` | Per-fixture call counts since last reset |

```bash
curl http://localhost:8765/_fixtures
curl http://localhost:8765/_stats
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT

---

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=airupt/stubllm&type=Date)](https://star-history.com/#airupt/stubllm&Date)
