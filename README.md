# stubllm

**Deterministic mock server for LLM APIs. Test your AI code without spending tokens.**

[![CI](https://github.com/airupt/stubllm/actions/workflows/ci.yml/badge.svg)](https://github.com/airupt/stubllm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/stubllm)](https://pypi.org/project/stubllm/)
[![Python](https://img.shields.io/pypi/pyversions/stubllm)](https://pypi.org/project/stubllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Works with: **OpenAI** · **Anthropic** · **Google Gemini**

---

## 30-second quickstart

```bash
# 1. Install
pip install stubllm

# 2. Create a fixture
mkdir fixtures
cat > fixtures/chat.yaml << 'EOF'
fixtures:
  - name: "greeting"
    match:
      provider: openai
      messages:
        - role: user
          content:
            contains: "hello"
    response:
      content: "Hello! How can I help you today?"
EOF

# 3. Start the server
stubllm serve --port 8765

# 4. Point your code at it
export OPENAI_BASE_URL=http://localhost:8765/v1/
python your_app.py  # no real API calls, no tokens spent
```

---

## Why stubllm?

| | stubllm | Real API | Ollama |
|---|---|---|---|
| Cost | Free | Paid | Free |
| Speed | <1ms | 1-30s | 5-30s |
| Deterministic | ✅ | ❌ | ❌ |
| Works offline | ✅ | ❌ | ✅ |
| No GPU needed | ✅ | ✅ | ❌ |
| Pytest integration | ✅ | ❌ | ❌ |
| Fixtures / record-replay | ✅ | ❌ | ❌ |
| Error injection | ✅ | ❌ | ❌ |
| CI-friendly | ✅ | Slow/expensive | Heavy |

---

## Installation

```bash
pip install stubllm

# With pytest support
pip install "stubllm[pytest]"
```

---

## Fixture format

Fixtures are YAML (or JSON) files that map request patterns to responses.

### Basic text response

```yaml
fixtures:
  - name: "greeting"
    match:
      provider: openai          # openai | anthropic | gemini | any
      endpoint: /v1/chat/completions   # optional
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

### Tool call response

```yaml
fixtures:
  - name: "weather_tool"
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

### Streaming with delay

```yaml
fixtures:
  - name: "slow_story"
    match:
      messages:
        - role: user
          content:
            contains: "story"
    response:
      content: "Once upon a time..."
      stream_chunk_delay_ms: 50   # simulate realistic streaming speed
```

### Error injection

Simulate rate limits, 500s, or any HTTP error. Each provider gets its native error format automatically.

```yaml
fixtures:
  - name: "rate_limit"
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

Use this to test retry logic, fallback behaviour, and error handling without ever hitting a real API.

### Content match strategies

```yaml
# Exact match (highest priority)
content:
  exact: "Hello, world!"

# Substring match (case-insensitive)
content:
  contains: "weather"

# Regular expression
content:
  regex: "tell me.*joke"
```

### Matching priority

Higher specificity = higher priority. When multiple fixtures match, the most specific wins:

1. `exact` message content (score: +10)
2. `contains` message content (score: +5)
3. `regex` message content (score: +4)
4. `model` specified (score: +2)
5. `tools_present` specified (score: +2)
6. `provider` specified (score: +1)
7. Fallback (no match criteria)

---

## Pytest plugin

### Basic setup

```python
# conftest.py — nothing needed, stubllm auto-registers as a pytest plugin
# The `stubllm_server` fixture is available automatically after installing stubllm
```

```python
# test_my_app.py
import openai
from stubllm.pytest_plugin import use_fixtures

@use_fixtures("fixtures/chat.yaml")
def test_greeting(stubllm_server):
    client = openai.OpenAI(
        base_url=stubllm_server.openai_url,  # includes /v1/ — openai SDK needs this
        api_key="test-key"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello world"}]
    )
    assert "Hello" in response.choices[0].message.content
    assert stubllm_server.call_count == 1
```

### Assertion helpers

```python
def test_with_assertions(stubllm_server):
    # ... make some calls ...

    # Assert specific prompt was sent
    stubllm_server.assert_called_with_prompt("hello")

    # Assert call counts
    stubllm_server.assert_called_once()
    stubllm_server.assert_called_n_times(3)
    stubllm_server.assert_not_called()

    # Assert which model was used
    stubllm_server.assert_model_was("gpt-4o")

    # Assert last call path
    stubllm_server.assert_last_call_path("/v1/chat/completions")

    # Raw access
    assert stubllm_server.call_count == 2
    for call in stubllm_server.calls:
        print(call["path"], call["body"])
```

### Multiple fixture files

```python
@use_fixtures("fixtures/chat.yaml", "fixtures/tools.yaml")
def test_combined(stubllm_server):
    ...  # both fixture files are active for this test
```

---

## Multi-provider support

### OpenAI

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8765/v1/",  # note: /v1/ required — the OpenAI SDK does not add it
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
    base_url="http://localhost:8765",  # Anthropic SDK adds /v1/ itself
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

## Streaming

All providers support streaming. Fixtures work identically — streaming is controlled by the `stream: true` parameter in the request, not the fixture.

```python
# OpenAI streaming
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content, end="", flush=True)
```

Control streaming speed in fixtures:
```yaml
response:
  content: "A long streaming response..."
  stream_chunk_delay_ms: 20   # default: 20ms between chunks
```

---

## Record and replay

Record real API interactions for later replay:

```bash
# Start in record mode (proxies to real OpenAI, saves fixtures)
stubllm record \
  --target https://api.openai.com \
  --fixture-dir ./recorded_fixtures

# Run your app against the recording proxy
OPENAI_BASE_URL=http://localhost:8765/v1/ python your_app.py

# Fixtures are saved to ./recorded_fixtures/
ls recorded_fixtures/
# recorded_hello_world_1706000000.yaml
# recorded_weather_query_1706000001.yaml
```

Recorded fixtures are sanitized (API keys removed) and can be committed to your repo.

---

## CLI reference

```bash
# Start server (auto-loads ./fixtures/ if it exists)
stubllm serve

# Custom port and fixture directory
stubllm serve --port 9000 --fixture-dir ./my-fixtures

# Multiple fixture directories
stubllm serve --fixture-dir ./fixtures/openai --fixture-dir ./fixtures/anthropic

# Individual fixture files
stubllm serve --fixture-file chat.yaml --fixture-file tools.yaml

# Record mode
stubllm record --target https://api.openai.com --fixture-dir ./recorded

# Version
stubllm --version
```

---

## Structured output (JSON schema)

When `response_format: { type: "json_schema" }` is set, stubllm validates that the fixture response is valid JSON. If it's not, it wraps the content automatically.

```yaml
fixtures:
  - name: "structured"
    match:
      provider: openai
    response:
      content: '{"name": "Alice", "age": 30}'  # must be valid JSON
```

---

## Project structure

```
stubllm/
├── src/stubllm/
│   ├── fixtures/       # YAML/JSON loading, Pydantic models, matching engine
│   ├── providers/      # OpenAI, Anthropic, Gemini endpoint handlers
│   ├── streaming/      # SSE streaming simulation
│   ├── recorder/       # Record-and-replay proxy
│   ├── pytest_plugin/  # pytest fixtures and @use_fixtures decorator
│   ├── server.py       # FastAPI app factory
│   └── cli.py          # click CLI
├── tests/              # >80% coverage
└── examples/           # Working examples (basic + advanced)
```

---

## Contributing

```bash
git clone https://github.com/airupt/stubllm
cd stubllm
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT
