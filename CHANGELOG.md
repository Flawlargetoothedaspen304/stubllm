# Changelog

All notable changes to this project will be documented in this file.

## [0.1.5] - 2026-04-02

### Fixed
- **Gemini streaming error responses**: `streamGenerateContent` now returns proper HTTP errors (4xx/5xx) instead of silently streaming error fixtures as SSE events.
- **Thread-unsafe fallback mutation**: `FixtureMatcher` no longer mutates `self._fallback.content` in-place when no fixture matches. A new `MockResponse` object is created each time, preventing data races under concurrent load.
- **mypy strict compliance**: Resolved all `no-any-return`, `no-redef`, and `unused-ignore` errors across `loader.py`, `gemini.py`, `server.py`, and `pytest_plugin/_helpers.py`.

### Added
- **PEP 561 `py.typed` marker**: stubllm is now a typed package — type checkers (mypy, pyright) will use its inline annotations automatically.
- **CI restructured**: `lint` and `type-check` are now separate CI jobs (previously steps inside the matrix job). This fixes branch protection status checks and avoids running ruff/mypy 4× per push.
- **Publish workflow runs tests**: The publish workflow now runs lint, type-check, and tests before building, ensuring nothing broken can be shipped via a tag.
- **OIDC trusted publishing**: Automated PyPI releases via GitHub Actions — no API token stored in secrets.
- **Dependabot**: Weekly updates for pip dependencies and GitHub Actions.
- **SECURITY.md**: Vulnerability reporting policy.
- **Branch protection on main**: Force-push and deletion blocked; all CI jobs required to pass before merge.

## [0.1.4] - 2026-04-02

### Fixed
- **`reset()` now resets sequences**: `MockLLMServerFixture.reset()` (and `MockLLMServer.reset_calls()`) now clears per-fixture call counters, so sequences restart from the beginning after a reset. Previously only the call log was cleared.

### Added
- **`GET /_stats` endpoint**: Returns `{"fixture_call_counts": {...}}` — per-fixture invocation counts, useful for debugging sequence state during development.
- **`MockLLMServerFixture.assert_fixture_hit(name, times=None)`**: Assert that a specific fixture was matched. Pass `times=N` to assert an exact count.

```python
stubllm_server.assert_fixture_hit("my_fixture")        # at least once
stubllm_server.assert_fixture_hit("my_fixture", times=3)  # exactly 3 times
```

## [0.1.3] - 2026-04-02

### Added
- **Response sequences**: a single fixture can now return different responses on successive calls. Use `sequence:` instead of `response:`. After the sequence is exhausted, the last entry repeats — perfect for testing retry logic (fail twice, then succeed).

```yaml
fixtures:
  - name: "retry"
    match:
      provider: openai
    sequence:
      - http_status: 429
        error_message: "Rate limit exceeded."
      - http_status: 429
        error_message: "Rate limit exceeded."
      - content: "Success after retry!"
```

- **Error injection**: fixtures can now simulate HTTP error responses (rate limits, 500s, etc.) using `http_status`, `error_message`, and `error_code` fields. Each provider returns the correct error envelope format (OpenAI, Anthropic, Gemini).

```yaml
fixtures:
  - name: "rate_limit"
    match:
      provider: openai
    response:
      http_status: 429
      error_message: "Rate limit exceeded."
      error_code: "rate_limit_exceeded"
```

## [0.1.1] - 2026-04-02

### Fixed
- **Fixture isolation bug**: `add_fixtures()` was appending to the existing list, causing fixtures from one test to persist into subsequent tests when using a session-scoped server. `replace_fixtures()` now clears the list first. The `@use_fixtures` decorator now calls `replace_fixtures()` instead of `add_fixtures()`.
- **Gemini SDK example**: README example updated to use `google-genai` (REST-based). The previous `google-generativeai` example used gRPC by default and would not connect to the HTTP server.

### Added
- `MockLLMServerFixture.assert_not_called()` — assert no calls were made
- `MockLLMServerFixture.assert_called_n_times(n)` — assert exactly n calls
- `MockLLMServerFixture.assert_model_was(model)` — assert a specific model was used in at least one call
- `MockLLMServer.replace_fixtures(fixtures)` — replace all loaded fixtures atomically
- Gemini streaming tests (text, tool calls, stop reason)
- OpenAI embeddings and models endpoint tests

## [0.1.0] - 2026-04-01

### Added
- Initial release
- Fixture engine with YAML/JSON loading and priority-based matching
- Multi-provider support: OpenAI, Anthropic, Google Gemini
- SSE streaming simulation with configurable chunk delay
- Record-and-replay proxy
- pytest plugin with session-scoped server and `@use_fixtures` decorator
- Structured output (JSON schema) validation
- CLI (`stubllm serve`, `stubllm record`)
