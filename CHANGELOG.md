# Changelog

All notable changes to this project will be documented in this file.

## [0.1.2] - 2026-04-02

### Added
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
