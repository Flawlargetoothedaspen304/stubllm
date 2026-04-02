# Contributing to stubllm

## Development setup

```bash
git clone https://github.com/airupt/stubllm
cd stubllm
pip install -e ".[dev]"
```

## Running checks

```bash
pytest tests/ -v                                      # tests + coverage (must stay ≥80%)
ruff check .                                          # lint
ruff check . --fix                                    # auto-fix lint errors
mypy src/stubllm --ignore-missing-imports             # type check
```

All three must be clean before a PR can merge — CI enforces this on Python 3.10–3.13.

## Making changes

1. Fork the repo and create a branch from `main`
2. Write tests first — the test suite is the spec
3. Make the tests pass
4. Run `ruff check .` and `mypy src/stubllm --ignore-missing-imports`
5. Open a pull request describing what changed and why

## Adding a new provider

New providers go in `src/stubllm/providers/`. Steps:

1. Create `src/stubllm/providers/<name>.py`
2. Subclass `BaseProvider`
3. Implement `format_response()` and `format_stream_chunk()`
4. Register the router in `server.py`
5. Add tests in `tests/test_providers/test_<name>.py`

## Reporting bugs

[Open an issue](https://github.com/airupt/stubllm/issues/new?template=bug_report.yml) with a minimal reproducible example.

## Security vulnerabilities

Do not open a public issue. Use [private vulnerability reporting](https://github.com/airupt/stubllm/security/advisories/new) instead.

## Releases

Releases are automated. Pushing a `v*` tag triggers the publish workflow, which runs tests, builds the package, publishes to PyPI via OIDC trusted publishing, and creates a GitHub Release.
