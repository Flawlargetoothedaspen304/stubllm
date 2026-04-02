# Contributing to stubllm

## Setup

```bash
git clone https://github.com/airupt/stubllm
cd stubllm
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

Coverage must stay above 80%.

## Linting

```bash
# Check for lint errors (run from project root — covers src/, tests/, examples/)
ruff check .

# Auto-fix what's fixable
ruff check . --fix
```

## Type checking

```bash
mypy src/stubllm --ignore-missing-imports
```

All mypy errors must be resolved before merging. The CI enforces this.

## Making changes

1. Fork the repo and create a branch
2. Write tests first (the test suite is your spec)
3. Make the tests pass
4. Run `ruff check .` and `mypy src/stubllm --ignore-missing-imports` — both must be clean
5. Open a pull request — describe what you changed and why

## Reporting bugs

Open an issue with:
- What you expected to happen
- What actually happened
- A minimal code example that reproduces it

## Adding provider support

New providers go in `src/stubllm/providers/`. Subclass `BaseProvider`, implement `format_response()` and `format_stream_chunk()`, register the router in `server.py`.
