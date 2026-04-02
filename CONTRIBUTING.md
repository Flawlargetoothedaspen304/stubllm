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

## Making changes

1. Fork the repo and create a branch
2. Write tests first (the test suite is your spec)
3. Make the tests pass
4. Open a pull request — describe what you changed and why

## Reporting bugs

Open an issue with:
- What you expected to happen
- What actually happened
- A minimal code example that reproduces it

## Adding provider support

New providers go in `src/stubllm/providers/`. Subclass `BaseProvider`, implement `format_response()` and `format_stream_chunk()`, register the router in `server.py`.
