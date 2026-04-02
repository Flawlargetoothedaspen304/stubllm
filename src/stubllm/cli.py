"""CLI entry point for stubllm."""

from __future__ import annotations

from pathlib import Path

import click
import uvicorn

from stubllm import __version__


@click.group()
@click.version_option(version=__version__, prog_name="stubllm")
def cli() -> None:
    """stubllm — Deterministic mock server for LLM APIs."""


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to.")
@click.option("--port", default=8765, show_default=True, help="Port to listen on.")
@click.option(
    "--fixture-dir",
    "fixture_dirs",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing fixture files (repeatable).",
)
@click.option(
    "--fixture-file",
    "fixture_files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Individual fixture file (repeatable).",
)
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload on file changes.")
@click.option("--log-level", default="info", show_default=True, help="Uvicorn log level.")
def serve(
    host: str,
    port: int,
    fixture_dirs: tuple[Path, ...],
    fixture_files: tuple[Path, ...],
    reload: bool,
    log_level: str,
) -> None:
    """Start the stubllm HTTP server."""
    from stubllm.server import create_app

    dirs = list(fixture_dirs)
    files = list(fixture_files)

    # Default: load from ./fixtures if it exists and nothing else specified
    if not dirs and not files:
        default_dir = Path("fixtures")
        if default_dir.exists():
            dirs = [default_dir]
            click.echo(f"Auto-loading fixtures from {default_dir}/")

    app = create_app(fixture_dirs=dirs or None, fixture_files=files or None)
    fixture_count = len(app.state.fixtures)

    click.echo(f"stubllm v{__version__} starting on http://{host}:{port}")
    click.echo(f"Loaded {fixture_count} fixture(s)")
    click.echo("Providers: OpenAI (/v1/...), Anthropic (/v1/messages), Gemini (/v1beta/...)")

    uvicorn.run(app, host=host, port=port, reload=reload, log_level=log_level)


@cli.command()
@click.option(
    "--target",
    required=True,
    help="Target LLM API base URL (e.g. https://api.openai.com).",
)
@click.option(
    "--fixture-dir",
    default="fixtures",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory to save recorded fixtures.",
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True)
@click.option("--log-level", default="info", show_default=True)
def record(
    target: str,
    fixture_dir: Path,
    host: str,
    port: int,
    log_level: str,
) -> None:
    """Start stubllm in record-and-replay mode (proxy to real API)."""
    from stubllm.recorder.proxy import create_recording_app

    fixture_dir.mkdir(parents=True, exist_ok=True)
    app = create_recording_app(target_url=target, fixture_dir=fixture_dir)

    click.echo(f"stubllm v{__version__} recording mode")
    click.echo(f"Proxying requests to: {target}")
    click.echo(f"Saving fixtures to: {fixture_dir}/")
    click.echo(f"Listening on http://{host}:{port}")

    uvicorn.run(app, host=host, port=port, log_level=log_level)
