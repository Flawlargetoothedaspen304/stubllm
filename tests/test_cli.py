"""Tests for the CLI entry point."""

from __future__ import annotations

from click.testing import CliRunner

from stubllm.cli import cli


class TestCLI:
    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.2" in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "stubllm" in result.output.lower()

    def test_serve_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--host" in result.output
        assert "--fixture-dir" in result.output

    def test_record_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["record", "--help"])
        assert result.exit_code == 0
        assert "--target" in result.output
        assert "--fixture-dir" in result.output

    def test_serve_with_fixture_dir(self, tmp_path: object) -> None:
        """serve command loads fixtures from a directory."""
        from pathlib import Path
        from unittest.mock import patch

        import yaml

        assert isinstance(tmp_path, Path)
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        (fixtures_dir / "test.yaml").write_text(
            yaml.dump([{"name": "t", "response": {"content": "ok"}}])
        )

        runner = CliRunner()
        with patch("uvicorn.run") as mock_run:
            runner.invoke(
                cli,
                ["serve", "--port", "9999", "--fixture-dir", str(fixtures_dir)],
            )
            assert mock_run.called
            # Check that the app was created with fixtures
            call_kwargs = mock_run.call_args
            assert call_kwargs is not None

    def test_serve_auto_loads_fixtures_dir(self, tmp_path: object) -> None:
        """serve command auto-loads ./fixtures if it exists."""
        from pathlib import Path
        from unittest.mock import patch

        import yaml

        assert isinstance(tmp_path, Path)

        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            os.makedirs("fixtures", exist_ok=True)
            with open("fixtures/auto.yaml", "w") as f:
                yaml.dump([{"name": "auto", "response": {"content": "auto"}}], f)

            with patch("uvicorn.run"):
                result = runner.invoke(cli, ["serve", "--port", "9998"])
                assert "Auto-loading fixtures" in result.output
