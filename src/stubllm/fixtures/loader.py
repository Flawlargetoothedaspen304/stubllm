"""YAML/JSON fixture file loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from stubllm.fixtures.models import Fixture


class FixtureLoader:
    """Loads fixture definitions from YAML or JSON files."""

    def load_file(self, path: Path) -> list[Fixture]:
        """Load fixtures from a single YAML or JSON file."""
        raw = self._read_raw(path)
        return self._parse_raw(raw, source=str(path))

    def load_directory(self, directory: Path) -> list[Fixture]:
        """Load all fixtures from a directory (recursive)."""
        fixtures: list[Fixture] = []
        all_paths = (
            sorted(directory.rglob("*.yaml"))
            + sorted(directory.rglob("*.yml"))
            + sorted(directory.rglob("*.json"))
        )
        for path in all_paths:
            fixtures.extend(self.load_file(path))
        return fixtures

    def load_from_dict(self, data: dict[str, Any] | list[Any]) -> list[Fixture]:
        """Load fixtures from an already-parsed data structure."""
        return self._parse_raw(data, source="<dict>")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_raw(self, path: Path) -> dict[str, Any] | list[Any]:
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return yaml.safe_load(text)  # type: ignore[no-any-return]
        if suffix == ".json":
            return json.loads(text)  # type: ignore[no-any-return]
        raise ValueError(
            f"Unsupported fixture file format: {path.suffix!r}. Use .yaml, .yml, or .json"
        )

    def _parse_raw(self, raw: dict[str, Any] | list[Any], source: str) -> list[Fixture]:
        if isinstance(raw, list):
            # Bare list of fixture dicts
            return [self._parse_one(item, i, source) for i, item in enumerate(raw)]
        if isinstance(raw, dict):
            if "fixtures" in raw:
                items = raw["fixtures"]
                return [self._parse_one(item, i, source) for i, item in enumerate(items)]
            # Single fixture as a top-level dict
            return [self._parse_one(raw, 0, source)]
        raise ValueError(
            f"Cannot parse fixture from {source}: expected dict or list, got {type(raw).__name__}"
        )

    def _parse_one(self, data: Any, index: int, source: str) -> Fixture:
        if not isinstance(data, dict):
            raise ValueError(
                f"Fixture #{index} in {source} must be a dict, got {type(data).__name__}"
            )
        try:
            return Fixture.model_validate(data)
        except Exception as exc:
            name = data.get("name", f"#{index}")
            raise ValueError(f"Invalid fixture {name!r} in {source}: {exc}") from exc
