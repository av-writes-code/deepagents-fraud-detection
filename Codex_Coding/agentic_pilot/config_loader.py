"""Configuration utilities for the Agentic ID Fraud pilot orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:  # pragma: no cover - optional dependency
    import yaml
except ModuleNotFoundError:  # pragma: no cover - handled by fallback parser
    yaml = None  # type: ignore[assignment]


@dataclass
class PilotConfig:
    """Typed view over the orchestrator configuration file."""

    raw: Dict[str, Any]

    @property
    def thresholds(self) -> Dict[str, float]:
        return self.raw.get("thresholds", {})

    @property
    def borderline_band(self) -> Dict[str, float]:
        band = self.raw.get("borderline_band", [0.0, 1.0])
        return {"lower": float(band[0]), "upper": float(band[1])}

    @property
    def uncertainty(self) -> Dict[str, float]:
        return self.raw.get("uncertainty", {})

    @property
    def actions(self) -> Dict[str, Any]:
        return self.raw.get("actions", {})

    @property
    def fusion(self) -> Dict[str, Any]:
        return self.raw.get("fusion", {})

    @property
    def timeouts(self) -> Dict[str, int]:
        return self.raw.get("timeouts_ms", {})

    @property
    def logging(self) -> Dict[str, Any]:
        return self.raw.get("logging", {})


def load_config(path: Path) -> PilotConfig:
    """Load a YAML configuration file from disk."""

    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = _fallback_yaml_parse(text)

    if not isinstance(data, dict):  # pragma: no cover - defensive guard
        raise ValueError("Configuration root must be a mapping")
    return PilotConfig(raw=data)


def _fallback_yaml_parse(text: str) -> Dict[str, Any]:
    """Parse a minimal YAML subset without third-party dependencies."""

    import ast

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"Invalid config line: {raw_line}")
        key, value = [part.strip() for part in line.split(":", 1)]

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Unbalanced indentation in config: {raw_line}")
        container = stack[-1][1]

        if not value:
            new_dict: Dict[str, Any] = {}
            container[key] = new_dict
            stack.append((indent, new_dict))
            continue

        container[key] = _parse_scalar(value, ast)

    return root


def _parse_scalar(value: str, ast_module: Any) -> Any:
    """Parse a scalar token from the fallback YAML parser."""

    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False

    try:
        return ast_module.literal_eval(value)
    except Exception:
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value