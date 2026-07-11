"""Validation interface for the deliberately fixed v2 runtime environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RuntimeConstraints:
    width: int
    height: int
    ui_scale: str
    language: str
    window_mode: str
    prompt_position: str
    supported_environment_only: bool


@dataclass(frozen=True)
class RuntimeEnvironmentValidation:
    supported: bool
    expected_size: tuple[int, int]
    actual_size: tuple[int, int]
    violations: tuple[str, ...]


class UnsupportedRuntimeEnvironmentError(RuntimeError):
    """Raised before inference when the fixed environment contract is violated."""


def load_runtime_constraints(config_path: str | Path) -> RuntimeConstraints:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    raw = data.get("runtime_constraints") if isinstance(data, dict) else None
    resolution = raw.get("resolution") if isinstance(raw, dict) else None
    if not isinstance(resolution, dict):
        raise ValueError("runtime_constraints.resolution is required")
    return RuntimeConstraints(
        width=int(resolution["width"]),
        height=int(resolution["height"]),
        ui_scale=str(raw["ui_scale"]),
        language=str(raw["language"]),
        window_mode=str(raw["window_mode"]),
        prompt_position=str(raw["prompt_position"]),
        supported_environment_only=bool(raw["supported_environment_only"]),
    )


def validate_frame_environment(
    width: int, height: int, constraints: RuntimeConstraints
) -> RuntimeEnvironmentValidation:
    violations: list[str] = []
    if (width, height) != (constraints.width, constraints.height):
        violations.append(
            f"unsupported_resolution:{width}x{height};expected:{constraints.width}x{constraints.height}"
        )
    return RuntimeEnvironmentValidation(
        supported=not violations,
        expected_size=(constraints.width, constraints.height),
        actual_size=(width, height),
        violations=tuple(violations),
    )


def require_supported_frame(
    width: int, height: int, constraints: RuntimeConstraints
) -> RuntimeEnvironmentValidation:
    result = validate_frame_environment(width, height, constraints)
    if not result.supported:
        raise UnsupportedRuntimeEnvironmentError("; ".join(result.violations))
    return result
