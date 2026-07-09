"""Configuration loading and normalized-ROI conversion for offline detection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypeAlias

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROI_CONFIG_PATH = PROJECT_ROOT / "config" / "roi.yaml"
DEFAULT_THRESHOLDS_CONFIG_PATH = PROJECT_ROOT / "config" / "thresholds.yaml"

NormalizedROI: TypeAlias = tuple[float, float, float, float]
PixelROI: TypeAlias = tuple[int, int, int, int]

ROI_NAMES = (
    "top_prompt",
    "center_space",
    "hook_bar",
    "press_sequence",
    "get_window",
    "right_quest_area",
)

STATE_NAMES = ("IDLE", "WAITING", "READY", "HOOK", "PRESS", "GET")

DEFAULT_ROI_DATA: dict[str, Any] = {
    "screen_reference": {"width": 2048, "height": 1151},
    "rois": {
        "top_prompt": {"x1": 0.30, "y1": 0.02, "x2": 0.70, "y2": 0.10},
        "center_space": {"x1": 0.38, "y1": 0.10, "x2": 0.58, "y2": 0.25},
        "hook_bar": {"x1": 0.38, "y1": 0.23, "x2": 0.62, "y2": 0.36},
        "press_sequence": {"x1": 0.38, "y1": 0.17, "x2": 0.62, "y2": 0.31},
        "get_window": {"x1": 0.75, "y1": 0.55, "x2": 0.94, "y2": 0.84},
        "right_quest_area": {"x1": 0.82, "y1": 0.20, "x2": 0.99, "y2": 0.75},
    },
}

DEFAULT_THRESHOLDS_DATA: dict[str, Any] = {
    "state_thresholds": {
        "min_confidence": 0.65,
        "unknown_below": 0.55,
        "stable_frames_required": 2,
    },
    "detectors": {state.lower(): {"min_confidence": 0.65} for state in STATE_NAMES},
    "debug": {
        "save_debug_images": True,
        "save_on_state_change": True,
        "save_on_low_confidence": True,
        "save_on_unknown": True,
        "max_debug_images": 200,
        "max_debug_size_mb": 500,
        "max_debug_age_days": 2,
    },
}


@dataclass(frozen=True)
class ROIConfig:
    screen_reference: tuple[int, int]
    rois: Mapping[str, NormalizedROI]
    source: Path | None

    def pixel_roi(self, name: str, image_width: int, image_height: int) -> PixelROI:
        """Convert a named normalized ROI to clipped image pixel coordinates."""
        try:
            roi = self.rois[name]
        except KeyError as exc:
            raise KeyError(f"Unknown ROI name: {name}") from exc
        return normalized_to_pixel_roi(roi, image_width, image_height)


@dataclass(frozen=True)
class DebugSettings:
    save_debug_images: bool
    save_on_state_change: bool
    save_on_low_confidence: bool
    save_on_unknown: bool
    max_debug_images: int
    max_debug_size_mb: int
    max_debug_age_days: int


@dataclass(frozen=True)
class ThresholdConfig:
    min_confidence: float
    unknown_below: float
    stable_frames_required: int
    detector_min_confidence: Mapping[str, float]
    debug: DebugSettings
    source: Path | None

    def min_confidence_for(self, state: str) -> float:
        return self.detector_min_confidence.get(state.lower(), self.min_confidence)


def _read_yaml_or_default(path: Path, default: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if not path.exists():
        return deepcopy(default), None
    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML configuration: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data, path


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _as_positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {comparator} integer")
    return value


def _as_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false")
    return value


def _read_normalized_roi(name: str, values: Any) -> NormalizedROI:
    if not isinstance(values, dict):
        raise ValueError(f"rois.{name} must be a mapping")
    try:
        roi = tuple(_as_float(values[field], f"rois.{name}.{field}") for field in ("x1", "y1", "x2", "y2"))
    except KeyError as exc:
        raise ValueError(f"rois.{name} is missing {exc.args[0]}") from exc
    x1, y1, x2, y2 = roi
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"rois.{name} must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1")
    return roi  # type: ignore[return-value]


def normalized_to_pixel_roi(roi: NormalizedROI, image_width: int, image_height: int) -> PixelROI:
    """Convert normalized coordinates into non-empty, image-bounded coordinates."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    x1, y1, x2, y2 = roi
    left = max(0, min(image_width - 1, round(x1 * image_width)))
    top = max(0, min(image_height - 1, round(y1 * image_height)))
    right = max(left + 1, min(image_width, round(x2 * image_width)))
    bottom = max(top + 1, min(image_height, round(y2 * image_height)))
    return left, top, right, bottom


def load_roi_config(path: str | Path | None = None) -> ROIConfig:
    """Read ROI configuration, or return safe built-in normalized defaults."""
    config_path = Path(path) if path is not None else DEFAULT_ROI_CONFIG_PATH
    data, source = _read_yaml_or_default(config_path, DEFAULT_ROI_DATA)
    reference = data.get("screen_reference")
    raw_rois = data.get("rois")
    if not isinstance(reference, dict) or not isinstance(raw_rois, dict):
        raise ValueError("ROI configuration requires screen_reference and rois mappings")
    try:
        screen_reference = (
            _as_positive_int(reference["width"], "screen_reference.width"),
            _as_positive_int(reference["height"], "screen_reference.height"),
        )
    except KeyError as exc:
        raise ValueError(f"screen_reference is missing {exc.args[0]}") from exc
    missing = [name for name in ROI_NAMES if name not in raw_rois]
    if missing:
        raise ValueError(f"ROI configuration is missing: {', '.join(missing)}")
    return ROIConfig(
        screen_reference=screen_reference,
        rois={name: _read_normalized_roi(name, raw_rois[name]) for name in ROI_NAMES},
        source=source,
    )


def save_roi_config(path: str | Path, config: ROIConfig) -> None:
    """Persist a complete, already-validated ROI configuration."""
    output_path = Path(path)
    data = {
        "screen_reference": {"width": config.screen_reference[0], "height": config.screen_reference[1]},
        "rois": {
            name: dict(zip(("x1", "y1", "x2", "y2"), config.rois[name], strict=True))
            for name in ROI_NAMES
        },
    }
    # Validate before writing so a malformed calibration can never replace a config.
    for name in ROI_NAMES:
        _read_normalized_roi(name, data["rois"][name])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def load_thresholds_config(path: str | Path | None = None) -> ThresholdConfig:
    """Read detection, UNKNOWN, and debug-retention thresholds with defaults."""
    config_path = Path(path) if path is not None else DEFAULT_THRESHOLDS_CONFIG_PATH
    data, source = _read_yaml_or_default(config_path, DEFAULT_THRESHOLDS_DATA)
    states = data.get("state_thresholds")
    detectors = data.get("detectors")
    debug = data.get("debug")
    if not isinstance(states, dict) or not isinstance(detectors, dict) or not isinstance(debug, dict):
        raise ValueError("Threshold configuration requires state_thresholds, detectors, and debug mappings")
    try:
        min_confidence = _as_float(states["min_confidence"], "state_thresholds.min_confidence")
        unknown_below = _as_float(states["unknown_below"], "state_thresholds.unknown_below")
        stable_frames_required = _as_positive_int(
            states["stable_frames_required"], "state_thresholds.stable_frames_required"
        )
    except KeyError as exc:
        raise ValueError(f"state_thresholds is missing {exc.args[0]}") from exc
    if not (0.0 <= unknown_below <= min_confidence <= 1.0):
        raise ValueError("state_thresholds must satisfy 0 <= unknown_below <= min_confidence <= 1")

    detector_thresholds: dict[str, float] = {}
    for state in STATE_NAMES:
        item = detectors.get(state.lower())
        if not isinstance(item, dict) or "min_confidence" not in item:
            raise ValueError(f"detectors.{state.lower()}.min_confidence is required")
        value = _as_float(item["min_confidence"], f"detectors.{state.lower()}.min_confidence")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"detectors.{state.lower()}.min_confidence must be between 0 and 1")
        detector_thresholds[state.lower()] = value

    try:
        debug_settings = DebugSettings(
            save_debug_images=_as_bool(debug["save_debug_images"], "debug.save_debug_images"),
            save_on_state_change=_as_bool(debug["save_on_state_change"], "debug.save_on_state_change"),
            save_on_low_confidence=_as_bool(debug["save_on_low_confidence"], "debug.save_on_low_confidence"),
            save_on_unknown=_as_bool(debug["save_on_unknown"], "debug.save_on_unknown"),
            max_debug_images=_as_positive_int(debug["max_debug_images"], "debug.max_debug_images", allow_zero=True),
            max_debug_size_mb=_as_positive_int(debug["max_debug_size_mb"], "debug.max_debug_size_mb", allow_zero=True),
            max_debug_age_days=_as_positive_int(debug["max_debug_age_days"], "debug.max_debug_age_days", allow_zero=True),
        )
    except KeyError as exc:
        raise ValueError(f"debug is missing {exc.args[0]}") from exc
    return ThresholdConfig(
        min_confidence=min_confidence,
        unknown_below=unknown_below,
        stable_frames_required=stable_frames_required,
        detector_min_confidence=detector_thresholds,
        debug=debug_settings,
        source=source,
    )
