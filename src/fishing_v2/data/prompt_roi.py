"""Fixed-environment Prompt ROI definitions.

Pixel coordinates at the configured 2560x1440 runtime resolution are the source
of truth. Normalized coordinates are derived display metadata only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class PromptROICandidate:
    candidate_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    note: str = ""
    intended_content: str = ""
    reference_width: int = 2560
    reference_height: int = 1440

    @property
    def pixel(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def normalized(self) -> tuple[float, float, float, float]:
        return (
            self.x1 / self.reference_width,
            self.y1 / self.reference_height,
            self.x2 / self.reference_width,
            self.y2 / self.reference_height,
        )

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def pixel_bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        if (width, height) != (self.reference_width, self.reference_height):
            raise ValueError(
                "Unsupported frame size for fixed Prompt ROI: "
                f"{width}x{height}; expected {self.reference_width}x{self.reference_height}"
            )
        return self.pixel


def _positive_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Prompt ROI {key} must be an integer pixel coordinate")
    return value


def _candidate_from_mapping(
    item: Mapping[str, Any], *, reference_width: int, reference_height: int
) -> PromptROICandidate:
    pixel = item.get("pixel")
    if not isinstance(pixel, Mapping):
        raise ValueError("Prompt ROI candidate requires pixel coordinates as source of truth")
    candidate = PromptROICandidate(
        candidate_id=str(item.get("id", "")),
        x1=_positive_int(pixel, "x1"),
        y1=_positive_int(pixel, "y1"),
        x2=_positive_int(pixel, "x2"),
        y2=_positive_int(pixel, "y2"),
        note=str(item.get("note", "")),
        intended_content=str(item.get("intended_content", "")),
        reference_width=reference_width,
        reference_height=reference_height,
    )
    if not candidate.candidate_id:
        raise ValueError("Prompt ROI candidate id is required")
    if not (
        0 <= candidate.x1 < candidate.x2 <= reference_width
        and 0 <= candidate.y1 < candidate.y2 <= reference_height
    ):
        raise ValueError(f"Prompt ROI candidate is outside the reference frame: {item}")
    normalized = item.get("normalized")
    if normalized is not None:
        if not isinstance(normalized, Mapping):
            raise ValueError("normalized Prompt ROI metadata must be a mapping")
        displayed = tuple(float(normalized[key]) for key in ("x1", "y1", "x2", "y2"))
        if any(abs(actual - shown) > 0.000001 for actual, shown in zip(candidate.normalized, displayed)):
            raise ValueError("normalized Prompt ROI metadata must be derived from pixel coordinates")
    return candidate


def load_roi_candidates(path: str | Path) -> list[PromptROICandidate]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("manual_approval_required") is not True:
        raise ValueError("Prompt ROI candidates must require manual approval")
    if data.get("source_of_truth") != "pixel" or data.get("roi_status") != "unapproved":
        raise ValueError("Prompt ROI review must use pixel source coordinates and remain unapproved")
    reference = data.get("runtime_reference")
    if not isinstance(reference, Mapping):
        raise ValueError("Prompt ROI candidates require runtime_reference")
    width = _positive_int(reference, "width")
    height = _positive_int(reference, "height")
    raw = data.get("candidates")
    if not isinstance(raw, list) or not raw:
        raise ValueError("At least one Prompt ROI candidate is required")
    candidates = [
        _candidate_from_mapping(item, reference_width=width, reference_height=height)
        for item in raw
        if isinstance(item, Mapping)
    ]
    if len(candidates) != len(raw):
        raise ValueError("Each Prompt ROI candidate must be a mapping")
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("Prompt ROI candidate ids must be unique")
    return candidates


def load_approved_prompt_roi(config_path: str | Path) -> PromptROICandidate | None:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    prompt = data.get("prompt", {}) if isinstance(data, dict) else {}
    status = prompt.get("roi_status")
    if status == "unapproved":
        return None
    if status != "approved" or prompt.get("roi_source_of_truth") != "pixel":
        raise ValueError("Approved Prompt ROI must explicitly use pixel source coordinates")
    roi = prompt.get("roi")
    constraints = data.get("runtime_constraints", {}) if isinstance(data, dict) else {}
    resolution = constraints.get("resolution", {}) if isinstance(constraints, dict) else {}
    if not isinstance(roi, Mapping) or not isinstance(resolution, Mapping):
        raise ValueError("Approved Prompt ROI requires pixel roi and runtime resolution")
    return _candidate_from_mapping(
        {"id": "approved", "pixel": roi, "note": "User-approved ROI"},
        reference_width=_positive_int(resolution, "width"),
        reference_height=_positive_int(resolution, "height"),
    )
