from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PromptROICandidate:
    candidate_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    note: str = ""

    @property
    def normalized(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def pixel_bounds(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            max(0, min(width - 1, round(self.x1 * width))),
            max(0, min(height - 1, round(self.y1 * height))),
            max(1, min(width, round(self.x2 * width))),
            max(1, min(height, round(self.y2 * height))),
        )


def load_roi_candidates(path: str | Path) -> list[PromptROICandidate]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("manual_approval_required") is not True:
        raise ValueError("Prompt ROI candidates must require manual approval")
    raw = data.get("candidates")
    if not isinstance(raw, list) or not raw:
        raise ValueError("At least one Prompt ROI candidate is required")
    candidates: list[PromptROICandidate] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each Prompt ROI candidate must be a mapping")
        candidate = PromptROICandidate(
            str(item.get("id", "")),
            float(item["x1"]), float(item["y1"]), float(item["x2"]), float(item["y2"]),
            str(item.get("note", "")),
        )
        if not candidate.candidate_id or not (0 <= candidate.x1 < candidate.x2 <= 1 and 0 <= candidate.y1 < candidate.y2 <= 1):
            raise ValueError(f"Invalid normalized Prompt ROI candidate: {item}")
        candidates.append(candidate)
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("Prompt ROI candidate ids must be unique")
    return candidates


def load_approved_prompt_roi(config_path: str | Path) -> PromptROICandidate | None:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    prompt = data.get("prompt", {}) if isinstance(data, dict) else {}
    status = prompt.get("roi_status")
    if status == "unapproved":
        return None
    if status != "approved" or not isinstance(prompt.get("roi"), dict):
        raise ValueError("prompt.roi_status must be unapproved or approved with explicit roi")
    roi = prompt["roi"]
    candidate = PromptROICandidate("approved", float(roi["x1"]), float(roi["y1"]), float(roi["x2"]), float(roi["y2"]), "User-approved ROI")
    if not (0 <= candidate.x1 < candidate.x2 <= 1 and 0 <= candidate.y1 < candidate.y2 <= 1):
        raise ValueError("Approved Prompt ROI coordinates are invalid")
    return candidate
