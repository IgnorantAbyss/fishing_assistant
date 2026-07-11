"""Human-authored prompt observation annotations, independent of global state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


class PromptAnnotationKind(str, Enum):
    IDLE_PROMPT = "IDLE_PROMPT"
    WAITING_PROMPT = "WAITING_PROMPT"
    READY_PROMPT = "READY_PROMPT"
    OTHER_PROMPT = "OTHER_PROMPT"
    NO_PROMPT = "NO_PROMPT"
    IGNORE = "IGNORE"


@dataclass(frozen=True)
class PromptFrameAnnotation:
    """Dataset annotation metadata; runtime consumers use only ``observation``."""

    observation: PromptAnnotationKind
    prompt_id: str | None = None
    notes: str | None = None


def validate_prompt_segments(
    segments: Iterable[Mapping[str, Any]], frame_count: int
) -> list[dict[str, Any]]:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    parsed: list[dict[str, Any]] = []
    for raw in segments:
        start, end, observation = raw.get("start"), raw.get("end"), raw.get("observation")
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError("Prompt segment start must be an integer")
        if not isinstance(end, int) or isinstance(end, bool):
            raise ValueError("Prompt segment end must be an integer")
        if not isinstance(observation, str):
            raise ValueError("Prompt segment observation must be a string")
        observation = observation.upper()
        if observation not in {item.value for item in PromptAnnotationKind}:
            raise ValueError(f"Invalid prompt observation: {observation}")
        if start < 1 or end < start or end > frame_count:
            raise ValueError(f"Invalid prompt range {start}-{end} for {frame_count} frames")
        parsed_segment: dict[str, Any] = {"start": start, "end": end, "observation": observation}
        if "prompt_id" in raw:
            prompt_id = raw.get("prompt_id")
            if prompt_id is not None and (not isinstance(prompt_id, str) or not prompt_id.strip()):
                raise ValueError("prompt_id must be a non-empty string or null")
            parsed_segment["prompt_id"] = prompt_id.strip() if isinstance(prompt_id, str) else None
        if "notes" in raw:
            notes = raw.get("notes")
            if notes is not None and not isinstance(notes, str):
                raise ValueError("Prompt annotation notes must be a string or null")
            parsed_segment["notes"] = notes
        parsed.append(parsed_segment)
    if not parsed:
        raise ValueError("Prompt annotation requires at least one segment")
    parsed.sort(key=lambda item: int(item["start"]))
    previous_end = 0
    for item in parsed:
        start, end = int(item["start"]), int(item["end"])
        if start <= previous_end:
            raise ValueError("Prompt ranges must not overlap")
        if start != previous_end + 1:
            raise ValueError(f"Prompt ranges contain a gap at frame {previous_end + 1}")
        previous_end = end
    if previous_end != frame_count:
        raise ValueError(f"Prompt ranges contain a gap at frame {previous_end + 1}")
    return parsed


def prompt_labels_from_segments(
    segments: Iterable[Mapping[str, Any]], frame_count: int
) -> dict[int, PromptAnnotationKind]:
    parsed = validate_prompt_segments(segments, frame_count)
    return {
        frame: PromptAnnotationKind(str(item["observation"]))
        for item in parsed
        for frame in range(int(item["start"]), int(item["end"]) + 1)
    }


def prompt_annotations_from_segments(
    segments: Iterable[Mapping[str, Any]], frame_count: int
) -> dict[int, PromptFrameAnnotation]:
    parsed = validate_prompt_segments(segments, frame_count)
    return {
        frame: PromptFrameAnnotation(
            observation=PromptAnnotationKind(str(item["observation"])),
            prompt_id=item.get("prompt_id"),
            notes=item.get("notes"),
        )
        for item in parsed
        for frame in range(int(item["start"]), int(item["end"]) + 1)
    }


def _load_segments(path: str | Path) -> list[Mapping[str, Any]]:
    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("segments"), list):
        raise ValueError(f"Prompt ground truth requires version: 1 and segments: {source}")
    return data["segments"]


def load_prompt_ground_truth(path: str | Path, frame_count: int) -> dict[int, PromptAnnotationKind]:
    return prompt_labels_from_segments(_load_segments(path), frame_count)


def load_prompt_annotations(path: str | Path, frame_count: int) -> dict[int, PromptFrameAnnotation]:
    """Load optional prompt_id/notes for dataset tooling, never runtime inference."""
    return prompt_annotations_from_segments(_load_segments(path), frame_count)


def write_prompt_ground_truth(
    path: str | Path,
    segments: Iterable[Mapping[str, Any]],
    frame_count: int,
    *,
    force: bool = False,
) -> Path:
    destination = Path(path)
    if destination.exists() and not force:
        raise FileExistsError(f"Prompt ground truth already exists: {destination}; use --force explicitly")
    parsed = validate_prompt_segments(segments, frame_count)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump({"version": 1, "segments": parsed}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def prompt_boundary_frames(labels: Mapping[int, PromptAnnotationKind]) -> set[int]:
    boundaries: set[int] = set()
    previous: PromptAnnotationKind | None = None
    for frame, label in sorted(labels.items()):
        if previous is not None and label != previous:
            boundaries.update({frame - 1, frame})
        previous = label
    return boundaries
