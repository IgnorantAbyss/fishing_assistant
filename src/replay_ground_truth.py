"""Validation and loading for human-authored replay ground truth."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


EVALUATED_STATES = ("IDLE", "WAITING", "READY", "HOOK", "PRESS", "GET")
IGNORE_STATE = "IGNORE"
GROUND_TRUTH_STATES = (*EVALUATED_STATES, IGNORE_STATE)


class _IndentedSafeDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


def validate_segments(
    segments: Iterable[Mapping[str, Any]], frame_count: int
) -> list[dict[str, int | str]]:
    """Return normalized segments that cover exactly frames 1..frame_count."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    normalized: list[dict[str, int | str]] = []
    for segment in segments:
        start, end, state = segment.get("start"), segment.get("end"), segment.get("state")
        if not isinstance(start, int) or isinstance(start, bool):
            raise ValueError("Ground-truth segment start must be an integer")
        if not isinstance(end, int) or isinstance(end, bool):
            raise ValueError("Ground-truth segment end must be an integer")
        if not isinstance(state, str):
            raise ValueError("Ground-truth segment state must be a string")
        state = state.upper()
        if state not in GROUND_TRUTH_STATES or start < 1 or end < start:
            raise ValueError(f"Invalid ground-truth segment: {dict(segment)}")
        if end > frame_count:
            raise ValueError(f"Range {start}-{end} exceeds session frame_count {frame_count}")
        normalized.append({"start": start, "end": end, "state": state})

    if not normalized:
        raise ValueError("Ground truth must contain at least one segment")
    normalized.sort(key=lambda segment: int(segment["start"]))
    previous_end = 0
    for segment in normalized:
        start, end = int(segment["start"]), int(segment["end"])
        if start <= previous_end:
            raise ValueError("Ground-truth ranges must not overlap")
        if start != previous_end + 1:
            raise ValueError(f"Ground-truth ranges contain a gap at frame {previous_end + 1}")
        previous_end = end
    if previous_end != frame_count:
        raise ValueError(f"Ground-truth ranges contain a gap at frame {previous_end + 1}")
    return normalized


def load_ground_truth(path: str | Path, frame_count: int) -> dict[int, str]:
    ground_truth_path = Path(path)
    if not ground_truth_path.is_file():
        raise FileNotFoundError(
            f"Ground truth not found: {ground_truth_path}; run tools/create_ground_truth.py first"
        )
    data = yaml.safe_load(ground_truth_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError(f"Ground truth must contain a segments list: {ground_truth_path}")
    segments = validate_segments(data["segments"], frame_count)
    return {
        frame_index: str(segment["state"])
        for segment in segments
        for frame_index in range(int(segment["start"]), int(segment["end"]) + 1)
    }


def write_ground_truth(
    path: str | Path, segments: Iterable[Mapping[str, Any]], frame_count: int
) -> Path:
    destination = Path(path)
    normalized = validate_segments(segments, frame_count)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as file:
        yaml.dump(
            {"segments": normalized},
            file,
            Dumper=_IndentedSafeDumper,
            allow_unicode=True,
            sort_keys=False,
        )
    return destination


def load_annotations(path: str | Path, frame_count: int) -> dict[str, Any]:
    """Load optional diagnostic event ranges; never create ground-truth labels."""
    annotations_path = Path(path)
    if not annotations_path.is_file():
        return {"events": {}, "notes": {}}
    data = yaml.safe_load(annotations_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Annotations root must be a mapping: {annotations_path}")
    raw_events = data.get("events", {})
    notes = data.get("notes", {})
    if not isinstance(raw_events, dict) or not isinstance(notes, dict):
        raise ValueError("Annotations require events and notes mappings")
    events: dict[str, dict[str, int]] = {}
    for name, event in raw_events.items():
        if not isinstance(name, str) or not isinstance(event, dict):
            raise ValueError("Every annotation event must be a named mapping")
        start, end = event.get("start"), event.get("end")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 1
            or end < start
            or end > frame_count
        ):
            raise ValueError(f"Invalid annotation event {name}: {event}")
        events[name] = {"start": start, "end": end}
    return {"events": events, "notes": dict(notes)}
