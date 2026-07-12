"""Validated, human-authored PRESS sequence episode annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PressSequenceGroundTruth:
    session_id: str
    press_start: int
    press_end: int
    sequence: tuple[str, ...]
    status: str
    source: str


def load_press_sequence_ground_truth(
    path: str | Path,
) -> tuple[PressSequenceGroundTruth, ...]:
    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("PRESS sequence ground truth requires version: 1")
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("PRESS sequence ground truth requires episodes")
    parsed: list[PressSequenceGroundTruth] = []
    seen: set[tuple[str, int, int]] = set()
    for raw in episodes:
        if not isinstance(raw, dict):
            raise ValueError("Every PRESS sequence episode must be a mapping")
        session_id = raw.get("session_id")
        start, end = raw.get("press_start"), raw.get("press_end")
        sequence = raw.get("sequence")
        status, annotation_source = raw.get("status"), raw.get("source")
        if not isinstance(session_id, str) or not session_id.startswith("session_"):
            raise ValueError(f"Invalid PRESS session id: {session_id}")
        if not isinstance(start, int) or isinstance(start, bool) or start < 1:
            raise ValueError(f"Invalid PRESS start: {start}")
        if not isinstance(end, int) or isinstance(end, bool) or end < start:
            raise ValueError(f"Invalid PRESS end: {end}")
        if not isinstance(sequence, str) or not sequence or any(key not in "WASD" for key in sequence):
            raise ValueError(f"Invalid confirmed PRESS sequence: {sequence}")
        if status != "confirmed" or annotation_source != "manual_original_frame_review":
            raise ValueError("Only manually confirmed original-frame reviews are ground truth")
        identity = (session_id, start, end)
        if identity in seen:
            raise ValueError(f"Duplicate PRESS sequence episode: {identity}")
        seen.add(identity)
        parsed.append(PressSequenceGroundTruth(
            session_id, start, end, tuple(sequence), status, annotation_source
        ))
    return tuple(parsed)
