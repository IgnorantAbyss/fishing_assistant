"""Validated, human-confirmed Hook Bar visibility annotations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import yaml


FORMAL_SESSION_IDS = (
    "session_20260709_192315",
    "session_20260710_061220",
    "session_20260710_123210",
    "session_20260710_124419",
    "session_20260710_125441",
    "session_20260710_130308",
    "session_20260710_131254",
)
TRIAL_SESSION_ID = "session_20260709_192231"


@dataclass(frozen=True)
class HookBarEpisodeGroundTruth:
    session_id: str
    episode_index: int
    global_hook_start: int
    global_hook_end: int
    visible_start: int
    visible_end: int
    first_clear_frame: int
    last_clear_frame: int
    confidence: str
    uncertain_frames: tuple[int, ...]
    notes: str

    def visible(self, frame_index: int) -> bool:
        return self.visible_start <= frame_index <= self.visible_end

    def clear(self, frame_index: int) -> bool:
        return self.first_clear_frame <= frame_index <= self.last_clear_frame


def _session_frame_count(session_root: Path, session_id: str) -> int:
    manifest = session_root / session_id / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"Missing replay manifest for {session_id}: {manifest}")
    return int(json.loads(manifest.read_text(encoding="utf-8"))["frame_count"])


def load_hook_bar_ground_truth(
    path: str | Path,
    *,
    session_root: str | Path | None = None,
) -> tuple[HookBarEpisodeGroundTruth, ...]:
    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("Hook Bar ground truth requires version: 1")
    if data.get("status") != "human_confirmed":
        raise ValueError("Hook Bar ground truth must be human_confirmed")
    if "human_review" not in str(data.get("source", "")):
        raise ValueError("Hook Bar source must record human review")
    sessions = data.get("sessions")
    if not isinstance(sessions, dict) or set(sessions) != set(FORMAL_SESSION_IDS):
        raise ValueError("Hook Bar ground truth must contain exactly the seven formal sessions")
    if TRIAL_SESSION_ID in sessions:
        raise ValueError("Trial session must not be Hook Bar ground truth")
    excluded = data.get("excluded_sessions", [])
    if TRIAL_SESSION_ID not in excluded:
        raise ValueError("Trial session must be explicitly excluded")

    root = Path(session_root) if session_root is not None else None
    parsed: list[HookBarEpisodeGroundTruth] = []
    for session_id in FORMAL_SESSION_IDS:
        raw_episodes = sessions[session_id].get("episodes")
        if not isinstance(raw_episodes, list) or not raw_episodes:
            raise ValueError(f"Missing Hook Bar episodes for {session_id}")
        frame_count = _session_frame_count(root, session_id) if root else None
        previous_global_end = 0
        previous_visible_end = 0
        for episode_index, raw in enumerate(raw_episodes, start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"Invalid Hook Bar episode in {session_id}")
            values = {
                key: raw.get(key) for key in (
                    "global_hook_start", "global_hook_end", "visible_start", "visible_end",
                    "first_clear_frame", "last_clear_frame",
                )
            }
            if any(not isinstance(value, int) or isinstance(value, bool) for value in values.values()):
                raise ValueError(f"Hook Bar ranges must be integer frames: {session_id} episode {episode_index}")
            global_start = values["global_hook_start"]
            global_end = values["global_hook_end"]
            visible_start = values["visible_start"]
            visible_end = values["visible_end"]
            clear_start = values["first_clear_frame"]
            clear_end = values["last_clear_frame"]
            if not (1 <= global_start <= global_end):
                raise ValueError(f"Invalid global Hook range: {session_id} episode {episode_index}")
            if not (global_start <= visible_start <= visible_end <= global_end):
                raise ValueError(f"Visible range must be inside global Hook range: {session_id} episode {episode_index}")
            if not (visible_start <= clear_start <= clear_end <= visible_end):
                raise ValueError(f"Clear range must be inside visible range: {session_id} episode {episode_index}")
            if global_start <= previous_global_end or visible_start <= previous_visible_end:
                raise ValueError(f"Overlapping Hook Bar episodes: {session_id} episode {episode_index}")
            if frame_count is not None and global_end > frame_count:
                raise ValueError(f"Hook Bar range exceeds {session_id} frame count")
            uncertain = raw.get("uncertain_frames", [])
            if not isinstance(uncertain, list) or any(
                not isinstance(frame, int) or frame < visible_start or frame > visible_end
                for frame in uncertain
            ):
                raise ValueError(f"Invalid uncertain frames: {session_id} episode {episode_index}")
            confidence = raw.get("confidence")
            if confidence not in {"high", "medium", "low"}:
                raise ValueError(f"Invalid Hook Bar confidence: {confidence}")
            parsed.append(HookBarEpisodeGroundTruth(
                session_id=session_id,
                episode_index=episode_index,
                global_hook_start=global_start,
                global_hook_end=global_end,
                visible_start=visible_start,
                visible_end=visible_end,
                first_clear_frame=clear_start,
                last_clear_frame=clear_end,
                confidence=confidence,
                uncertain_frames=tuple(uncertain),
                notes=str(raw.get("notes", "")),
            ))
            previous_global_end = global_end
            previous_visible_end = visible_end
    if len(parsed) != 9:
        raise ValueError(f"Expected 9 Hook Bar episodes, found {len(parsed)}")
    return tuple(parsed)
