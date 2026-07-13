"""Bounded event/session output for live detect-only observation."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import cv2


REVIEW_FIELDS = (
    "cycle_id", "ready_frame", "hook_would_fire_frame", "press_sequence",
    "get_frame", "final_state", "automatic_result", "human_result", "notes",
)
TRANSITION_FIELDS = (
    "timestamp", "frame_index", "previous_state", "next_state", "reason",
    "screenshot_reference",
)


def create_live_session_directory(root: str | Path, *, timestamp: datetime | None = None) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    moment = timestamp or datetime.now(timezone.utc)
    stem = f"session_{moment.strftime('%Y%m%d_%H%M%S')}"
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    (candidate / "screenshots").mkdir()
    return candidate


class LiveSessionLogger:
    def __init__(self, root: str | Path, *, bundle_version: str) -> None:
        self.path = create_live_session_directory(root)
        self.bundle_version = bundle_version
        self.events_path = self.path / "events.jsonl"
        self.transitions_path = self.path / "transitions.csv"
        self.review_path = self.path / "review_items.csv"
        self._event_counts: Counter[str] = Counter()
        self._cycles: dict[int, dict[str, Any]] = {}
        self._warnings: list[str] = []
        with self.transitions_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=TRANSITION_FIELDS).writeheader()

    def save_screenshot(self, frame: Any, frame_index: int, event_type: str) -> str:
        safe_type = "".join(char.lower() if char.isalnum() else "_" for char in event_type)
        relative = Path("screenshots") / f"{frame_index:06d}_{safe_type}.jpg"
        destination = self.path / relative
        if not cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 86]):
            raise OSError(f"Could not save live diagnostic screenshot: {destination}")
        return relative.as_posix()

    def event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        row = {"event_type": event_type, "bundle_version": self.bundle_version, **dict(payload)}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._event_counts[event_type] += 1
        if event_type in {"preflight_failure", "capture_failure", "SYNC_REQUIRED", "detector_conflict"}:
            self._warnings.append(str(payload.get("reason", event_type)))

    def transition(
        self,
        *,
        timestamp: float,
        frame_index: int,
        previous_state: str,
        next_state: str,
        reason: str,
        screenshot_reference: str | None,
    ) -> None:
        with self.transitions_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRANSITION_FIELDS)
            writer.writerow({
                "timestamp": timestamp,
                "frame_index": frame_index,
                "previous_state": previous_state,
                "next_state": next_state,
                "reason": reason,
                "screenshot_reference": screenshot_reference or "",
            })
        self.event("runtime_transition", {
            "timestamp": timestamp,
            "frame_index": frame_index,
            "previous_state": previous_state,
            "next_state": next_state,
            "reason": reason,
            "screenshot_reference": screenshot_reference,
        })

    def update_cycle(self, cycle_id: int, event_type: str, payload: Mapping[str, Any]) -> None:
        row = self._cycles.setdefault(cycle_id, {
            "cycle_id": cycle_id,
            "ready_frame": "",
            "hook_would_fire_frame": "",
            "press_sequence": "",
            "get_frame": "",
            "final_state": "",
            "automatic_result": "observed",
            "human_result": "",
            "notes": "",
        })
        frame = payload.get("frame_index", "")
        if event_type == "WOULD_START_HOOK":
            row["ready_frame"] = frame
        elif event_type == "WOULD_HOOK_ACTION":
            row["hook_would_fire_frame"] = frame
        elif event_type == "WOULD_PRESS_SEQUENCE":
            action_payload = payload.get("action_payload", {})
            sequence = action_payload.get("sequence", ()) if isinstance(action_payload, Mapping) else ()
            row["press_sequence"] = "".join(str(item) for item in sequence)
        elif event_type == "get_appearance":
            row["get_frame"] = frame
        if payload.get("runtime_state"):
            row["final_state"] = payload["runtime_state"]

    def finalize(self, summary: Mapping[str, Any]) -> None:
        cycles = [self._cycles[index] for index in sorted(self._cycles)]
        with self.review_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(cycles)
        complete = {
            **dict(summary),
            "bundle_version": self.bundle_version,
            "session_path": str(self.path),
            "event_counts": dict(self._event_counts),
            "warnings": self._warnings,
            "review_cycle_count": len(cycles),
            "events_path": str(self.events_path),
            "transitions_path": str(self.transitions_path),
            "review_items_path": str(self.review_path),
        }
        (self.path / "session_summary.json").write_text(
            json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        lines = [
            "# Live Detect-Only Session",
            "",
            f"- Result: **{complete.get('result', 'unknown')}**",
            f"- Bundle: `{self.bundle_version}`",
            f"- Frames captured / processed: **{complete.get('captured_frames', 0)} / {complete.get('processed_frames', 0)}**",
            f"- Capture FPS: **{complete.get('capture_fps', 0.0):.2f}**",
            f"- Mean processing latency: **{complete.get('mean_processing_latency_ms', 0.0):.2f} ms**",
            f"- Actions applied: **{complete.get('actions_applied', 0)}**",
            f"- Raw proposals: `{complete.get('raw_action_proposals', {})}`",
            f"- Unique would-fire: `{complete.get('unique_would_fire', {})}`",
            f"- Warnings: `{self._warnings}`",
            "",
            "## Cycle review",
            "",
            "| cycle | READY | HOOK would-fire | PRESS sequence | GET | final state | warning |",
            "|---:|---:|---:|---|---:|---|---|",
        ]
        for row in cycles:
            lines.append(
                f"| {row['cycle_id']} | {row['ready_frame']} | {row['hook_would_fire_frame']} | "
                f"{row['press_sequence']} | {row['get_frame']} | {row['final_state']} | {row['notes']} |"
            )
        if not cycles:
            lines.append("| - | - | - | - | - | - | no completed cycle evidence |")
        (self.path / "session_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
