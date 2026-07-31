"""Bounded event/session output for live detect-only observation."""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Mapping

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
        self._event_listener: Callable[[str, Mapping[str, Any]], None] | None = None
        self._logging_disabled = False
        self._logging_failure_reason: str | None = None
        try:
            with self.transitions_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                csv.DictWriter(
                    handle, fieldnames=TRANSITION_FIELDS
                ).writeheader()
        except OSError as exc:
            self._disable_logging("transitions_header", exc)

    @property
    def logging_disabled(self) -> bool:
        return self._logging_disabled

    @property
    def logging_failure_reason(self) -> str | None:
        return self._logging_failure_reason

    def _disable_logging(self, operation: str, exc: BaseException) -> None:
        if self._logging_disabled:
            return
        self._logging_disabled = True
        self._logging_failure_reason = (
            f"{operation}: {type(exc).__name__}: {exc}"
        )
        self._warnings.append(self._logging_failure_reason)

    def set_event_listener(
        self, listener: Callable[[str, Mapping[str, Any]], None] | None
    ) -> None:
        self._event_listener = listener

    def save_screenshot(self, frame: Any, frame_index: int, event_type: str) -> str:
        if self._logging_disabled:
            return ""
        safe_type = "".join(char.lower() if char.isalnum() else "_" for char in event_type)
        relative = Path("screenshots") / f"{frame_index:06d}_{safe_type}.jpg"
        destination = self.path / relative
        try:
            if not cv2.imwrite(
                str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 86]
            ):
                raise OSError(
                    f"Could not save live diagnostic screenshot: {destination}"
                )
        except OSError as exc:
            self._disable_logging("screenshot", exc)
            return ""
        return relative.as_posix()

    def event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        row = {"event_type": event_type, "bundle_version": self.bundle_version, **dict(payload)}
        self._event_counts[event_type] += 1
        if event_type in {
            "preflight_failure", "preflight_failed", "capture_failure", "capture_backend_fallback",
            "capture_backend_warning", "SYNC_REQUIRED", "detector_conflict",
        }:
            self._warnings.append(str(payload.get("reason", event_type)))
        if self._logging_disabled:
            return
        try:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
            if self._event_listener is not None:
                self._event_listener(event_type, payload)
        except (OSError, RuntimeError) as exc:
            self._disable_logging("event", exc)

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
        if not self._logging_disabled:
            try:
                with self.transitions_path.open(
                    "a", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=TRANSITION_FIELDS
                    )
                    writer.writerow({
                        "timestamp": timestamp,
                        "frame_index": frame_index,
                        "previous_state": previous_state,
                        "next_state": next_state,
                        "reason": reason,
                        "screenshot_reference": screenshot_reference or "",
                    })
            except OSError as exc:
                self._disable_logging("transition", exc)
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
            "logging_disabled": self._logging_disabled,
            "logging_failure_reason": self._logging_failure_reason,
        }
        if not self._logging_disabled:
            try:
                with self.review_path.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=REVIEW_FIELDS
                    )
                    writer.writeheader()
                    writer.writerows(cycles)
                (self.path / "session_summary.json").write_text(
                    json.dumps(complete, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                self._disable_logging("finalize", exc)
        lines = [
            "# Live Detect-Only Session",
            "",
            f"- Result: **{complete.get('result', 'unknown')}**",
            f"- Preflight passed / reason: **{complete.get('preflight_passed', False)} / {complete.get('preflight_failure_reason')}**",
            f"- Bundle: `{self.bundle_version}`",
            f"- Frames captured / processed: **{complete.get('captured_frames', 0)} / {complete.get('processed_frames', 0)}**",
            f"- Capture FPS: **{complete.get('capture_fps', 0.0):.2f}**",
            f"- Mean processing latency: **{complete.get('mean_processing_latency_ms', 0.0):.2f} ms**",
            f"- Actions applied: **{complete.get('actions_applied', 0)}**",
            f"- Action sink / emit: **{complete.get('action_sink_type', 'none')} / {complete.get('emit_actions', False)}**",
            f"- Action allowlist: `{complete.get('action_allowlist', [])}`",
            f"- Proposed actions: `{complete.get('proposed_action_counts', {})}`",
            f"- Attempted actions: `{complete.get('attempted_action_counts', {})}`",
            f"- Applied actions: `{complete.get('applied_action_counts', {})}`",
            f"- Rejected actions: `{complete.get('rejected_action_counts', {})}`",
            f"- Partial actions: `{complete.get('partial_action_counts', {})}`",
            f"- Failed actions: `{complete.get('failed_action_counts', {})}`",
            f"- OS input emitted: `{complete.get('os_input_emitted_counts', {})}`",
            f"- Action applied semantics: `{complete.get('action_applied_semantics', 'unknown')}`",
            f"- COLLECT visual acknowledged: **{complete.get('collect_visual_acknowledged', False)}**",
            f"- COLLECT completed / visual timeout: **{complete.get('collect_completed_count', 0)} / {complete.get('collect_visual_timeout_count', 0)}**",
            f"- COLLECT attempts / retries: `{complete.get('collect_attempt_counts', {})}` / `{complete.get('collect_retry_counts', {})}`",
            f"- Physical GET episodes / COLLECT opportunities / terminal episodes: **{complete.get('physical_get_episode_count', 0)} / {complete.get('collect_opportunity_count', 0)} / {complete.get('collect_terminal_episode_count', 0)}**",
            f"- COLLECT attempts by physical GET episode: `{complete.get('collect_attempt_counts_by_get_episode', {})}`",
            f"- CAST opportunities / attempts: **{complete.get('cast_opportunity_count', 0)} / {complete.get('cast_attempt_count', 0)}**",
            f"- Post-cycle clearances: **{complete.get('post_cycle_clearance_count', 0)}** `{complete.get('post_cycle_clearance_counts_by_source', {})}`",
            f"- Post-collect clearances: **{complete.get('post_collect_clearance_count', 0)}**",
            f"- No-GET clearances: **{complete.get('no_get_clearance_count', 0)}**",
            f"- CAST visual acknowledged / timeout: **{complete.get('cast_visual_acknowledged_count', 0)} / {complete.get('cast_timeout_count', 0)}**",
            f"- CAST terminal outcomes / pending: `{complete.get('cast_terminal_outcome_counts', {})}` / **{complete.get('cast_pending_count', 0)}**",
            f"- Action rejections by reason: `{complete.get('rejection_counts_by_reason', {})}`",
            f"- Panic / focus loss: **{complete.get('panic_triggered', False)} / {complete.get('focus_loss_count', 0)}**",
            f"- Capture backend: **{complete.get('capture_backend', 'unknown')}**",
            f"- MSS fallback used: **{complete.get('capture_fallback_used', False)}**",
            f"- Evidence mode: **{complete.get('evidence_mode', 'minimal')}**",
            f"- Diagnostic video: `{complete.get('video_path')}`",
            f"- Requested / actual video codec: **{complete.get('requested_video_codec')} / {complete.get('actual_video_codec')}**",
            f"- Video codec attempts: `{complete.get('attempted_codecs', [])}`",
            f"- Video codec fallback used: **{complete.get('video_codec_fallback_used', False)}**",
            f"- Video codec initialization errors: `{complete.get('codec_initialization_errors', [])}`",
            f"- Diagnostic video frames / FPS: **{complete.get('video_frame_count', 0)} / {complete.get('video_fps', 0.0)}**",
            f"- Diagnostic video timestamps: **{complete.get('first_timestamp')} – {complete.get('last_timestamp')}**",
            f"- Dropped diagnostic video frames: **{complete.get('dropped_video_frames', 0)}**",
            f"- Full video suspended during Hook critical: **{complete.get('full_video_suspended_during_hook_critical', False)}**",
            f"- Hook video suspension episodes: **{complete.get('full_video_suspension_episode_count', 0)}** `{complete.get('full_video_suspensions', [])}`",
            f"- PRESS ROI clip / frames: `{complete.get('press_roi_clip_path')}` / **{complete.get('press_roi_frame_count', 0)}**",
            f"- PRESS ROI pre-roll / post-roll: **{complete.get('press_roi_pre_roll_seconds', 0.0)} / {complete.get('press_roi_post_roll_seconds', 0.0)} seconds**",
            f"- PRESS initial delay range: `{complete.get('press_initial_delay_range_ms', [])}` ms",
            f"- PRESS key-up to next key-down gap range: `{complete.get('press_inter_key_gap_range_ms', [])}` ms",
            f"- PRESS key hold: **{complete.get('press_key_hold_ms', 0)} ms**",
            f"- ROI evidence by episode: `{complete.get('roi_evidence_counts_by_episode', {})}`",
            f"- Evidence gaps: **{complete.get('has_evidence_gaps', False)}** `{complete.get('evidence_gap_intervals', [])}`",
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
        if not self._logging_disabled:
            try:
                (self.path / "session_summary.md").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
            except OSError as exc:
                self._disable_logging("summary_markdown", exc)
