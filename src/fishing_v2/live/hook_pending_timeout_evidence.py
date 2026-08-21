"""Bounded pre-qualification Hook evidence for HOOK_PENDING timeouts."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from src.fishing_v2.domain.observations import HookObservation, PromptObservation
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode
from src.fishing_v2.runtime.detector_evidence import EvidenceQualification


@dataclass(frozen=True)
class HookPendingTimeoutEvidenceConfig:
    enabled: bool = False
    max_samples: int = 140
    max_episodes: int = 20

    def __post_init__(self) -> None:
        if self.max_samples < 1 or self.max_episodes < 1:
            raise ValueError("HOOK_PENDING evidence bounds must be positive")


@dataclass(frozen=True)
class _Sample:
    frame_index: int
    timestamp: float
    pixels: np.ndarray
    metadata: Mapping[str, Any]


@dataclass
class _Episode:
    episode_id: str
    cycle_id: int
    ready_identity: str | None
    start_hook_opportunity_id: str | None
    start_hook_action_id: str | None
    start_hook_emission_started_at: float
    start_hook_applied_at: float
    hook_pending_started_at: float
    samples: deque[_Sample]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class HookPendingTimeoutEvidenceRecorder:
    """Keep native Hook search ROIs in RAM and dump timeout episodes only."""

    def __init__(
        self,
        session_path: str | Path,
        config: HookPendingTimeoutEvidenceConfig | None = None,
    ) -> None:
        self.config = config or HookPendingTimeoutEvidenceConfig()
        self.root = Path(session_path) / "hook_pending_anomalies"
        self._episode_sequence = 0
        self._active: _Episode | None = None
        self._dump_attempt_count = 0
        self._saved_count = 0
        self._cap_drop_count = 0
        self._failures: list[str] = []
        self._completed_events: list[dict[str, Any]] = []
        self._futures: list[Future[dict[str, Any]]] = []
        self._executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="hook-pending-evidence-writer",
            )
            if self.config.enabled else None
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def active(self) -> bool:
        return self._active is not None

    @property
    def sample_count(self) -> int:
        return len(self._active.samples) if self._active is not None else 0

    @property
    def worst_case_ram_bytes(self) -> int | None:
        if self._active is None or not self._active.samples:
            return None
        return int(
            self._active.samples[0].pixels.nbytes * self.config.max_samples
        )

    def begin(
        self,
        *,
        runtime_state: RuntimeState,
        cycle_id: int,
        ready_identity: str | None,
        start_hook_opportunity_id: str | None,
        start_hook_action_id: str | None,
        start_hook_emission_started_at: float | None,
        start_hook_applied_at: float | None,
        hook_pending_started_at: float,
        start_hook_action_applied: bool,
    ) -> bool:
        """Start only at the authoritative applied START_HOOK boundary."""
        if (
            not self.enabled
            or runtime_state != RuntimeState.HOOK_PENDING
            or not start_hook_action_applied
            or start_hook_emission_started_at is None
            or start_hook_applied_at is None
            or self._active is not None
        ):
            return False
        self._episode_sequence += 1
        self._active = _Episode(
            episode_id=f"hook_pending:{self._episode_sequence}",
            cycle_id=int(cycle_id),
            ready_identity=(
                str(ready_identity) if ready_identity is not None else None
            ),
            start_hook_opportunity_id=(
                str(start_hook_opportunity_id)
                if start_hook_opportunity_id is not None else None
            ),
            start_hook_action_id=(
                str(start_hook_action_id)
                if start_hook_action_id is not None else None
            ),
            start_hook_emission_started_at=float(
                start_hook_emission_started_at
            ),
            start_hook_applied_at=float(start_hook_applied_at),
            hook_pending_started_at=float(hook_pending_started_at),
            samples=deque(maxlen=self.config.max_samples),
        )
        return True

    @staticmethod
    def _raw_payload(
        observation: HookObservation | None,
        *,
        evaluated: bool,
    ) -> dict[str, Any]:
        if not evaluated or observation is None:
            return {
                "status": "not_evaluated",
                "reason": "hook_detector_not_executed",
            }
        evidence = dict(observation.evidence)
        debug = evidence.get("legacy_debug")
        debug = dict(debug) if isinstance(debug, Mapping) else {}
        raw_values = debug.get("raw_values")
        raw_values = (
            dict(raw_values) if isinstance(raw_values, Mapping) else {}
        )
        features = tuple(evidence.get("matched_features", ()))
        divider_found = bool(evidence.get("divider_line_detected"))
        fill_endpoint = evidence.get("fill_endpoint_x")
        fill_found = bool(fill_endpoint is not None or "bar_fill" in features)
        bar_bbox = evidence.get("bar_bbox")
        geometry_context_valid = bool(
            evidence.get("crossing_geometry_version") == 1
        )
        geometry_valid = bool(
            geometry_context_valid
            and divider_found
            and evidence.get("divider_line_x") is not None
            and fill_endpoint is not None
        )
        return _json_safe({
            "status": "evaluated",
            "hook_evidence_version": evidence.get(
                "crossing_geometry_version"
            ),
            "detected": observation.detected,
            "confidence": observation.confidence,
            "hook_roi": debug.get("roi_name"),
            "bar_bbox": bar_bbox,
            "locator_success": bar_bbox is not None,
            "locator_rejection_reason": (
                None if bar_bbox is not None
                else evidence.get("exception") or "bar_bbox_not_found"
            ),
            "geometry_context_valid": geometry_context_valid,
            "matched_features": list(features),
            "hook_instruction_evidence": (
                "hook_prompt_match" in features
            ),
            "hook_prompt_score": raw_values.get("hook_prompt_score"),
            "red_mask_pixel_count": raw_values.get("red_pixels"),
            "cyan_fill_mask_pixel_count": raw_values.get("cyan_pixels"),
            "connected_component_count": None,
            "divider_found": divider_found,
            "divider_candidate_count": None,
            "divider_x": evidence.get("divider_line_x"),
            "divider_confidence": evidence.get("divider_confidence"),
            "divider_rejection_reason": (
                None if divider_found else "divider_line_not_detected"
            ),
            "fill_found": fill_found,
            "fill_candidate_count": None,
            "fill_start_x": None,
            "fill_end_x": fill_endpoint,
            "fill_endpoint_x": fill_endpoint,
            "fill_ratio": observation.fill_ratio,
            "fill_rejection_reason": (
                None if fill_found else "fill_not_detected"
            ),
            "divider_fill_relationship": {
                "divider_x": evidence.get("divider_line_x"),
                "fill_endpoint_x": fill_endpoint,
                "endpoint_minus_divider_px": (
                    float(fill_endpoint)
                    - float(evidence["divider_line_x"])
                    if fill_endpoint is not None
                    and evidence.get("divider_line_x") is not None
                    else None
                ),
            },
            "divider_margin_passed": evidence.get(
                "divider_margin_passed"
            ),
            "geometry_valid": geometry_valid,
            "legacy_debug_raw_values": raw_values,
        })

    @staticmethod
    def _qualification_payload(
        observation: HookObservation | None,
        qualification: EvidenceQualification | None,
        *,
        evaluated: bool,
        fsm_diagnostics: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not evaluated or qualification is None:
            return {
                "status": "not_evaluated",
                "reason": "hook_detector_not_executed",
            }
        kind = (
            qualification.hook_evidence_kind.value
            if qualification.hook_evidence_kind is not None else None
        )
        strong_candidate = bool(
            qualification.raw_detected
            and kind == "ACTIVE_HOOK_BAR"
        )
        transition_reason = fsm_diagnostics.get("transition_reason")
        return _json_safe({
            "status": "evaluated",
            "raw_detected": qualification.raw_detected,
            "strong_hook_candidate": strong_candidate,
            "strong_hook_evidence": qualification.qualified_detected,
            "strong_hook_confidence": (
                observation.confidence if observation is not None else None
            ),
            "used_by_fusion": qualification.used_by_fusion,
            "hook_evidence_kind": kind,
            "temporal_candidate": fsm_diagnostics.get(
                "recommended_state"
            ),
            # FishingFSM currently exposes the result/reason, not the private
            # counter. Keep absence explicit instead of inventing telemetry.
            "stability_count": None,
            "stable_strong_hook_evidence": bool(
                transition_reason == "stable_strong_hook_evidence"
            ),
            "qualification_reason": qualification.qualification_reason,
            "abstain_reason": (
                None if qualification.qualified_detected
                else qualification.qualification_reason
            ),
            "fsm": dict(fsm_diagnostics),
        })

    def record(
        self,
        *,
        frame_index: int,
        timestamp: float,
        runtime_state: RuntimeState,
        roi_pixels: np.ndarray | None,
        hook_detector_executed: bool,
        detector_mode: DetectorActivationMode,
        cadence: Mapping[str, Any],
        prompt: PromptObservation | None,
        raw_observation: HookObservation | None,
        qualified_observation: HookObservation | None,
        qualification: EvidenceQualification | None,
        fsm_diagnostics: Mapping[str, Any],
    ) -> bool:
        episode = self._active
        if (
            episode is None
            or roi_pixels is None
            or roi_pixels.size == 0
        ):
            return False
        evaluated = bool(hook_detector_executed)
        prompt_age = (
            max(0.0, float(timestamp) - float(prompt.timestamp))
            if prompt is not None else None
        )
        metadata = {
            "frame_index": int(frame_index),
            "monotonic_timestamp": float(timestamp),
            "hook_pending_age": max(
                0.0, float(timestamp) - episode.hook_pending_started_at
            ),
            "runtime_state": runtime_state.value,
            "raw_roi_shape": list(roi_pixels.shape),
            "raw_roi_bytes": int(roi_pixels.nbytes),
            "hook_detector_executed": evaluated,
            "detector_mode": detector_mode.value,
            "detector_scheduling": _json_safe(dict(cadence)),
            "prompt": {
                "status": "evaluated" if prompt is not None else "not_evaluated",
                "label": prompt.kind.value if prompt is not None else None,
                "hook_instruction": bool(
                    prompt is not None
                    and prompt.kind.value == "HOOK_INSTRUCTION"
                ),
                "confidence": (
                    prompt.confidence if prompt is not None else None
                ),
                "observation_frame_index": (
                    prompt.frame_index if prompt is not None else None
                ),
                "observation_age_seconds": prompt_age,
                "fresh_on_capture_frame": bool(
                    prompt is not None
                    and prompt.frame_index == int(frame_index)
                ),
            },
            "raw_observation": self._raw_payload(
                raw_observation,
                evaluated=evaluated,
            ),
            "qualification": self._qualification_payload(
                qualified_observation,
                qualification,
                evaluated=evaluated,
                fsm_diagnostics=fsm_diagnostics,
            ),
        }
        episode.samples.append(_Sample(
            frame_index=int(frame_index),
            timestamp=float(timestamp),
            pixels=roi_pixels.copy(),
            metadata=metadata,
        ))
        return True

    def finish(
        self,
        *,
        next_state: RuntimeState,
        timestamp: float,
        reason: str,
    ) -> bool:
        episode = self._active
        if episode is None:
            return False
        self._active = None
        if not (
            next_state == RuntimeState.SYNC_REQUIRED
            and reason == "hook_pending_timeout"
        ):
            return False
        if (
            self._executor is None
            or self._dump_attempt_count >= self.config.max_episodes
        ):
            self._cap_drop_count += 1
            return False
        self._dump_attempt_count += 1
        snapshot = tuple(episode.samples)
        self._futures.append(self._executor.submit(
            self._write_episode,
            episode,
            snapshot,
            next_state,
            float(timestamp),
            str(reason),
        ))
        return True

    def _write_episode(
        self,
        episode: _Episode,
        samples: tuple[_Sample, ...],
        next_state: RuntimeState,
        timeout_at: float,
        reason: str,
    ) -> dict[str, Any]:
        safe_id = episode.episode_id.replace(":", "_")
        destination = self.root / f"episode_{safe_id.split('_')[-1]}"
        destination.mkdir(parents=True, exist_ok=True)
        sample_payloads: list[dict[str, Any]] = []
        for sample in samples:
            filename = f"frame_{sample.frame_index:06d}_raw.png"
            if not cv2.imwrite(str(destination / filename), sample.pixels):
                raise OSError(f"Failed to write {filename}")
            sample_payloads.append({
                **dict(sample.metadata),
                "raw_image_filename": filename,
            })

        def count_raw(key: str) -> int:
            return sum(
                item["raw_observation"].get(key) is True
                for item in sample_payloads
            )

        def first_age(section: str, key: str) -> float | None:
            return next((
                float(item["hook_pending_age"])
                for item in sample_payloads
                if item[section].get(key) is True
            ), None)

        detector_executed_count = sum(
            item["hook_detector_executed"] for item in sample_payloads
        )
        hook_instruction_seen_count = sum(
            item["prompt"].get("hook_instruction") is True
            or item["raw_observation"].get(
                "hook_instruction_evidence"
            ) is True
            for item in sample_payloads
        )
        divider_seen_count = count_raw("divider_found")
        fill_seen_count = count_raw("fill_found")
        geometry_valid_count = count_raw("geometry_valid")
        strong_candidate_count = sum(
            item["qualification"].get("strong_hook_candidate") is True
            for item in sample_payloads
        )
        strong_evidence_count = sum(
            item["qualification"].get("strong_hook_evidence") is True
            for item in sample_payloads
        )
        stable_seen = any(
            item["qualification"].get(
                "stable_strong_hook_evidence"
            ) is True
            for item in sample_payloads
        )
        duration = max(0.0, timeout_at - episode.hook_pending_started_at)
        timestamps = [sample.timestamp for sample in samples]
        intervals = [
            (right - left) * 1000.0
            for left, right in zip(timestamps, timestamps[1:])
        ]
        roi_dimensions = (
            [int(samples[0].pixels.shape[1]), int(samples[0].pixels.shape[0])]
            if samples else None
        )
        manifest = {
            "anomaly_type": "hook_pending_timeout",
            "episode_id": episode.episode_id,
            "cycle_id": episode.cycle_id,
            "ready_identity": episode.ready_identity,
            "start_hook_opportunity_id": episode.start_hook_opportunity_id,
            "start_hook_action_id": episode.start_hook_action_id,
            "start_hook_emission_started_at": (
                episode.start_hook_emission_started_at
            ),
            "start_hook_action_applied": True,
            "start_hook_emission_completed_at": episode.start_hook_applied_at,
            "hook_pending_started_at": episode.hook_pending_started_at,
            "timeout_at": timeout_at,
            "duration": duration,
            "sample_target_fps": (
                sample_payloads[0]["detector_scheduling"].get(
                    "target_fps"
                ) if sample_payloads else None
            ),
            "sample_observed_fps": (
                len(samples) / duration if duration > 0.0 else 0.0
            ),
            "sample_count": len(samples),
            "max_samples": self.config.max_samples,
            "raw_roi_dimensions": roi_dimensions,
            "frame_interval_mean_ms": (
                float(np.mean(intervals)) if intervals else None
            ),
            "frame_interval_p95_ms": (
                float(np.percentile(intervals, 95)) if intervals else None
            ),
            "frame_interval_max_ms": max(intervals) if intervals else None,
            "final_transition": {
                "from": RuntimeState.HOOK_PENDING.value,
                "to": next_state.value,
            },
            "terminal_reason": reason,
            "episode_summary": {
                "hook_detector_executed_count": detector_executed_count,
                "hook_instruction_seen_count": hook_instruction_seen_count,
                "divider_seen_count": divider_seen_count,
                "fill_seen_count": fill_seen_count,
                "geometry_valid_count": geometry_valid_count,
                "strong_hook_candidate_count": strong_candidate_count,
                "strong_hook_evidence_count": strong_evidence_count,
                "stable_strong_hook_evidence_seen": stable_seen,
                "first_divider_seen_age": first_age(
                    "raw_observation", "divider_found"
                ),
                "first_fill_seen_age": first_age(
                    "raw_observation", "fill_found"
                ),
                "first_strong_candidate_age": first_age(
                    "qualification", "strong_hook_candidate"
                ),
                "first_hook_instruction_age": next((
                    float(item["hook_pending_age"])
                    for item in sample_payloads
                    if item["prompt"].get("hook_instruction") is True
                    or item["raw_observation"].get(
                        "hook_instruction_evidence"
                    ) is True
                ), None),
            },
            "samples": sample_payloads,
        }
        (destination / "manifest.json").write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "episode_id": episode.episode_id,
            "sample_count": len(samples),
            "duration": duration,
            "evidence_path": str(destination),
            "detector_executed_count": detector_executed_count,
            "divider_seen_count": divider_seen_count,
            "fill_seen_count": fill_seen_count,
            "strong_candidate_count": strong_candidate_count,
            "stable_strong_hook_evidence_seen": stable_seen,
        }

    def poll_completed(self, *, force: bool = False) -> None:
        remaining: list[Future[dict[str, Any]]] = []
        for future in self._futures:
            if not force and not future.done():
                remaining.append(future)
                continue
            try:
                event = future.result()
            except Exception as exc:  # evidence must not affect Runtime
                self._failures.append(f"{type(exc).__name__}: {exc}")
            else:
                self._saved_count += 1
                self._completed_events.append(event)
        self._futures = remaining

    def drain_completed_events(self) -> tuple[dict[str, Any], ...]:
        events = tuple(self._completed_events)
        self._completed_events.clear()
        return events

    def close(self) -> dict[str, Any]:
        self._active = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self.poll_completed(force=True)
        return {
            "hook_pending_timeout_evidence_saved_count": self._saved_count,
            "hook_pending_timeout_evidence_failure_count": len(
                self._failures
            ),
            "hook_pending_timeout_evidence_failures": list(
                self._failures
            ),
            "hook_pending_timeout_evidence_cap_drop_count": (
                self._cap_drop_count
            ),
            "hook_pending_timeout_evidence_path": (
                str(self.root) if self._saved_count else None
            ),
            "hook_pending_timeout_max_samples": self.config.max_samples,
            "hook_pending_timeout_max_episodes": self.config.max_episodes,
        }
