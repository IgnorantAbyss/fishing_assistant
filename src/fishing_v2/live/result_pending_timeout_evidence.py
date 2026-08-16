"""Bounded pre-qualification PRESS evidence for RESULT_PENDING timeouts."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from src.fishing_v2.domain.observations import PressObservation
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode
from src.fishing_v2.runtime.detector_evidence import EvidenceQualification


@dataclass(frozen=True)
class ResultPendingTimeoutEvidenceConfig:
    enabled: bool = False
    sample_interval_seconds: float = 0.2
    max_samples: int = 60
    max_episodes: int = 20

    def __post_init__(self) -> None:
        if self.sample_interval_seconds <= 0.0:
            raise ValueError("sample_interval_seconds must be positive")
        if self.max_samples < 1 or self.max_episodes < 1:
            raise ValueError("RESULT_PENDING evidence bounds must be positive")


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
    hook_action_identity: str | None
    started_at: float
    samples: deque[_Sample]
    next_sample_at: float


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


class ResultPendingTimeoutEvidenceRecorder:
    """Copy sampled PRESS ROIs to RAM and write only maximum-timeout cases."""

    def __init__(
        self,
        session_path: str | Path,
        config: ResultPendingTimeoutEvidenceConfig | None = None,
    ) -> None:
        self.config = config or ResultPendingTimeoutEvidenceConfig()
        self.root = Path(session_path) / "result_pending_anomalies"
        self._episode_sequence = 0
        self._active: _Episode | None = None
        self._timeout_count = 0
        self._dump_attempt_count = 0
        self._saved_count = 0
        self._cap_drop_count = 0
        self._failures: list[str] = []
        self._completed_events: list[dict[str, Any]] = []
        self._futures: list[Future[dict[str, Any]]] = []
        self._executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="result-pending-evidence-writer",
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
        return int(self._active.samples[0].pixels.nbytes * self.config.max_samples)

    def begin(
        self,
        *,
        runtime_state: RuntimeState,
        started_at: float,
        cycle_id: int,
        hook_action_identity: str | None,
    ) -> bool:
        if (
            not self.enabled
            or runtime_state != RuntimeState.RESULT_PENDING
            or self._active is not None
        ):
            return False
        self._episode_sequence += 1
        self._active = _Episode(
            episode_id=f"result_pending:{self._episode_sequence}",
            cycle_id=int(cycle_id),
            hook_action_identity=(
                str(hook_action_identity)
                if hook_action_identity is not None else None
            ),
            started_at=float(started_at),
            samples=deque(maxlen=self.config.max_samples),
            next_sample_at=float(started_at),
        )
        return True

    @staticmethod
    def _raw_payload(
        observation: PressObservation | None,
        *,
        evaluated: bool,
    ) -> dict[str, Any]:
        if not evaluated or observation is None:
            return {
                "status": "not_evaluated",
                "reason": "press_detector_not_executed",
            }
        evidence = dict(observation.evidence)
        v3 = evidence.get("v3")
        v3 = dict(v3) if isinstance(v3, Mapping) else {}
        slots_value = v3.get("slots", evidence.get("slots", ()))
        slots = list(slots_value) if isinstance(slots_value, (list, tuple)) else []
        occupied_mask = [
            str(slot.get("occupancy", "")) == "OCCUPIED"
            for slot in slots if isinstance(slot, Mapping)
        ]
        panel_found = bool(
            v3.get("panel_present", observation.panel_present)
        )
        return _json_safe({
            "status": "evaluated",
            "press_evidence_version": evidence.get(
                "press_evidence_version"
            ),
            "detected": observation.detected,
            "confidence": observation.confidence,
            "panel_found": panel_found,
            "panel_bbox": (
                v3.get("key_strip_bbox")
                or evidence.get("panel_bbox")
            ),
            "panel_geometry_valid": bool(
                v3.get("geometry_stable", panel_found)
            ),
            "sequence_candidate": list(observation.sequence_candidate),
            "occupied_mask": occupied_mask,
            "occupied_prefix_length": int(
                v3.get("occupied_slot_count", sum(occupied_mask)) or 0
            ),
            "frame_complete": bool(v3.get("frame_complete", False)),
            "classification_confidence": float(
                v3.get(
                    "sequence_confidence",
                    observation.sequence_confidence,
                ) or 0.0
            ),
            "slots": slots,
            "rejection_reason": (
                v3.get("rejection_reason")
                or observation.sequence_qualification_reason
                or observation.panel_qualification_reason
            ),
        })

    @staticmethod
    def _qualification_payload(
        observation: PressObservation | None,
        qualification: EvidenceQualification | None,
        *,
        evaluated: bool,
    ) -> dict[str, Any]:
        if not evaluated or qualification is None:
            return {
                "status": "not_evaluated",
                "reason": "press_detector_not_executed",
            }
        evidence = (
            dict(observation.evidence)
            if observation is not None else {}
        )
        completeness = evidence.get("press_completeness_certificate")
        completeness = (
            dict(completeness) if isinstance(completeness, Mapping) else None
        )
        return _json_safe({
            "status": "evaluated",
            "press_evidence_version": evidence.get(
                "press_evidence_version"
            ),
            "sequence_stability_count": (
                completeness.get("sequence_stability_count", 0)
                if completeness is not None else 0
            ),
            "occupancy_stability_count": (
                completeness.get("occupancy_stability_count", 0)
                if completeness is not None else 0
            ),
            "temporal_candidate": list(
                observation.sequence_candidate
                if observation is not None else ()
            ),
            "sequence_ready": bool(qualification.sequence_ready),
            "completeness_certificate": completeness,
            "completeness_state": (
                "complete"
                if completeness and completeness.get("complete") is True
                else "incomplete"
                if completeness is not None else "not_available"
            ),
            "qualification_reason": qualification.qualification_reason,
            "abstain_reason": (
                None if qualification.sequence_ready
                else qualification.sequence_qualification_reason
            ),
            "frozen_sequence": list(
                evidence.get("frozen_sequence", ())
            ),
            "press_evidence_kind": (
                qualification.press_evidence_kind.value
                if qualification.press_evidence_kind is not None else None
            ),
        })

    @staticmethod
    def _input_effect_payload(
        observation: PressObservation | None,
        *,
        evaluated: bool,
    ) -> dict[str, Any]:
        if not evaluated or observation is None:
            return {
                "status": "not_evaluated",
                "reason": "press_detector_not_executed",
            }
        evidence = dict(observation.evidence)
        v3 = evidence.get("v3")
        v3 = dict(v3) if isinstance(v3, Mapping) else {}
        effect = v3.get("input_effect_evidence")
        effect = dict(effect) if isinstance(effect, Mapping) else {}
        slot_deltas = effect.get(
            "per_slot_deltas",
            v3.get("per_slot_input_effect_deltas", ()),
        )
        changed_slots = [
            item.get("slot_index")
            for item in slot_deltas
            if isinstance(item, Mapping)
            and item.get("diffuse_halo_or_flash") is True
        ]
        return _json_safe({
            "status": "evaluated",
            "input_effect_detected": bool(
                v3.get("input_effect_detected", False)
            ),
            "input_started": bool(
                v3.get("episode_input_started", False)
            ),
            "post_input_frame": bool(v3.get("post_input_frame", False)),
            "panel_phase": v3.get(
                "panel_phase", evidence.get("panel_phase")
            ),
            "frame_local_effect": effect.get("frame_local_effect"),
            "temporal_effect": effect.get("temporal_baseline_effect"),
            "changed_slots": changed_slots,
            "effect_reason": (
                effect.get("reason") or v3.get("input_effect_reason")
            ),
        })

    def record(
        self,
        *,
        frame_index: int,
        timestamp: float,
        runtime_state: RuntimeState,
        roi_pixels: np.ndarray | None,
        press_detector_executed: bool,
        detector_mode: DetectorActivationMode,
        cadence: Mapping[str, Any],
        raw_observation: PressObservation | None,
        qualified_observation: PressObservation | None,
        qualification: EvidenceQualification | None,
        get_evidence_seen: bool,
    ) -> bool:
        episode = self._active
        if (
            episode is None
            or roi_pixels is None
            or roi_pixels.size == 0
            or float(timestamp) + 1e-9 < episode.next_sample_at
        ):
            return False
        evaluated = bool(press_detector_executed)
        metadata = {
            "frame_index": int(frame_index),
            "monotonic_timestamp": float(timestamp),
            "result_pending_age": max(
                0.0, float(timestamp) - episode.started_at
            ),
            "runtime_state": runtime_state.value,
            "raw_roi_shape": list(roi_pixels.shape),
            "raw_roi_bytes": int(roi_pixels.nbytes),
            "press_detector_executed": evaluated,
            "detector_mode": detector_mode.value,
            "detector_scheduling": _json_safe(dict(cadence)),
            "raw_observation": self._raw_payload(
                raw_observation,
                evaluated=evaluated,
            ),
            "qualification": self._qualification_payload(
                qualified_observation,
                qualification,
                evaluated=evaluated,
            ),
            "input_effect": self._input_effect_payload(
                raw_observation,
                evaluated=evaluated,
            ),
            "get_evidence_seen": bool(get_evidence_seen),
        }
        episode.samples.append(_Sample(
            frame_index=int(frame_index),
            timestamp=float(timestamp),
            pixels=roi_pixels.copy(),
            metadata=metadata,
        ))
        episode.next_sample_at = (
            float(timestamp) + self.config.sample_interval_seconds
        )
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
        if reason != "result_pending_maximum_timeout":
            return False
        self._timeout_count += 1
        if (
            self._executor is None
            or self._dump_attempt_count >= self.config.max_episodes
        ):
            self._cap_drop_count += 1
            return False
        self._dump_attempt_count += 1
        snapshot = tuple(episode.samples)
        future = self._executor.submit(
            self._write_episode,
            episode,
            snapshot,
            next_state,
            float(timestamp),
            str(reason),
        )
        self._futures.append(future)
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
        panel_seen_count = sum(
            item["raw_observation"].get("panel_found") is True
            for item in sample_payloads
        )
        sequence_candidate_seen_count = sum(
            bool(item["raw_observation"].get("sequence_candidate"))
            for item in sample_payloads
        )
        complete_frame_count = sum(
            item["raw_observation"].get("frame_complete") is True
            for item in sample_payloads
        )
        sequence_ready_seen = any(
            item["qualification"].get("sequence_ready") is True
            for item in sample_payloads
        )
        duration = max(0.0, timeout_at - episode.started_at)
        manifest = {
            "anomaly_type": "result_pending_maximum_timeout",
            "episode_id": episode.episode_id,
            "cycle_id": episode.cycle_id,
            "hook_action_identity": episode.hook_action_identity,
            "result_pending_started_at": episode.started_at,
            "timeout_at": timeout_at,
            "duration": duration,
            "sample_rate": (
                len(samples) / duration if duration > 0.0 else 0.0
            ),
            "configured_sample_rate": (
                1.0 / self.config.sample_interval_seconds
            ),
            "sample_count": len(samples),
            "final_transition": {
                "from": RuntimeState.RESULT_PENDING.value,
                "to": next_state.value,
            },
            "terminal_reason": reason,
            "episode_summary": {
                "any_press_detector_execution": any(
                    item["press_detector_executed"]
                    for item in sample_payloads
                ),
                "any_panel_seen": panel_seen_count > 0,
                "any_sequence_candidate_seen": (
                    sequence_candidate_seen_count > 0
                ),
                "any_frame_complete_seen": complete_frame_count > 0,
                "any_sequence_ready_seen": sequence_ready_seen,
                "any_GET_evidence_seen": any(
                    item["get_evidence_seen"] for item in sample_payloads
                ),
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
            "panel_seen_count": panel_seen_count,
            "sequence_candidate_seen_count": (
                sequence_candidate_seen_count
            ),
            "complete_frame_count": complete_frame_count,
            "sequence_ready_seen": sequence_ready_seen,
        }

    def close(self) -> dict[str, Any]:
        self._active = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self.poll_completed(force=True)
        return {
            "result_pending_timeout_count": self._timeout_count,
            "result_pending_timeout_evidence_saved_count": self._saved_count,
            "result_pending_timeout_evidence_failure_count": len(
                self._failures
            ),
            "result_pending_timeout_evidence_failures": list(
                self._failures
            ),
            "result_pending_timeout_evidence_cap_drop_count": (
                self._cap_drop_count
            ),
            "result_pending_timeout_evidence_path": (
                str(self.root) if self._saved_count else None
            ),
            "result_pending_timeout_sample_interval_seconds": (
                self.config.sample_interval_seconds
            ),
            "result_pending_timeout_max_samples": self.config.max_samples,
            "result_pending_timeout_max_episodes": self.config.max_episodes,
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
