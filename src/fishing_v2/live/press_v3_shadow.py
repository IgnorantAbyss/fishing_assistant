"""Non-blocking PRESS V3 shadow comparison and bounded ROI evidence."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np

from src.config_loader import ROIConfig, load_roi_config
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import PressObservation
from src.fishing_v2.legacy_adapters.press_background_subtraction_v3_adapter import (
    create_v3_detector,
    serialize_v3_result,
    v3_result_to_observation,
)
from src.fishing_v2.runtime.press_sequence_aggregator import (
    PressSequenceTemporalAggregator,
)


@dataclass(frozen=True)
class PressV3ShadowConfig:
    debug_evidence: bool = False
    debug_max_episodes: int = 20
    debug_max_frames_per_episode: int = 8

    def __post_init__(self) -> None:
        if self.debug_max_episodes < 0 or self.debug_max_frames_per_episode < 0:
            raise ValueError("PRESS V3 debug limits must be non-negative")


class _PressV3DebugWriter:
    def __init__(self, root: Path, config: PressV3ShadowConfig) -> None:
        self.root = root
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="press-v3-debug")
        self._counts: dict[int, int] = {}
        self._futures: list[Future[Any]] = []

    @staticmethod
    def _write(
        root: Path,
        episode: int,
        frame_index: int,
        panel: np.ndarray,
        result: dict[str, Any],
        legacy: PressObservation | None,
        reason: str,
    ) -> None:
        target = root / f"episode_{episode:03d}" / f"frame_{frame_index:06d}"
        target.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target / "press_panel.png"), panel)
        strip = result.get("key_strip_crop")
        if isinstance(strip, np.ndarray):
            cv2.imwrite(str(target / "key_strip.png"), strip)
        masks = [
            item.get("binary_mask") for item in result.get("slots", ())
            if isinstance(item.get("binary_mask"), np.ndarray)
        ]
        if masks:
            combined = np.hstack(masks)
            cv2.imwrite(str(target / "foreground_slots.png"), combined)
            for index, mask in enumerate(masks):
                cv2.imwrite(str(target / f"slot_{index:02d}_mask.png"), mask)
        overlay = panel.copy()
        locator = result.get("locator", {})
        for index, bbox in enumerate(locator.get("slot_bboxes", ())):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 1)
            cv2.putText(overlay, str(index), (x1 + 2, y1 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.imwrite(str(target / "components.png"), overlay)
        payload = {
            "reason": reason,
            "v3": serialize_v3_result(result),
            "legacy": {
                "sequence": list(legacy.sequence_candidate),
                "sequence_ready": legacy.sequence_ready,
                "occupied_count": legacy.evidence.get("occupied_slot_count"),
            } if legacy is not None else None,
        }
        (target / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def submit(
        self,
        *,
        episode: int,
        frame_index: int,
        panel: np.ndarray,
        result: dict[str, Any],
        legacy: PressObservation | None,
        reason: str,
    ) -> None:
        if not self.config.debug_evidence or episode > self.config.debug_max_episodes:
            return
        count = self._counts.get(episode, 0)
        if count >= self.config.debug_max_frames_per_episode:
            return
        self._counts[episode] = count + 1
        self._futures.append(self._executor.submit(
            self._write, self.root, episode, frame_index, panel.copy(), result, legacy, reason
        ))

    def close(self) -> dict[str, Any]:
        self._executor.shutdown(wait=True, cancel_futures=False)
        total = sum(self._counts.values())
        return {
            "press_v3_debug_evidence_path": str(self.root) if total else None,
            "press_v3_debug_evidence_frames": total,
        }


class PressV3ShadowRunner:
    """Latest-only worker; its output never enters RuntimeController or Safety."""

    def __init__(
        self,
        *,
        output_root: Path,
        config: PressV3ShadowConfig | None = None,
        detector: Any | None = None,
        roi_config: ROIConfig | None = None,
    ) -> None:
        self.config = config or PressV3ShadowConfig()
        self.detector = detector or create_v3_detector()
        self.roi_config = roi_config or load_roi_config()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="press-v3-shadow")
        self._future: Future[dict[str, Any]] | None = None
        self._pending: tuple[FrameContext, PressObservation | None, np.ndarray] | None = None
        self._aggregator = PressSequenceTemporalAggregator()
        self._debug = _PressV3DebugWriter(
            output_root / "diagnostic_evidence" / "press_v3_debug", self.config
        )
        self._episode = 0
        self._active = False
        self._episode_started_at: float | None = None
        self._last_candidate: tuple[str, ...] = ()
        self._last_disagreement: str | None = None
        self._complete_logged = False
        self._latencies: deque[float] = deque(maxlen=2048)
        self._processed = 0
        self._dropped = 0
        self._episode_summaries: deque[dict[str, Any]] = deque(maxlen=200)
        self._latest_legacy: PressObservation | None = None
        self._latest_v3: PressObservation | None = None
        self._latest_aggregation: Any = None
        self._visual_ack_outcome: str | None = None
        self._debug_saved_reasons: set[str] = set()
        self._episode_legacy_sequence: tuple[str, ...] = ()
        self._episode_legacy_complete = False
        self._episode_legacy_occupied_count: int | None = None
        self._episode_v3_sequence: tuple[str, ...] = ()
        self._episode_v3_complete = False
        self._episode_v3_occupied_count: int | None = None
        self._episode_locator_confidence: float | None = None
        self._episode_background_stability: float | None = None

    def _process_completed(self) -> list[tuple[str, dict[str, Any]]]:
        if self._future is None or not self._future.done() or self._pending is None:
            return []
        context, legacy, panel = self._pending
        try:
            result = self._future.result()
        except Exception as exc:
            self._future = None
            self._pending = None
            return [("press_v3_disagreement", {
                "frame_index": context.frame_index,
                "reason": "v3_worker_exception",
                "exception": f"{type(exc).__name__}: {exc}",
            })]
        self._future = None
        self._pending = None
        observation = v3_result_to_observation(result, context)
        aggregation = self._aggregator.update(observation)
        self._processed += 1
        self._latencies.append(float(result.get("processing_latency_ms", 0.0)))
        self._latest_legacy = legacy
        self._latest_v3 = observation
        self._latest_aggregation = aggregation
        events: list[tuple[str, dict[str, Any]]] = []
        if observation.panel_present and not self._active:
            self._episode += 1
            self._active = True
            self._episode_started_at = context.timestamp
            self._last_candidate = ()
            self._last_disagreement = None
            self._complete_logged = False
            self._visual_ack_outcome = None
            self._debug_saved_reasons.clear()
            self._episode_legacy_sequence = ()
            self._episode_legacy_complete = False
            self._episode_legacy_occupied_count = None
            self._episode_v3_sequence = ()
            self._episode_v3_complete = False
            self._episode_v3_occupied_count = None
            self._episode_locator_confidence = float(
                result.get("locator", {}).get("locator_confidence", 0.0)
            )
            self._episode_background_stability = float(
                result.get("background_model", {}).get(
                    "background_stability", 0.0
                )
            )
            events.extend((
                ("press_v3_strip_located", {
                    "episode_index": self._episode,
                    "frame_index": context.frame_index,
                    **dict(result.get("locator", {})),
                }),
                ("press_v3_background_model", {
                    "episode_index": self._episode,
                    "frame_index": context.frame_index,
                    **dict(result.get("background_model", {})),
                }),
            ))
        candidate = tuple(observation.sequence_candidate)
        if self._active and candidate and candidate != self._last_candidate:
            self._last_candidate = candidate
            events.append(("press_v3_candidate", {
                "episode_index": self._episode,
                "frame_index": context.frame_index,
                "sequence": list(candidate),
                "occupied_count": observation.evidence.get("occupied_slot_count"),
                "complete": bool(result.get("frame_complete", False)),
            }))
        v3_complete = bool(aggregation.sequence_ready)
        v3_sequence = tuple(aggregation.sequence_candidate)
        if self._active and v3_complete and not self._complete_logged:
            self._complete_logged = True
            events.append(("press_v3_complete", {
                "episode_index": self._episode,
                "frame_index": context.frame_index,
                "sequence": list(v3_sequence),
            }))
        legacy_sequence = tuple(legacy.sequence_candidate) if legacy is not None else ()
        legacy_complete = bool(
            legacy is not None
            and legacy.evidence.get("clean_frame_eligible") is True
            and legacy_sequence
        )
        if legacy_sequence and (
            legacy_complete or len(legacy_sequence) > len(self._episode_legacy_sequence)
        ):
            self._episode_legacy_sequence = legacy_sequence
            self._episode_legacy_complete = legacy_complete
            self._episode_legacy_occupied_count = int(
                legacy.evidence.get("occupied_slot_count", len(legacy_sequence))
            )
        if candidate and (
            bool(result.get("frame_complete"))
            or len(candidate) > len(self._episode_v3_sequence)
        ):
            self._episode_v3_sequence = candidate
            self._episode_v3_occupied_count = int(
                observation.evidence.get("occupied_slot_count", len(candidate))
            )
        if v3_complete:
            self._episode_v3_sequence = v3_sequence
            self._episode_v3_complete = True
        disagreement = None
        if candidate and legacy_sequence and candidate != legacy_sequence:
            disagreement = "sequence_mismatch"
        elif bool(result.get("frame_complete")) != legacy_complete:
            disagreement = "completeness_mismatch"
        if disagreement is not None and disagreement != self._last_disagreement:
            self._last_disagreement = disagreement
            events.append(("press_v3_disagreement", {
                "episode_index": self._episode,
                "frame_index": context.frame_index,
                "reason": disagreement,
                "legacy_sequence": list(legacy_sequence),
                "v3_sequence": list(candidate),
            }))
        debug_reason = (
            disagreement
            or (
                "best_complete_candidate"
                if result.get("frame_complete") else
                f"incomplete_abstain:{result.get('rejection_reason')}"
            )
        )
        if self._active and debug_reason not in self._debug_saved_reasons:
            self._debug_saved_reasons.add(debug_reason)
            self._debug.submit(
                episode=self._episode,
                frame_index=context.frame_index,
                panel=panel,
                result=result,
                legacy=legacy,
                reason=debug_reason,
            )
        if self._active and aggregation.panel_disappeared:
            events.extend(self._close_episode(context.timestamp))
        return events

    def _close_episode(self, timestamp: float) -> list[tuple[str, dict[str, Any]]]:
        if not self._active:
            return []
        payload = {
            "episode_index": self._episode,
            "legacy_sequence": list(self._episode_legacy_sequence),
            "legacy_complete": self._episode_legacy_complete,
            "v3_sequence": list(self._episode_v3_sequence),
            "v3_complete": self._episode_v3_complete,
            "legacy_occupied_count": self._episode_legacy_occupied_count,
            "v3_occupied_count": self._episode_v3_occupied_count,
            "legacy_decoded_count": len(self._episode_legacy_sequence),
            "v3_decoded_count": len(self._episode_v3_sequence),
            "visual_ack_outcome": self._visual_ack_outcome,
            "panel_duration": round(timestamp - (self._episode_started_at or timestamp), 6),
            "strip_locator_confidence": self._episode_locator_confidence,
            "background_stability": self._episode_background_stability,
            "disagreement_reason": self._last_disagreement,
        }
        self._episode_summaries.append(payload)
        self._active = False
        self._aggregator.reset()
        return [("press_v3_episode_summary", payload)]

    def record_visual_ack(self, outcome: str) -> None:
        if self._active:
            self._visual_ack_outcome = str(outcome)

    def observe(
        self,
        frame: np.ndarray,
        context: FrameContext,
        legacy: PressObservation | None,
    ) -> list[tuple[str, dict[str, Any]]]:
        events = self._process_completed()
        if self._future is not None:
            self._dropped += 1
            return events
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.roi_config.pixel_roi("press_sequence", width, height)
        panel = frame[y1:y2, x1:x2].copy()
        self._pending = (context, legacy, panel)
        self._future = self._executor.submit(self.detector.detect, panel)
        return events

    def finish(self, timestamp: float) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
        if self._future is not None:
            try:
                self._future.result()
            except Exception:
                pass
        events = self._process_completed()
        events.extend(self._close_episode(timestamp))
        self._executor.shutdown(wait=True, cancel_futures=False)
        debug_summary = self._debug.close()
        values = sorted(self._latencies)
        p95 = values[min(len(values) - 1, max(0, int(np.ceil(len(values) * 0.95)) - 1))] if values else 0.0
        return events, {
            "press_v3_shadow_processed_frames": self._processed,
            "press_v3_shadow_dropped_busy_frames": self._dropped,
            "press_v3_processing_latency_mean_ms": round(mean(values), 4) if values else 0.0,
            "press_v3_processing_latency_p95_ms": round(float(p95), 4),
            "press_v3_episode_count": len(self._episode_summaries),
            "press_v3_episode_summaries": list(self._episode_summaries),
            **debug_summary,
        }
