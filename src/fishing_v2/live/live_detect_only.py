"""Real-frame observation loop with a hard no-action boundary."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import cv2
import numpy as np
import yaml

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import (
    GetObservation,
    HookObservation,
    PressObservation,
    PromptObservation,
    PromptObservationKind,
    ResultBannerObservation,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import ObservationFusion
from src.fishing_v2.legacy_adapters.get_detector_adapter import LegacyGetDetectorAdapter
from src.fishing_v2.legacy_adapters.hook_detector_adapter import LegacyHookDetectorAdapter
from src.fishing_v2.legacy_adapters.press_detector_adapter import LegacyPressDetectorAdapter
from src.fishing_v2.live.session_logger import LiveSessionLogger
from src.fishing_v2.live.diagnostic_evidence import (
    EVIDENCE_MODES,
    DiagnosticEvidenceConfig,
    DiagnosticEvidenceRecorder,
)
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.perception.prompt_bundle import LoadedPromptBundle
from src.fishing_v2.perception.result_banner_observer import (
    ResultBannerConfig,
    ResultBannerObserver,
)
from src.fishing_v2.replay.v2_replay_runner import _config_objects
from src.fishing_v2.runtime.detector_activation import (
    DetectorActivationMode,
    DetectorActivationPolicy,
    DetectorActivationSnapshot,
)
from src.fishing_v2.runtime.detector_evidence import DetectorEvidenceQualifier
from src.fishing_v2.runtime.fishing_fsm import FishingFSM
from src.fishing_v2.runtime.runtime_controller import ActionExecutionMode, RuntimeController
from src.fishing_v2.runtime.safety_policy import SafetyPolicy
from src.fishing_v2.runtime.scheduling import PromptPollingConfig, RuntimeSchedulePolicy
from src.fishing_v2.runtime.synchronization import StartupSynchronizer
from src.screen_capture import validate_bgr_frame
from src.config_loader import load_roi_config


EXPECTED_RESOLUTION = (2560, 1440)
WOULD_FIRE_NAMES = {
    ActionIntent.CAST: "WOULD_CAST",
    ActionIntent.START_HOOK: "WOULD_START_HOOK",
    ActionIntent.HOOK_ACTION: "WOULD_HOOK_ACTION",
    ActionIntent.PRESS_SEQUENCE: "WOULD_PRESS_SEQUENCE",
    ActionIntent.COLLECT: "WOULD_COLLECT",
}


class LivePreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveDetectOnlyConfig:
    duration_seconds: float = 180.0
    max_fps: float = 25.0
    show_overlay: bool = True
    save_transition_frames: bool = True
    unknown_screenshot_seconds: float = 2.0
    evidence_mode: str = "minimal"
    evidence_video_fps: float = 10.0
    max_completed_cycles: int | None = None

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.max_fps <= 0:
            raise ValueError("max_fps must be positive")
        if self.unknown_screenshot_seconds <= 0:
            raise ValueError("unknown_screenshot_seconds must be positive")
        if self.evidence_mode not in EVIDENCE_MODES:
            raise ValueError(f"evidence_mode must be one of {EVIDENCE_MODES}")
        if self.evidence_video_fps <= 0:
            raise ValueError("evidence_video_fps must be positive")
        if self.max_completed_cycles is not None and self.max_completed_cycles <= 0:
            raise ValueError("max_completed_cycles must be positive when provided")


class WouldFireDeduplicator:
    """Summarize repeated proposals as one opportunity per fishing cycle."""

    def __init__(self) -> None:
        self.cycle_id = 1
        self.raw_proposals: Counter[str] = Counter()
        self.unique_events: Counter[str] = Counter()
        self._seen: set[tuple[int, ActionIntent]] = set()

    @property
    def has_cycle_activity(self) -> bool:
        return any(cycle == self.cycle_id for cycle, _ in self._seen)

    def finish_cycle(self) -> None:
        if self.has_cycle_activity:
            self.cycle_id += 1

    def reset_for_sync_recovery(self) -> None:
        """Prevent pre-loss proposals from suppressing the recovered cycle."""
        if self.has_cycle_activity:
            self.cycle_id += 1
        self._seen.clear()

    def observe(
        self,
        request: ActionRequest,
        *,
        safety_reason: str,
        frame_index: int,
        timestamp: float,
        runtime_state: str,
        prompt_evidence: Mapping[str, Any] | None,
        specialized_evidence: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if request.intent == ActionIntent.NONE:
            return None
        self.raw_proposals[request.intent.value] += 1
        # This reason means every production safety check passed and only the
        # immutable detect-only emission switch prevented execution.
        if safety_reason != "action_emission_disabled":
            return None
        key = (self.cycle_id, request.intent)
        if key in self._seen:
            return None
        self._seen.add(key)
        event_type = WOULD_FIRE_NAMES[request.intent]
        self.unique_events[event_type] += 1
        return {
            "event_type": event_type,
            "timestamp": timestamp,
            "frame_index": frame_index,
            "cycle_id": self.cycle_id,
            "runtime_state": runtime_state,
            "prompt_evidence": dict(prompt_evidence or {}),
            "specialized_evidence": dict(specialized_evidence),
            "confidence": request.confidence,
            "action_payload": dict(request.payload),
            "safety_decision": "WAIT",
            "safety_reason": safety_reason,
            "action_applied": False,
            "deduplication_key": f"cycle:{self.cycle_id}:{request.intent.value}",
        }


class DiagnosticOverlay:
    WINDOW_NAME = "Fishing Assistant - DETECT ONLY"

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._created = False

    def show(self, frame: np.ndarray, lines: list[str]) -> None:
        if not self.enabled:
            return
        display = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        panel_height = min(26 * (len(lines) + 1), display.shape[0])
        cv2.rectangle(display, (0, 0), (display.shape[1], panel_height), (12, 12, 12), -1)
        for index, line in enumerate(lines, start=1):
            cv2.putText(
                display, line, (12, index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.56, (90, 255, 120), 1, cv2.LINE_AA,
            )
        cv2.imshow(self.WINDOW_NAME, display)
        cv2.waitKey(1)
        if not self._created:
            self._created = True
            self._make_non_activating()

    def _make_non_activating(self) -> None:
        # Best-effort Windows diagnostic-window style. It never activates or
        # focuses the game window and never emits input.
        import os
        if os.name != "nt":
            return
        try:
            import ctypes
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            hwnd = int(user32.FindWindowW(None, self.WINDOW_NAME))
            if not hwnd:
                return
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_NOACTIVATE = 0x0010
            SWP_SHOWWINDOW = 0x0040
            current = int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, current | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
            user32.SetWindowPos(
                hwnd, -1, 0, 0, 0, 0,
                SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
            )
        except Exception:
            return

    def close(self) -> None:
        if self.enabled and self._created:
            cv2.destroyWindow(self.WINDOW_NAME)


def validate_emit_actions(emit_actions: bool) -> None:
    if emit_actions:
        raise LivePreflightError("--emit-actions=true is forbidden: live runtime is detect-only")


def _prompt_polling(config: Mapping[str, Any]) -> RuntimeSchedulePolicy:
    return RuntimeSchedulePolicy(PromptPollingConfig(
        waiting_interval_seconds=float(config["prompt_polling"]["waiting_interval_seconds"]),
        waiting_min_seconds=float(config["prompt_polling"]["waiting_min_seconds"]),
        waiting_max_seconds=float(config["prompt_polling"]["waiting_max_seconds"]),
        ready_fps=float(config["prompt_polling"]["ready_fps"]),
        result_pending_fps=float(config["prompt_polling"]["result_pending_fps"]),
        ready_confirmation_timeout_seconds=float(
            config["prompt_polling"]["ready_confirmation_timeout_seconds"]
        ),
    ))


class LiveDetectOnlyRuntime:
    def __init__(
        self,
        *,
        config_path: str | Path,
        prompt_bundle: LoadedPromptBundle,
        capture: Any,
        logger: LiveSessionLogger,
        live_config: LiveDetectOnlyConfig,
        emit_actions: bool = False,
        hook_detector: Any | None = None,
        press_detector: Any | None = None,
        get_detector: Any | None = None,
        result_banner_observer: Any | None = None,
        evidence_recorder: DiagnosticEvidenceRecorder | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        validate_emit_actions(emit_actions)
        self.config_path = Path(config_path)
        self.prompt_bundle = prompt_bundle
        self.capture = capture
        self.logger = logger
        self.live_config = live_config
        self.hook_detector = hook_detector or LegacyHookDetectorAdapter()
        self.press_detector = press_detector or LegacyPressDetectorAdapter()
        self.get_detector = get_detector or LegacyGetDetectorAdapter()
        self.clock = clock
        self.sleep = sleep
        self.overlay = DiagnosticOverlay(live_config.show_overlay)
        self.deduplicator = WouldFireDeduplicator()
        self._raw_config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        banner_config = ResultBannerConfig.from_mapping(
            self._raw_config["result"]["banner_observer"]
        )
        self.result_banner_observer = (
            result_banner_observer or ResultBannerObserver(banner_config)
        )
        (
            fusion_config, fsm_config, sync_config, safety_config,
            activation_config, qualification_config,
        ) = _config_objects(self.config_path)
        if safety_config.emit_actions or self._raw_config["safety"]["emit_actions"] is not False:
            raise LivePreflightError("Live detect-only requires safety.emit_actions=false")
        self.fsm = FishingFSM(fsm_config, initial_state=RuntimeState.SYNCING)
        self.activation_policy = DetectorActivationPolicy(activation_config)
        self.controller = RuntimeController(
            ObservationFusion(fusion_config),
            self.fsm,
            SafetyPolicy(safety_config),
            action_sink=None,
            activation_policy=self.activation_policy,
            evidence_qualifier=DetectorEvidenceQualifier(qualification_config),
        )
        self.synchronizer = StartupSynchronizer(sync_config, started_at=0.0)
        self.recovery_synchronizer = StartupSynchronizer(sync_config, started_at=0.0)
        self.schedule = _prompt_polling(self._raw_config)
        self._opened = False
        self._capture_diagnostics: dict[str, Any] = {}
        self.evidence_recorder = evidence_recorder
        if self.live_config.evidence_mode == "diagnostic":
            self.evidence_recorder = self.evidence_recorder or DiagnosticEvidenceRecorder(
                self.logger.path,
                config=DiagnosticEvidenceConfig(video_fps=self.live_config.evidence_video_fps),
            )
            self.logger.set_event_listener(self.evidence_recorder.mark_event)
        elif self.evidence_recorder is not None:
            raise ValueError("evidence_recorder requires evidence_mode='diagnostic'")
        self._diagnostic_roi_bounds: dict[str, tuple[int, int, int, int]] = {}

    def preflight(self) -> np.ndarray:
        try:
            self.capture.open()
            self._opened = True
            frame = validate_bgr_frame(self.capture.capture())
            diagnostics = getattr(self.capture, "diagnostics", None)
            self._capture_diagnostics = dict(diagnostics()) if callable(diagnostics) else {}
        except Exception as exc:
            raise LivePreflightError(f"Capture preflight failed: {exc}") from exc
        height, width = frame.shape[:2]
        if (width, height) != EXPECTED_RESOLUTION:
            raise LivePreflightError(
                f"Unsupported capture resolution {width}x{height}; expected 2560x1440"
            )
        self.prompt_bundle.roi.pixel_bounds(width, height)
        if self.evidence_recorder is not None:
            roi_config = load_roi_config()
            self._diagnostic_roi_bounds = {
                "prompt": self.prompt_bundle.roi.pixel_bounds(width, height),
                "hook": roi_config.pixel_roi("hook_bar", width, height),
                "press": roi_config.pixel_roi("press_sequence", width, height),
                "get": roi_config.pixel_roi(
                    "get_search" if "get_search" in roi_config.rois else "get_window",
                    width,
                    height,
                ),
            }
            prepare_video = getattr(self.evidence_recorder, "prepare_video", None)
            if callable(prepare_video):
                try:
                    prepare_video(frame)
                except Exception as exc:
                    raise LivePreflightError(
                        f"Diagnostic video preflight failed: {type(exc).__name__}: {exc}"
                    ) from exc
        expected_count = int(self.prompt_bundle.bundle.get("prototype_count", 0))
        if len(self.prompt_bundle.model.prototypes) != expected_count or expected_count not in {36, 37}:
            raise LivePreflightError("Final Prompt bundle must contain 35 medoids plus reviewed Live candidates")
        return frame

    @staticmethod
    def _activation_payload(snapshot: DetectorActivationSnapshot) -> dict[str, Any]:
        return {
            "hook": snapshot.hook.value,
            "press": snapshot.press.value,
            "get": snapshot.get.value,
            "hook_target_fps": snapshot.hook_fps,
            "press_target_fps": snapshot.press_fps,
            "get_target_fps": snapshot.get_fps,
        }

    @staticmethod
    def _specialized_payload(result: Any) -> dict[str, Any]:
        hook = result.qualified.bundle.hook
        press = result.qualified.bundle.press
        get = result.qualified.bundle.get
        result_banner = result.qualified.bundle.result_banner
        return {
            "hook": asdict(hook) if hook else None,
            "press": asdict(press) if press else None,
            "get": asdict(get) if get else None,
            "result_banner": asdict(result_banner) if result_banner else None,
            "qualified": {
                "hook": result.qualified.hook.qualified_detected,
                "press": result.qualified.press.qualified_detected,
                "get": result.qualified.get.qualified_detected,
            },
        }

    @staticmethod
    def _observation_summary(observation: Any | None) -> dict[str, Any] | None:
        if observation is None:
            return None
        base = {
            "detected": bool(getattr(observation, "detected", False)),
            "confidence": float(getattr(observation, "confidence", 0.0)),
            "frame_index": int(observation.frame_index),
            "timestamp": float(observation.timestamp),
            "source": str(getattr(observation, "source", "")),
        }
        evidence = getattr(observation, "evidence", {})
        selected_evidence_keys = (
            "matched_features", "exception", "bar_bbox", "fill_endpoint_x",
            "divider_line_x", "divider_line_detected", "divider_margin_passed",
            "panel_bbox", "panel_phase", "selected_clean_frame", "panel_disappeared",
        )
        base["evidence"] = {
            key: evidence[key] for key in selected_evidence_keys if key in evidence
        }
        if isinstance(observation, HookObservation):
            base.update({
                "fill_ratio": observation.fill_ratio,
                "divider_ratio": observation.divider_ratio,
            })
        elif isinstance(observation, PressObservation):
            base.update({
                "panel_candidate": observation.panel_candidate,
                "panel_present": observation.panel_present,
                "sequence_candidate": list(observation.sequence_candidate),
                "sequence_ready": observation.sequence_ready,
                "sequence_confidence": observation.sequence_confidence,
            })
        elif isinstance(observation, GetObservation):
            # GET localization failures cannot be reconstructed from the
            # flattened feature list. Diagnostic mode intentionally keeps the
            # complete JSON-only adapter payload for later Live audits.
            if "legacy_debug" in evidence:
                base["evidence"]["legacy_debug"] = evidence["legacy_debug"]
        return base

    @staticmethod
    def _get_diagnostic_fields(
        raw: GetObservation | None,
        qualified: GetObservation | None,
        qualification: Any,
    ) -> dict[str, Any]:
        evidence = raw.evidence if raw is not None else {}
        debug = evidence.get("legacy_debug", {})
        sliding = debug.get("vertical_sliding") or {}
        sliding_selected = sliding.get("selected") or {}
        fallback = debug.get("fixed_fallback") or {}
        structure = debug.get("structure_debug") or {}
        if not structure:
            structure = sliding_selected.get("structure_debug") or fallback.get("structure_debug") or {}
        dark_ratio = sliding_selected.get("dark_ratio")
        if dark_ratio is None:
            dark_ratio = fallback.get("dark_ratio")
        qualified_evidence = qualified.evidence if qualified is not None else {}
        localization_source = debug.get("localization_source")
        return {
            "raw_candidate": bool(raw and raw.detected),
            "qualified": bool(qualification.qualified_detected),
            "confidence": float(raw.confidence) if raw is not None else None,
            "search_roi": debug.get("search_roi"),
            "candidate_bbox": debug.get("panel_bbox"),
            "candidate_bbox_global": debug.get("panel_bbox_global"),
            "vertical_anchor_px": sliding_selected.get("vertical_anchor_px"),
            "vertical_offset_ratio": sliding_selected.get("vertical_offset_ratio"),
            "panel_confidence": debug.get("panel_confidence"),
            "fallback_used": localization_source in {
                "vertical_sliding_strong_grid", "fixed_geometry_strong_grid_fallback"
            },
            "fallback_reason": (
                None if debug.get("panel_bbox") is not None
                else sliding.get("rejection_reason") or debug.get("rejection_reason")
            ),
            "localization_source": localization_source,
            "dark_ratio": dark_ratio,
            "grid_contour_count": structure.get("grid_cell_candidates"),
            "title_bright_ratio": structure.get("title_bright_ratio"),
            "button_bright_ratio": structure.get("button_bright_ratio"),
            "temporal_confirmation_count": qualified_evidence.get("get_confirmation_frames", 0),
            "activation_mode": qualification.activation_mode.value,
            "evidence_eligible_for_fusion": bool(qualification.used_by_fusion),
            "rejection_reason": qualification.qualification_reason,
        }

    @classmethod
    def _diagnostic_metadata(
        cls,
        *,
        prompt: PromptObservation | None,
        raw_bundle: ObservationBundle,
        result: Any,
        executed: Mapping[str, bool],
    ) -> dict[str, Any]:
        prompt_evidence = prompt.evidence if prompt is not None else {}
        payload: dict[str, Any] = {
            "prompt": {
                "executed": bool(executed["prompt"]),
                "raw": {
                    "predicted_label": prompt_evidence.get("raw_predicted_label"),
                    "similarity": prompt_evidence.get("similarity"),
                    "second_label": prompt_evidence.get("second_label"),
                    "ambiguity_margin": prompt_evidence.get("ambiguity_margin"),
                } if prompt is not None else None,
                "qualified": bool(prompt and prompt.kind != PromptObservationKind.UNKNOWN),
                "result": {
                    "label": prompt.kind.value,
                    "confidence": prompt.confidence,
                } if prompt is not None else None,
                "rejection_reason": prompt_evidence.get("rejection_reason"),
            }
        }
        for name in ("hook", "press", "get"):
            raw = getattr(raw_bundle, name)
            qualification = getattr(result.qualified, name)
            qualified_observation = getattr(result.qualified.bundle, name)
            payload[name] = {
                "executed": bool(executed[name]),
                "raw": cls._observation_summary(raw) if executed[name] else None,
                "qualified": asdict(qualification),
                "result": cls._observation_summary(qualified_observation),
                "rejection_reason": qualification.qualification_reason,
            }
            if name == "get":
                payload[name]["diagnostics"] = cls._get_diagnostic_fields(
                    raw, qualified_observation, qualification
                )
        banner = raw_bundle.result_banner
        payload["result_banner"] = {
            "executed": banner is not None,
            "detected": bool(banner and banner.detected),
            "confidence": float(banner.confidence) if banner else None,
            "evidence": dict(banner.evidence) if banner else None,
            "collect_eligible": False,
        }
        return payload

    def _prompt_interval(self) -> float:
        configured = self.schedule.prompt_interval_seconds(self.fsm.state)
        return configured if configured is not None else 0.2

    def _detector_interval(self, activation: DetectorActivationSnapshot) -> float | None:
        targets = [
            activation.hook_fps if activation.hook != DetectorActivationMode.OFF else 0.0,
            activation.press_fps if activation.press != DetectorActivationMode.OFF else 0.0,
            activation.get_fps if activation.get != DetectorActivationMode.OFF else 0.0,
        ]
        target = min(self.live_config.max_fps, max(targets))
        return 1.0 / target if target > 0 else None

    def _overlay_lines(
        self,
        *,
        capture_fps: float,
        latency_ms: float,
        prompt: PromptObservation | None,
        result: Any | None,
        activation: DetectorActivationSnapshot,
    ) -> list[str]:
        evidence = prompt.evidence if prompt else {}
        hook = result.qualified.bundle.hook if result else None
        press = result.qualified.bundle.press if result else None
        get = result.qualified.bundle.get if result else None
        result_banner = result.qualified.bundle.result_banner if result else None
        request = result.fsm.action_request if result else ActionRequest(ActionIntent.NONE, 0.0, "none")
        return [
            f"DETECT ONLY | capture={capture_fps:.1f} FPS latency={latency_ms:.1f} ms bundle={self.prompt_bundle.bundle_version}",
            f"Prompt raw={evidence.get('raw_predicted_label', 'N/A')} predicted={prompt.kind.value if prompt else 'N/A'} sim={evidence.get('similarity', 0.0):.3f} second={evidence.get('second_label', 'N/A')} margin={evidence.get('ambiguity_margin', 0.0):.3f}",
            f"Runtime={self.fsm.state.value} activation H/P/G={activation.hook.value}/{activation.press.value}/{activation.get.value}",
            f"Hook raw/qualified={bool(hook)}/{bool(hook and hook.detected)} fill={getattr(hook, 'fill_ratio', None)} crossed={bool(hook and hook.evidence.get('divider_margin_passed'))}",
            f"Press panel={bool(press and press.panel_present)} sequence={''.join(press.sequence_candidate) if press else ''} frozen={bool(press and press.sequence_ready)}",
            f"Get panel={bool(get and get.detected)} | would-fire={request.intent.value} | action_applied=false",
            f"Result banner={bool(result_banner and result_banner.detected)} hold-only=true",
        ]

    def run(self, *, max_frames: int | None = None) -> dict[str, Any]:
        captured = processed = actions_applied = 0
        completed_cycles = 0
        evidence_episode_id = 1
        stop_after_completed_cycle = False
        evidence_failure_reason: str | None = None
        latencies: list[float] = []
        detector_runs: Counter[str] = Counter()
        result_name = "completed"
        last_prompt: PromptObservation | None = None
        last_prompt_kind: str | None = None
        last_result: Any | None = None
        last_activation: DetectorActivationSnapshot | None = None
        hook_crossed = False
        hook_crossed_at: float | None = None
        press_frozen_frame: int | None = None
        get_visible = False
        result_banner_visible = False
        unknown_started: float | None = None
        unknown_saved = False
        conflict_active = False
        sync_required_active = False
        last_terminal = 0.0
        started = self.clock()
        next_prompt_due = 0.0
        next_detector_due = 0.0
        initial_bundle = ObservationBundle(0, 0.0)
        activation = self.activation_policy.evaluate(
            RuntimeState.SYNCING, initial_bundle, recorded_observation=True
        )
        try:
            self.preflight()
            self.logger.event("preflight_passed", {
                "timestamp": 0.0,
                "resolution": list(EXPECTED_RESOLUTION),
                "approved_roi": list(self.prompt_bundle.roi.pixel),
                "bundle_sha256": self.prompt_bundle.bundle_sha256,
                "emit_actions": False,
                "action_sink": None,
                "capture": self._capture_diagnostics,
            })
            if self._capture_diagnostics.get("fallback_used"):
                self.logger.event("capture_backend_fallback", {
                    "timestamp": 0.0,
                    "reason": self._capture_diagnostics.get("fallback_reason"),
                    "requested_backend": self._capture_diagnostics.get("requested_backend"),
                    "active_backend": self._capture_diagnostics.get("backend"),
                })
            if (
                self.live_config.show_overlay
                and self._capture_diagnostics.get("overlay_capture_warning")
            ):
                self.logger.event("capture_backend_warning", {
                    "timestamp": 0.0,
                    "reason": "mss-region may capture the diagnostic overlay or other covering windows",
                    "backend": self._capture_diagnostics.get("backend"),
                })
            while self.clock() - started < self.live_config.duration_seconds:
                if max_frames is not None and captured >= max_frames:
                    break
                frame_loop_started = self.clock()
                elapsed = frame_loop_started - started
                try:
                    frame = validate_bgr_frame(self.capture.capture())
                except KeyboardInterrupt:
                    result_name = "interrupted_by_user"
                    break
                except Exception as exc:
                    result_name = "safe_stop_capture_failure"
                    self.logger.event("capture_failure", {
                        "timestamp": elapsed,
                        "frame_index": captured + 1,
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
                    break
                captured += 1
                height, width = frame.shape[:2]
                if (width, height) != EXPECTED_RESOLUTION:
                    result_name = "safe_stop_resolution_changed"
                    self.logger.event("capture_failure", {
                        "timestamp": elapsed,
                        "frame_index": captured,
                        "reason": f"resolution_changed_to_{width}x{height}",
                    })
                    break
                if self.evidence_recorder is not None:
                    try:
                        self.evidence_recorder.record_frame(
                            frame, capture_frame_index=captured, timestamp=elapsed
                        )
                    except Exception as exc:
                        result_name = "safe_stop_evidence_failure"
                        evidence_failure_reason = f"video: {type(exc).__name__}: {exc}"
                        self.logger.event("diagnostic_evidence_failure", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "reason": evidence_failure_reason,
                        })
                        break
                prompt_due = elapsed >= next_prompt_due
                detector_interval = self._detector_interval(activation)
                detector_due = detector_interval is not None and elapsed >= next_detector_due
                should_process = prompt_due or detector_due
                if should_process:
                    processing_started = self.clock()
                    context = FrameContext(captured, elapsed, metadata={"source": "live_detect_only"})
                    ready_burst_update = None
                    if prompt_due:
                        last_prompt = self.prompt_bundle.observer.observe(frame, context)
                        ready_burst_update = self.schedule.observe_prompt(
                            self.fsm.state, last_prompt.kind, elapsed
                        )
                        if ready_burst_update.candidate_reset_required:
                            self.fsm.clear_transition_candidate()
                        next_prompt_due = elapsed + self._prompt_interval()
                    prompt = last_prompt
                    run_detectors = detector_due or any(
                        mode != DetectorActivationMode.OFF
                        for mode in (activation.hook, activation.press, activation.get)
                    )
                    hook = self.hook_detector.observe(frame, context) if run_detectors and activation.hook != DetectorActivationMode.OFF else None
                    press = self.press_detector.observe(frame, context) if run_detectors and activation.press != DetectorActivationMode.OFF else None
                    run_get = run_detectors and activation.get != DetectorActivationMode.OFF
                    if self.fsm.state == RuntimeState.IDLE:
                        run_get = True  # required read-only GET guard before WOULD_CAST
                    get = self.get_detector.observe(frame, context) if run_get else None
                    run_result_banner = bool(
                        run_detectors and self.fsm.state == RuntimeState.RESULT_PENDING
                    )
                    result_banner = (
                        self.result_banner_observer.observe(frame, context)
                        if run_result_banner else None
                    )
                    detector_runs.update({
                        "hook": int(hook is not None),
                        "press": int(press is not None),
                        "get": int(get is not None),
                        "result_banner": int(result_banner is not None),
                    })
                    raw_bundle = ObservationBundle(
                        captured, elapsed, prompt, hook, press, get, result_banner
                    )
                    sync_reason = None
                    processing_from_sync_required = self.fsm.state == RuntimeState.SYNC_REQUIRED
                    transition_results: list[Any] = []
                    if self.fsm.state == RuntimeState.SYNCING:
                        _, startup_qualified = self.controller.qualify_raw_bundle(
                            raw_bundle, action_mode=ActionExecutionMode.RECORDED_OBSERVATION
                        )
                        sync = self.synchronizer.observe(startup_qualified.bundle)
                        sync_reason = sync.reason
                        if sync.synchronized:
                            transition_results.append(
                                self.fsm.force_state(sync.state, elapsed, sync.reason)
                            )
                        elif sync.state == RuntimeState.SYNC_REQUIRED:
                            transition_results.append(self.fsm.force_state(
                                RuntimeState.SYNC_REQUIRED, elapsed, sync.reason
                            ))
                    last_result = self.controller.process(
                        raw_bundle,
                        foreground=self.capture.is_foreground(),
                        runtime_environment_supported=True,
                        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
                    )
                    if last_result.action_applied:
                        raise RuntimeError("Detect-only safety invariant violated: action_applied=true")
                    actions_applied += int(last_result.action_applied)
                    processed += 1
                    if last_result.fsm.changed:
                        transition_results.append(last_result.fsm)
                        if (
                            last_result.fsm.previous_state == RuntimeState.WAITING
                            and last_result.fsm.next_state == RuntimeState.READY
                        ):
                            confirmation = self.schedule.confirm_ready(elapsed)
                            next_prompt_due = min(
                                next_prompt_due,
                                elapsed + self._prompt_interval(),
                            )
                            self.logger.event("ready_confirmation_burst_confirmed", {
                                "timestamp": elapsed,
                                "frame_index": captured,
                                "support_frames": confirmation.ready_support_frames,
                                "duration_seconds": confirmation.candidate_age_seconds,
                                "action_intent": ActionIntent.NONE.value,
                                "action_applied": False,
                            })

                    if ready_burst_update is not None:
                        burst_payload = {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "support_frames": ready_burst_update.ready_support_frames,
                            "duration_seconds": ready_burst_update.candidate_age_seconds,
                            "reason": ready_burst_update.reason,
                            "target_fps": self.schedule.config.ready_fps,
                            "timeout_seconds": (
                                self.schedule.config.ready_confirmation_timeout_seconds
                            ),
                            "action_intent": ActionIntent.NONE.value,
                            "action_applied": False,
                        }
                        if ready_burst_update.burst_timed_out:
                            self.logger.event("ready_confirmation_burst_timeout", burst_payload)
                        if ready_burst_update.burst_cancelled:
                            self.logger.event("ready_confirmation_burst_cancelled", burst_payload)
                        if ready_burst_update.burst_started:
                            self.logger.event("ready_confirmation_burst_started", burst_payload)

                    entered_sync_required = bool(
                        self.fsm.state == RuntimeState.SYNC_REQUIRED
                        and not processing_from_sync_required
                    )
                    if entered_sync_required:
                        self.controller.reset_for_sync_recovery(elapsed)
                        self.recovery_synchronizer.reset(started_at=elapsed)
                        self.deduplicator.reset_for_sync_recovery()
                        self.logger.event("sync_recovery_started", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "recovery_target": None,
                            "support_frames": 0,
                            "duration_seconds": 0.0,
                            "rejection_reason": "fresh_window_after_sync_required",
                        })
                    elif processing_from_sync_required:
                        recovery = self.recovery_synchronizer.observe_recovery(
                            last_result.qualified,
                            has_conflict=last_result.evidence.has_conflict,
                        )
                        recovery_payload = {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "recovery_target": (
                                recovery.candidate_state.value
                                if recovery.candidate_state is not None else None
                            ),
                            "support_frames": recovery.support_frames,
                            "observed_frames": recovery.observed_frames,
                            "duration_seconds": recovery.duration_seconds,
                            "confidence": recovery.confidence,
                            "rejection_reason": recovery.rejection_reason,
                            "reason": recovery.reason,
                        }
                        self.logger.event("sync_recovery_candidate", recovery_payload)
                        if recovery.synchronized:
                            recovered = self.fsm.recover_from_sync_required(
                                recovery.state, elapsed, recovery.reason
                            )
                            transition_results.append(recovered)
                            self.logger.event("sync_recovered", {
                                **recovery_payload,
                                "recovery_target": recovery.state.value,
                                "action_intent": ActionIntent.NONE.value,
                                "action_applied": False,
                            })

                    activation = last_result.next_activation
                    if self.fsm.state != last_result.fsm.next_state:
                        activation = self.activation_policy.evaluate(
                            self.fsm.state,
                            raw_bundle,
                            recorded_observation=True,
                        )
                    next_interval = self._detector_interval(activation)
                    next_detector_due = elapsed + next_interval if next_interval is not None else float("inf")
                    latency_ms = (self.clock() - processing_started) * 1000.0
                    latencies.append(latency_ms)

                    if self.evidence_recorder is not None:
                        executed = {
                            "prompt": prompt_due,
                            "hook": hook is not None,
                            "press": press is not None,
                            "get": get is not None,
                            "result_banner": result_banner is not None,
                        }
                        try:
                            self.evidence_recorder.record_detector_evidence(
                                frame,
                                capture_frame_index=captured,
                                timestamp=elapsed,
                                episode_id=evidence_episode_id,
                                runtime_state=last_result.fsm.next_state.value,
                                roi_bounds=self._diagnostic_roi_bounds,
                                detector_metadata=self._diagnostic_metadata(
                                    prompt=prompt,
                                    raw_bundle=raw_bundle,
                                    result=last_result,
                                    executed=executed,
                                ),
                                executed=executed,
                            )
                        except Exception as exc:
                            result_name = "safe_stop_evidence_failure"
                            evidence_failure_reason = f"roi: {type(exc).__name__}: {exc}"
                            self.logger.event("diagnostic_evidence_failure", {
                                "timestamp": elapsed,
                                "frame_index": captured,
                                "reason": evidence_failure_reason,
                            })
                            break

                    if prompt and prompt.kind.value != last_prompt_kind:
                        self.logger.event("prompt_label_change", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "previous_label": last_prompt_kind,
                            "next_label": prompt.kind.value,
                            "prompt_evidence": dict(prompt.evidence),
                        })
                        last_prompt_kind = prompt.kind.value
                    current_banner_visible = bool(result_banner and result_banner.detected)
                    if current_banner_visible != result_banner_visible:
                        self.logger.event(
                            "result_banner_appearance" if current_banner_visible else "result_banner_disappearance",
                            {
                                "timestamp": elapsed,
                                "frame_index": captured,
                                "experimental": bool(
                                    result_banner and result_banner.evidence.get("experimental")
                                ),
                                "runtime_effect": "hold_result_pending_only",
                                "action_intent": ActionIntent.NONE.value,
                                "action_applied": False,
                            },
                        )
                        result_banner_visible = current_banner_visible
                    activation_payload = self._activation_payload(activation)
                    if last_activation != activation:
                        self.logger.event("detector_activation_change", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            **activation_payload,
                        })
                        last_activation = activation

                    qualified_hook = last_result.qualified.bundle.hook
                    crossed_now = bool(qualified_hook and qualified_hook.evidence.get("divider_margin_passed"))
                    if crossed_now and not hook_crossed:
                        hook_crossed = True
                        hook_crossed_at = elapsed
                        self.logger.event("hook_threshold_crossing", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "fill_ratio": qualified_hook.fill_ratio,
                            "evidence": dict(qualified_hook.evidence),
                        })
                    if not qualified_hook or not qualified_hook.detected:
                        hook_crossed = False
                        hook_crossed_at = None

                    qualified_press = last_result.qualified.bundle.press
                    selected_clean = (
                        qualified_press.evidence.get("selected_clean_frame")
                        if qualified_press else None
                    )
                    if isinstance(selected_clean, int) and selected_clean != press_frozen_frame:
                        press_frozen_frame = selected_clean
                        self.logger.event("press_sequence_frozen", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "selected_clean_frame": selected_clean,
                            "sequence": list(qualified_press.sequence_candidate),
                        })

                    qualified_get = last_result.qualified.bundle.get
                    get_now = bool(qualified_get and qualified_get.detected)
                    if get_now != get_visible:
                        event_type = "get_appearance" if get_now else "get_disappearance"
                        payload = {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "runtime_state": self.fsm.state.value,
                        }
                        self.logger.event(event_type, payload)
                        if get_now:
                            self.logger.update_cycle(self.deduplicator.cycle_id, event_type, payload)
                        get_visible = get_now

                    if last_result.evidence.has_conflict and not conflict_active:
                        screenshot = self.logger.save_screenshot(frame, captured, "detector_conflict")
                        self.logger.event("detector_conflict", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "reason": last_result.evidence.reason,
                            "screenshot_reference": screenshot,
                        })
                    conflict_active = last_result.evidence.has_conflict

                    if prompt and prompt.kind == PromptObservationKind.UNKNOWN:
                        unknown_started = elapsed if unknown_started is None else unknown_started
                        if (
                            not unknown_saved
                            and elapsed - unknown_started >= self.live_config.unknown_screenshot_seconds
                        ):
                            screenshot = self.logger.save_screenshot(frame, captured, "unknown_sustained")
                            self.logger.event("unknown_sustained", {
                                "timestamp": elapsed,
                                "frame_index": captured,
                                "duration_seconds": elapsed - unknown_started,
                                "screenshot_reference": screenshot,
                            })
                            unknown_saved = True
                    else:
                        unknown_started = None
                        unknown_saved = False

                    for transition_result in transition_results:
                        if transition_result.previous_state == transition_result.next_state:
                            continue
                        screenshot = (
                            self.logger.save_screenshot(frame, captured, "runtime_transition")
                            if self.live_config.save_transition_frames else None
                        )
                        self.logger.transition(
                            timestamp=elapsed,
                            frame_index=captured,
                            previous_state=transition_result.previous_state.value,
                            next_state=transition_result.next_state.value,
                            reason=transition_result.transition_reason,
                            screenshot_reference=screenshot,
                        )
                        self.logger.update_cycle(self.deduplicator.cycle_id, "runtime_transition", {
                            "frame_index": captured,
                            "runtime_state": transition_result.next_state.value,
                        })
                        if (
                            transition_result.next_state == RuntimeState.IDLE
                            and transition_result.previous_state not in {
                                RuntimeState.IDLE,
                                RuntimeState.SYNCING,
                                RuntimeState.SYNC_REQUIRED,
                            }
                        ):
                            self.deduplicator.finish_cycle()
                            completed_cycles += 1
                            if self.evidence_recorder is not None:
                                self.logger.event("diagnostic_cycle_completed", {
                                    "timestamp": elapsed,
                                    "frame_index": captured,
                                    "episode_id": evidence_episode_id,
                                    "completed_cycles": completed_cycles,
                                    "actions_applied": 0,
                                })
                            evidence_episode_id += 1
                            stop_after_completed_cycle = bool(
                                self.live_config.max_completed_cycles is not None
                                and completed_cycles >= self.live_config.max_completed_cycles
                            )

                    if self.fsm.state == RuntimeState.SYNC_REQUIRED and not sync_required_active:
                        screenshot = self.logger.save_screenshot(frame, captured, "sync_required")
                        sync_entry = next(
                            (
                                item for item in reversed(transition_results)
                                if item.next_state == RuntimeState.SYNC_REQUIRED
                            ),
                            None,
                        )
                        self.logger.event("SYNC_REQUIRED", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "reason": (
                                sync_entry.transition_reason
                                if sync_entry is not None
                                else sync_reason or last_result.fsm.transition_reason
                            ),
                            "screenshot_reference": screenshot,
                        })
                    sync_required_active = self.fsm.state == RuntimeState.SYNC_REQUIRED

                    specialized = self._specialized_payload(last_result)
                    would_fire = self.deduplicator.observe(
                        last_result.fsm.action_request,
                        safety_reason=last_result.safety.reason,
                        frame_index=captured,
                        timestamp=elapsed,
                        runtime_state=self.fsm.state.value,
                        prompt_evidence=prompt.evidence if prompt else None,
                        specialized_evidence=specialized,
                    )
                    if would_fire:
                        event_type = would_fire.pop("event_type")
                        screenshot = self.logger.save_screenshot(frame, captured, event_type)
                        would_fire["screenshot_reference"] = screenshot
                        if event_type == "WOULD_HOOK_ACTION" and hook_crossed_at is not None:
                            would_fire["threshold_crossing_to_would_fire_ms"] = (
                                elapsed - hook_crossed_at
                            ) * 1000.0
                        self.logger.event(event_type, would_fire)
                        self.logger.update_cycle(self.deduplicator.cycle_id, event_type, would_fire)

                capture_fps = captured / max(1e-9, self.clock() - started)
                latency_ms = latencies[-1] if latencies else 0.0
                self.overlay.show(frame, self._overlay_lines(
                    capture_fps=capture_fps,
                    latency_ms=latency_ms,
                    prompt=last_prompt,
                    result=last_result,
                    activation=activation,
                ))
                if elapsed - last_terminal >= 1.0:
                    print(
                        f"frame={captured} capture_fps={capture_fps:.1f} latency_ms={latency_ms:.1f} "
                        f"prompt={last_prompt.kind.value if last_prompt else 'N/A'} state={self.fsm.state.value} "
                        f"activation={activation.hook.value}/{activation.press.value}/{activation.get.value} "
                        f"intent={last_result.fsm.action_request.intent.value if last_result else 'NONE'} "
                        "action_applied=false",
                        flush=True,
                    )
                    last_terminal = elapsed
                if stop_after_completed_cycle:
                    result_name = "completed_target_cycles"
                    break
                loop_interval = 1.0 / self.live_config.max_fps
                self.sleep(max(0.0, loop_interval - (self.clock() - frame_loop_started)))
        except KeyboardInterrupt:
            result_name = "interrupted_by_user"
        except LivePreflightError as exc:
            result_name = "preflight_failed"
            self.logger.event("preflight_failure", {
                "timestamp": 0.0,
                "frame_index": 0,
                "reason": str(exc),
            })
        finally:
            elapsed_total = max(0.0, self.clock() - started)
            evidence_summary: dict[str, Any] = {
                "evidence_mode": "minimal",
                "video_path": None,
                "video_codec": None,
                "requested_video_codec": None,
                "attempted_codecs": [],
                "actual_video_codec": None,
                "video_codec_fallback_used": False,
                "codec_initialization_errors": [],
                "actual_video_path": None,
                "video_frame_count": 0,
                "video_fps": 0.0,
                "first_timestamp": None,
                "last_timestamp": None,
                "dropped_video_frames": 0,
                "roi_evidence_counts_by_episode": {},
                "has_evidence_gaps": False,
                "evidence_gap_intervals": [],
            }
            if self.evidence_recorder is not None:
                try:
                    evidence_summary = self.evidence_recorder.finalize()
                except Exception as exc:
                    result_name = "safe_stop_evidence_finalize_failure"
                    self.logger.event("diagnostic_evidence_failure", {
                        "timestamp": elapsed_total,
                        "frame_index": captured,
                        "reason": f"finalize: {type(exc).__name__}: {exc}",
                    })
                    evidence_summary = {
                        **evidence_summary,
                        "evidence_mode": "diagnostic",
                        "has_evidence_gaps": True,
                        "evidence_gap_intervals": [{
                            "reason": f"finalize_failure: {type(exc).__name__}: {exc}"
                        }],
                    }
            if evidence_failure_reason is not None:
                evidence_summary = {
                    **evidence_summary,
                    "has_evidence_gaps": True,
                    "evidence_gap_intervals": [
                        *evidence_summary.get("evidence_gap_intervals", []),
                        {"reason": evidence_failure_reason},
                    ],
                }
            self.overlay.close()
            if self._opened:
                self.capture.close()
                self._opened = False
            summary = {
                "result": result_name,
                "duration_seconds": elapsed_total,
                "captured_frames": captured,
                "processed_frames": processed,
                "capture_fps": captured / elapsed_total if elapsed_total > 0 else 0.0,
                "mean_processing_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
                "max_processing_latency_ms": max(latencies) if latencies else 0.0,
                "detector_runs": dict(detector_runs),
                "detector_achieved_fps": {
                    name: count / elapsed_total if elapsed_total > 0 else 0.0
                    for name, count in detector_runs.items()
                },
                "raw_action_proposals": dict(self.deduplicator.raw_proposals),
                "unique_would_fire": dict(self.deduplicator.unique_events),
                "actions_applied": actions_applied,
                "final_state": self.fsm.state.value,
                "emit_actions": False,
                "action_sink": None,
                "capture_backend": self._capture_diagnostics.get(
                    "backend", getattr(self.capture, "backend_name", "unknown")
                ),
                "capture_fallback_used": bool(self._capture_diagnostics.get("fallback_used", False)),
                "capture_diagnostics": self._capture_diagnostics,
                "completed_cycles": completed_cycles,
                "max_completed_cycles": self.live_config.max_completed_cycles,
                **evidence_summary,
            }
            self.logger.finalize(summary)
        return summary
