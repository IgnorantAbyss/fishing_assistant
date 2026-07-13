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
from src.fishing_v2.domain.observations import PromptObservation, PromptObservationKind
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import ObservationFusion
from src.fishing_v2.legacy_adapters.get_detector_adapter import LegacyGetDetectorAdapter
from src.fishing_v2.legacy_adapters.hook_detector_adapter import LegacyHookDetectorAdapter
from src.fishing_v2.legacy_adapters.press_detector_adapter import LegacyPressDetectorAdapter
from src.fishing_v2.live.session_logger import LiveSessionLogger
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.perception.prompt_bundle import LoadedPromptBundle
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

    def __post_init__(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.max_fps <= 0:
            raise ValueError("max_fps must be positive")
        if self.unknown_screenshot_seconds <= 0:
            raise ValueError("unknown_screenshot_seconds must be positive")


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
        self.schedule = _prompt_polling(self._raw_config)
        self._opened = False

    def preflight(self) -> np.ndarray:
        try:
            self.capture.open()
            self._opened = True
            frame = self.capture.capture()
        except Exception as exc:
            raise LivePreflightError(f"Capture preflight failed: {exc}") from exc
        height, width = frame.shape[:2]
        if (width, height) != EXPECTED_RESOLUTION:
            raise LivePreflightError(
                f"Unsupported capture resolution {width}x{height}; expected 2560x1440"
            )
        self.prompt_bundle.roi.pixel_bounds(width, height)
        if len(self.prompt_bundle.model.prototypes) != 35:
            raise LivePreflightError("Final Prompt bundle must contain 35 prototypes")
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
        return {
            "hook": asdict(hook) if hook else None,
            "press": asdict(press) if press else None,
            "get": asdict(get) if get else None,
            "qualified": {
                "hook": result.qualified.hook.qualified_detected,
                "press": result.qualified.press.qualified_detected,
                "get": result.qualified.get.qualified_detected,
            },
        }

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
        request = result.fsm.action_request if result else ActionRequest(ActionIntent.NONE, 0.0, "none")
        return [
            f"DETECT ONLY | capture={capture_fps:.1f} FPS latency={latency_ms:.1f} ms bundle={self.prompt_bundle.bundle_version}",
            f"Prompt raw={evidence.get('raw_predicted_label', 'N/A')} predicted={prompt.kind.value if prompt else 'N/A'} sim={evidence.get('similarity', 0.0):.3f} second={evidence.get('second_label', 'N/A')} margin={evidence.get('ambiguity_margin', 0.0):.3f}",
            f"Runtime={self.fsm.state.value} activation H/P/G={activation.hook.value}/{activation.press.value}/{activation.get.value}",
            f"Hook raw/qualified={bool(hook)}/{bool(hook and hook.detected)} fill={getattr(hook, 'fill_ratio', None)} crossed={bool(hook and hook.evidence.get('divider_margin_passed'))}",
            f"Press panel={bool(press and press.panel_present)} sequence={''.join(press.sequence_candidate) if press else ''} frozen={bool(press and press.sequence_ready)}",
            f"Get panel={bool(get and get.detected)} | would-fire={request.intent.value} | action_applied=false",
        ]

    def run(self, *, max_frames: int | None = None) -> dict[str, Any]:
        captured = processed = actions_applied = 0
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
            })
            while self.clock() - started < self.live_config.duration_seconds:
                if max_frames is not None and captured >= max_frames:
                    break
                frame_loop_started = self.clock()
                elapsed = frame_loop_started - started
                try:
                    frame = self.capture.capture()
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
                prompt_due = elapsed >= next_prompt_due
                detector_interval = self._detector_interval(activation)
                detector_due = detector_interval is not None and elapsed >= next_detector_due
                should_process = prompt_due or detector_due
                if should_process:
                    processing_started = self.clock()
                    context = FrameContext(captured, elapsed, metadata={"source": "live_detect_only"})
                    if prompt_due:
                        last_prompt = self.prompt_bundle.observer.observe(frame, context)
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
                    detector_runs.update({
                        "hook": int(hook is not None),
                        "press": int(press is not None),
                        "get": int(get is not None),
                    })
                    raw_bundle = ObservationBundle(captured, elapsed, prompt, hook, press, get)
                    sync_reason = None
                    if self.fsm.state == RuntimeState.SYNCING:
                        _, startup_qualified = self.controller.qualify_raw_bundle(
                            raw_bundle, action_mode=ActionExecutionMode.RECORDED_OBSERVATION
                        )
                        sync = self.synchronizer.observe(startup_qualified.bundle)
                        sync_reason = sync.reason
                        if sync.synchronized:
                            self.fsm.force_state(sync.state, elapsed, sync.reason)
                        elif sync.state == RuntimeState.SYNC_REQUIRED:
                            self.fsm.force_state(RuntimeState.SYNC_REQUIRED, elapsed, sync.reason)
                    previous_state = self.fsm.state
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
                    activation = last_result.next_activation
                    next_interval = self._detector_interval(activation)
                    next_detector_due = elapsed + next_interval if next_interval is not None else float("inf")
                    latency_ms = (self.clock() - processing_started) * 1000.0
                    latencies.append(latency_ms)

                    if prompt and prompt.kind.value != last_prompt_kind:
                        self.logger.event("prompt_label_change", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "previous_label": last_prompt_kind,
                            "next_label": prompt.kind.value,
                            "prompt_evidence": dict(prompt.evidence),
                        })
                        last_prompt_kind = prompt.kind.value
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

                    if last_result.fsm.previous_state != last_result.fsm.next_state:
                        screenshot = (
                            self.logger.save_screenshot(frame, captured, "runtime_transition")
                            if self.live_config.save_transition_frames else None
                        )
                        self.logger.transition(
                            timestamp=elapsed,
                            frame_index=captured,
                            previous_state=last_result.fsm.previous_state.value,
                            next_state=last_result.fsm.next_state.value,
                            reason=sync_reason or last_result.fsm.transition_reason,
                            screenshot_reference=screenshot,
                        )
                        self.logger.update_cycle(self.deduplicator.cycle_id, "runtime_transition", {
                            "frame_index": captured,
                            "runtime_state": last_result.fsm.next_state.value,
                        })
                        if (
                            last_result.fsm.next_state == RuntimeState.IDLE
                            and last_result.fsm.previous_state not in {RuntimeState.IDLE, RuntimeState.SYNCING}
                        ):
                            self.deduplicator.finish_cycle()

                    if self.fsm.state == RuntimeState.SYNC_REQUIRED and not sync_required_active:
                        screenshot = self.logger.save_screenshot(frame, captured, "sync_required")
                        self.logger.event("SYNC_REQUIRED", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "reason": sync_reason or last_result.fsm.transition_reason,
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
            }
            self.logger.finalize(summary)
        return summary
