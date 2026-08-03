"""Real-frame observation loop with detect-only defaults and guarded opt-in actions."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
from src.fishing_v2.legacy_adapters.press_background_subtraction_v3_adapter import (
    BackgroundSubtractionPressDetectorAdapter,
)
from src.fishing_v2.live.session_logger import LiveSessionLogger
from src.fishing_v2.live.cast_opportunity import (
    CastBlocker,
    CastAttempt,
    CastOpportunityConfig,
    CastOpportunityController,
    CastOpportunityEvent,
    PostCycleClearanceStatus,
    PostCycleClearanceTracker,
    PresenceState,
)
from src.fishing_v2.live.collect_retry import (
    CollectAttempt,
    CollectRetryConfig,
    CollectRetryController,
    CollectRetryEvent,
)
from src.fishing_v2.live.diagnostic_evidence import (
    EVIDENCE_MODES,
    DiagnosticEvidenceConfig,
    DiagnosticEvidenceRecorder,
)
from src.fishing_v2.live.console_events import LiveConsoleEventQueue
from src.fishing_v2.live.hook_critical_loop import (
    HookCriticalFrameAssembler,
    HookDecisionTraceBuffer,
    HookEpisodeTelemetry,
    HookROIFrame,
    LatestHookFrameSlot,
)
from src.fishing_v2.live.hook_action_lifecycle import HookActionLifecycle
from src.fishing_v2.live.idle_recovery import (
    CAST_SOURCE_POST_CYCLE,
    CAST_SOURCE_RECOVERY,
    CAST_SOURCE_STARTUP,
    CastArmingLifecycle,
    IdleRecoveryCertificate,
    IdleRecoveryConfig,
    IdleRecoveryTracker,
)
from src.fishing_v2.live.press_shadow_verification import PressShadowVerifier
from src.fishing_v2.live.press_v3_shadow import (
    PressV3ShadowConfig,
    PressV3ShadowRunner,
)
from src.fishing_v2.live.press_anomaly_evidence import (
    PressAnomalyEvidenceConfig,
    PressAnomalyEvidenceRecorder,
)
from src.fishing_v2.live.press_live_emission import (
    PressLiveEmissionConfig,
    PressLiveEmissionTracker,
    PressTimingRng,
    pending_press_cancellation_reason,
)
from src.fishing_v2.live.windows_action_sink import (
    ACTION_SINK_NONE,
    ACTION_SINK_SENDINPUT,
    ActionIntegrityPreflightError,
    EXPECTED_GAME_PROCESS,
    WindowsActionConfig,
    WindowsSendInputActionSink,
    parse_action_allowlist,
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
from src.fishing_v2.runtime.scheduling import (
    MissedReadyRecoveryTracker,
    PromptPollingConfig,
    RuntimeSchedulePolicy,
)
from src.fishing_v2.runtime.synchronization import StartupSynchronizer
from src.fishing_v2.ports.action_sink import (
    ActionExecutionContext,
    ActionExecutionResult,
    ActionSink,
)
from src.screen_capture import validate_bgr_frame
from src.config_loader import load_roi_config


EXPECTED_RESOLUTION = (2560, 1440)
RUNTIME_PROFILES = ("production", "diagnostic")
WOULD_FIRE_NAMES = {
    ActionIntent.CAST: "WOULD_CAST",
    ActionIntent.START_HOOK: "WOULD_START_HOOK",
    ActionIntent.HOOK_ACTION: "WOULD_HOOK_ACTION",
    ActionIntent.PRESS_SEQUENCE: "WOULD_PRESS_SEQUENCE",
    ActionIntent.COLLECT: "WOULD_COLLECT",
}
LIVE_ACTION_ALLOWLIST = frozenset({
    ActionIntent.CAST,
    ActionIntent.START_HOOK,
    ActionIntent.HOOK_ACTION,
    ActionIntent.COLLECT,
    ActionIntent.PRESS_SEQUENCE,
})
LIVE_ACTION_ALLOWLIST_DISPLAY = (
    "CAST,START_HOOK,HOOK_ACTION,COLLECT"
)
PRESS_DETECTOR_MODES = (
    "legacy",
    "background-subtraction-shadow",
    "background-subtraction-live",
)


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
    hook_critical_target_fps: float = 40.0
    max_completed_cycles: int | None = None
    press_initial_delay_min_ms: int = 300
    press_initial_delay_max_ms: int = 500
    press_inter_key_gap_min_ms: int = 90
    press_inter_key_gap_max_ms: int = 170
    press_key_hold_ms: int = 40
    hook_action_stall_timeout_seconds: float = 3.0
    idle_recovery_window_size: int = 5
    idle_recovery_required_count: int = 4
    idle_recovery_min_window_seconds: float = 0.5
    idle_recovery_freshness_ms: float = 250.0
    idle_recovery_cast_cooldown_seconds: float = 0.5
    idle_cast_retry_min_interval_seconds: float = 3.0
    idle_cast_liveness_timeout_seconds: float = 3.0
    press_anomaly_evidence: bool = False
    press_anomaly_buffer_frames: int = 12
    press_anomaly_max_episodes: int = 20
    press_detector_mode: str = "legacy"
    press_v3_debug_evidence: bool = False
    press_v3_debug_max_episodes: int = 20
    press_v3_debug_max_frames_per_episode: int = 8
    # Programmatic callers retain the historical diagnostic behavior; the
    # public CLI explicitly defaults to Production.
    runtime_profile: str = "diagnostic"

    def __post_init__(self) -> None:
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        if self.max_fps <= 0:
            raise ValueError("max_fps must be positive")
        if self.unknown_screenshot_seconds <= 0:
            raise ValueError("unknown_screenshot_seconds must be positive")
        if self.evidence_mode not in EVIDENCE_MODES:
            raise ValueError(f"evidence_mode must be one of {EVIDENCE_MODES}")
        if self.evidence_video_fps <= 0:
            raise ValueError("evidence_video_fps must be positive")
        if self.hook_critical_target_fps < 30.0:
            raise ValueError("hook_critical_target_fps must be at least 30")
        if self.max_completed_cycles is not None and self.max_completed_cycles < 0:
            raise ValueError("max_completed_cycles must be non-negative when provided")
        if self.runtime_profile not in RUNTIME_PROFILES:
            raise ValueError(
                f"runtime_profile must be one of {RUNTIME_PROFILES}"
            )
        if self.press_initial_delay_min_ms < 0:
            raise ValueError("press_initial_delay_min_ms must be non-negative")
        if self.press_initial_delay_min_ms > self.press_initial_delay_max_ms:
            raise ValueError("PRESS initial delay range is inverted")
        if self.press_inter_key_gap_min_ms < 0:
            raise ValueError("press_inter_key_gap_min_ms must be non-negative")
        if self.press_inter_key_gap_min_ms > self.press_inter_key_gap_max_ms:
            raise ValueError("PRESS inter-key gap range is inverted")
        if self.press_key_hold_ms <= 0:
            raise ValueError("press_key_hold_ms must be positive")
        PressAnomalyEvidenceConfig(
            enabled=self.press_anomaly_evidence,
            buffer_frames=self.press_anomaly_buffer_frames,
            max_episodes=self.press_anomaly_max_episodes,
        )
        if self.press_detector_mode not in PRESS_DETECTOR_MODES:
            raise ValueError(
                f"press_detector_mode must be one of {PRESS_DETECTOR_MODES}"
            )
        PressV3ShadowConfig(
            debug_evidence=self.press_v3_debug_evidence,
            debug_max_episodes=self.press_v3_debug_max_episodes,
            debug_max_frames_per_episode=(
                self.press_v3_debug_max_frames_per_episode
            ),
        )
        if self.hook_action_stall_timeout_seconds <= 0:
            raise ValueError(
                "hook_action_stall_timeout_seconds must be positive"
            )
        IdleRecoveryConfig(
            window_size=self.idle_recovery_window_size,
            required_count=self.idle_recovery_required_count,
            min_window_seconds=(
                self.idle_recovery_min_window_seconds
            ),
            freshness_ms=self.idle_recovery_freshness_ms,
            cast_cooldown_seconds=(
                self.idle_recovery_cast_cooldown_seconds
            ),
            cast_retry_min_interval_seconds=(
                self.idle_cast_retry_min_interval_seconds
            ),
            cast_liveness_timeout_seconds=(
                self.idle_cast_liveness_timeout_seconds
            ),
        )


class WouldFireDeduplicator:
    """Summarize repeated proposals as one opportunity per fishing cycle."""

    def __init__(self) -> None:
        self.cycle_id = 1
        self.raw_proposals: Counter[str] = Counter()
        self.unique_events: Counter[str] = Counter()
        self._seen: set[tuple[int, ActionIntent, str | None]] = set()
        self._recovered_cycles: set[int] = set()

    @property
    def has_cycle_activity(self) -> bool:
        return any(cycle == self.cycle_id for cycle, _, _ in self._seen)

    def finish_cycle(self) -> None:
        if self.has_cycle_activity or self.cycle_id in self._recovered_cycles:
            self.cycle_id += 1

    def begin_recovered_cycle(self) -> int:
        """Reserve the current clean identity for a manually-started cycle."""
        if self.has_cycle_activity:
            self.cycle_id += 1
        self._recovered_cycles.add(self.cycle_id)
        return self.cycle_id

    def reset_for_sync_recovery(self, *, preserve_cycle: bool = False) -> None:
        """Prevent pre-loss proposals from suppressing the recovered cycle."""
        if (
            not preserve_cycle
            and (self.has_cycle_activity or self.cycle_id in self._recovered_cycles)
        ):
            self.cycle_id += 1
        if not preserve_cycle:
            self._seen.clear()

    def release(
        self,
        intent: ActionIntent,
        identity_suffix: str | None = None,
    ) -> None:
        """Release a proposal that never reached action emission."""
        self._seen.discard((self.cycle_id, intent, identity_suffix))

    def opportunity_id(
        self,
        intent: ActionIntent,
        identity_suffix: str | None = None,
    ) -> str:
        value = f"cycle:{self.cycle_id}:{intent.value}"
        return f"{value}:{identity_suffix}" if identity_suffix else value

    def already_consumed(
        self,
        intent: ActionIntent,
        identity_suffix: str | None = None,
    ) -> bool:
        return (self.cycle_id, intent, identity_suffix) in self._seen

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
        identity_suffix: str | None = None,
        count_raw: bool = True,
    ) -> dict[str, Any] | None:
        if request.intent == ActionIntent.NONE:
            return None
        if count_raw:
            self.record_raw_proposal(request)
        # This reason means every production safety check passed and only the
        # immutable detect-only emission switch prevented execution.
        if safety_reason != "action_emission_disabled":
            return None
        key = (self.cycle_id, request.intent, identity_suffix)
        if key in self._seen:
            return None
        self._seen.add(key)
        event_type = WOULD_FIRE_NAMES[request.intent]
        self.unique_events[event_type] += 1
        deduplication_key = self.opportunity_id(
            request.intent,
            identity_suffix,
        )
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
            "deduplication_key": deduplication_key,
        }

    def record_raw_proposal(self, request: ActionRequest) -> None:
        if request.intent != ActionIntent.NONE:
            self.raw_proposals[request.intent.value] += 1


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


def validate_emit_actions(
    emit_actions: bool,
    action_sink: str = ACTION_SINK_NONE,
    action_allowlist: str | tuple[str, ...] | list[str] = (),
    *,
    enable_live_press_sequence: bool = False,
) -> None:
    """Require two independent CLI opt-ins before constructing an input sink."""
    if emit_actions and action_sink != ACTION_SINK_SENDINPUT:
        raise LivePreflightError(
            "--emit-actions=true requires the explicit --action-sink=sendinput opt-in"
        )
    if not emit_actions and action_sink != ACTION_SINK_NONE:
        raise LivePreflightError(
            "--action-sink=sendinput requires --emit-actions=true"
        )
    try:
        allowlist = parse_action_allowlist(action_allowlist)
    except ValueError as exc:
        raise LivePreflightError(str(exc)) from exc
    if emit_actions and not allowlist:
        raise LivePreflightError(
            "An explicit non-empty --action-allowlist is required when actions are enabled"
        )
    press_allowlisted = ActionIntent.PRESS_SEQUENCE in allowlist
    if press_allowlisted and not enable_live_press_sequence:
        raise LivePreflightError(
            "PRESS_SEQUENCE requires the explicit "
            "--enable-live-press-sequence opt-in"
        )
    if enable_live_press_sequence and not emit_actions:
        raise LivePreflightError(
            "--enable-live-press-sequence requires --emit-actions=true"
        )
    if enable_live_press_sequence and not press_allowlisted:
        raise LivePreflightError(
            "--enable-live-press-sequence requires PRESS_SEQUENCE in "
            "--action-allowlist"
        )
    unsupported = allowlist - LIVE_ACTION_ALLOWLIST
    if emit_actions and unsupported:
        names = ", ".join(sorted(item.value for item in unsupported))
        raise LivePreflightError(
            "Live actions are limited to "
            f"{LIVE_ACTION_ALLOWLIST_DISPLAY}; refused: {names}"
        )


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
        action_sink_name: str = ACTION_SINK_NONE,
        action_allowlist: str | tuple[str, ...] | list[str] = (),
        enable_live_press_sequence: bool = False,
        panic_key: str = "F12",
        action_sink_factory: Callable[..., ActionSink] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        press_timing_rng: PressTimingRng | None = None,
        console_queue: LiveConsoleEventQueue | None = None,
    ) -> None:
        validate_emit_actions(
            emit_actions,
            action_sink_name,
            action_allowlist,
            enable_live_press_sequence=enable_live_press_sequence,
        )
        self.config_path = Path(config_path)
        self.prompt_bundle = prompt_bundle
        self.capture = capture
        self.logger = logger
        self.live_config = live_config
        self.hook_detector = hook_detector or LegacyHookDetectorAdapter()
        if press_detector is not None:
            self.press_detector = press_detector
        elif live_config.press_detector_mode == "background-subtraction-live":
            self.press_detector = BackgroundSubtractionPressDetectorAdapter()
        else:
            self.press_detector = LegacyPressDetectorAdapter()
        self.get_detector = get_detector or LegacyGetDetectorAdapter()
        self.emit_actions = bool(emit_actions)
        self.action_sink_name = action_sink_name
        self.action_allowlist = parse_action_allowlist(action_allowlist)
        self.enable_live_press_sequence = bool(
            enable_live_press_sequence
        )
        self.panic_key = panic_key
        self.action_sink_factory = action_sink_factory or WindowsSendInputActionSink
        self.action_sink: ActionSink | None = None
        self.clock = clock
        self.sleep = sleep
        self.console = console_queue or LiveConsoleEventQueue()
        self.overlay = DiagnosticOverlay(live_config.show_overlay)
        self.deduplicator = WouldFireDeduplicator()
        self._raw_config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.collect_retry = CollectRetryController(
            CollectRetryConfig.from_mapping(self._raw_config.get("collect"))
        )
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
            raise LivePreflightError(
                "Runtime safety.emit_actions must remain false; the guarded Live sink is CLI-only"
            )
        self.fsm = FishingFSM(fsm_config, initial_state=RuntimeState.SYNCING)
        cast_config = CastOpportunityConfig(
            visual_ack_timeout_seconds=fsm_config.cast_pending_timeout_sec,
            clearance_freshness_seconds=fsm_config.cast_pending_timeout_sec,
            stable_idle_frames_required=fsm_config.stable_frames,
        )
        self.cast_opportunity = CastOpportunityController(cast_config)
        self.cast_clearance = PostCycleClearanceTracker(cast_config)
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
        self.missed_ready_recovery = MissedReadyRecoveryTracker(
            stable_frames=fsm_config.stable_frames,
            min_confidence=fusion_config.prompt_min_confidence,
            confirmation_timeout_seconds=(
                self.schedule.config.ready_confirmation_timeout_seconds
            ),
        )
        self._opened = False
        self._capture_diagnostics: dict[str, Any] = {}
        self.evidence_recorder = evidence_recorder
        if (
            self.live_config.runtime_profile == "diagnostic"
            and self.live_config.evidence_mode == "diagnostic"
        ):
            self.evidence_recorder = self.evidence_recorder or DiagnosticEvidenceRecorder(
                self.logger.path,
                config=DiagnosticEvidenceConfig(video_fps=self.live_config.evidence_video_fps),
            )
            self.logger.set_event_listener(self.evidence_recorder.mark_event)
        elif self.evidence_recorder is not None:
            raise ValueError("evidence_recorder requires evidence_mode='diagnostic'")
        self._diagnostic_roi_bounds: dict[str, tuple[int, int, int, int]] = {}
        self._hook_critical_bounds: tuple[int, int, int, int] | None = None
        self._hook_frame_assembler: HookCriticalFrameAssembler | None = None
        self._latest_hook_frame = LatestHookFrameSlot()
        self._hook_episode_telemetry = HookEpisodeTelemetry()
        self._hook_action_lifecycle = HookActionLifecycle(
            self.live_config.hook_action_stall_timeout_seconds
        )
        idle_recovery_config = IdleRecoveryConfig(
            window_size=self.live_config.idle_recovery_window_size,
            required_count=(
                self.live_config.idle_recovery_required_count
            ),
            min_window_seconds=(
                self.live_config.idle_recovery_min_window_seconds
            ),
            freshness_ms=self.live_config.idle_recovery_freshness_ms,
            cast_cooldown_seconds=(
                self.live_config.idle_recovery_cast_cooldown_seconds
            ),
            cast_retry_min_interval_seconds=(
                self.live_config.idle_cast_retry_min_interval_seconds
            ),
            cast_liveness_timeout_seconds=(
                self.live_config.idle_cast_liveness_timeout_seconds
            ),
        )
        self._idle_recovery = IdleRecoveryTracker(idle_recovery_config)
        self._cast_arming = CastArmingLifecycle(
            retry_min_interval_seconds=(
                idle_recovery_config.cast_retry_min_interval_seconds
            )
        )
        self._startup_idle_handled = False
        self._runtime_cycle_started = False
        self._idle_recovery_sequence = 0
        self._idle_candidate_origin_state: RuntimeState | None = None
        self._idle_liveness_started_at: float | None = None
        self._idle_liveness_physical_idle_id: str | None = None
        self._idle_liveness_armed_physical_ids: set[str] = set()
        self._action_emission_in_progress = False
        self._hook_roi_clip_samples: list[HookROIFrame] | None = (
            [] if self.evidence_recorder is not None else None
        )
        self._hook_decision_trace = (
            HookDecisionTraceBuffer(max_entries=512)
            if self.evidence_recorder is not None else None
        )
        self._hook_decision_trace_path: Path | None = None
        self._hook_decision_trace_rows_written = 0
        self._press_shadow = PressShadowVerifier(
            diagnostics_enabled=self.evidence_recorder is not None
        )
        self._press_v3_shadow = (
            PressV3ShadowRunner(
                output_root=self.logger.path,
                config=PressV3ShadowConfig(
                    debug_evidence=self.live_config.press_v3_debug_evidence,
                    debug_max_episodes=(
                        self.live_config.press_v3_debug_max_episodes
                    ),
                    debug_max_frames_per_episode=(
                        self.live_config.press_v3_debug_max_frames_per_episode
                    ),
                ),
            )
            if self.live_config.press_detector_mode
            == "background-subtraction-shadow"
            else None
        )
        self._press_anomaly_evidence = PressAnomalyEvidenceRecorder(
            self.logger.path,
            PressAnomalyEvidenceConfig(
                enabled=self.live_config.press_anomaly_evidence,
                buffer_frames=self.live_config.press_anomaly_buffer_frames,
                max_episodes=self.live_config.press_anomaly_max_episodes,
            ),
        )
        self._last_press_completeness_log_at = float("-inf")
        self._press_abstained_episodes: set[int] = set()
        self._press_live_emission = PressLiveEmissionTracker(
            PressLiveEmissionConfig(
                visual_ack_timeout_seconds=float(
                    self._raw_config["action"].get(
                        "press_visual_ack_timeout_seconds",
                        3.0,
                    )
                ),
                initial_delay_min_ms=(
                    self.live_config.press_initial_delay_min_ms
                ),
                initial_delay_max_ms=(
                    self.live_config.press_initial_delay_max_ms
                ),
                inter_key_gap_min_ms=(
                    self.live_config.press_inter_key_gap_min_ms
                ),
                inter_key_gap_max_ms=(
                    self.live_config.press_inter_key_gap_max_ms
                ),
                key_hold_ms=int(
                    self.live_config.press_key_hold_ms
                ),
            ),
            rng=press_timing_rng,
        )
        self._hook_video_suspensions: list[dict[str, Any]] = []
        self._hook_video_suspension_active: dict[str, Any] | None = None
        self._press_seen_cycles: set[int] = set()
        self._preflight_passed = False
        self._preflight_failure_reason: str | None = None
        self._preflight_failure_message: str | None = None
        self._action_preflight_diagnostics: dict[str, Any] = {}
        self._foreground_unavailable_event_active = False
        self._last_cast_blockers: tuple[str, ...] | None = None
        self._logged_hook_action_blockers: set[
            tuple[str, tuple[str, ...]]
        ] = set()
        self._post_collect_banner_fps = float(
            self._raw_config["result"].get("get_burst_fps", 20.0)
        )

    def _log_collect_events(
        self,
        events: tuple[CollectRetryEvent, ...],
        *,
        timestamp: float,
        frame_index: int,
        runtime_state: str,
    ) -> None:
        integrity_diagnostics: Mapping[str, Any] = {}
        if self.action_sink is not None:
            sink_summary = getattr(self.action_sink, "summary", None)
            if callable(sink_summary):
                integrity_diagnostics = dict(
                    sink_summary().get("integrity_diagnostics", {})
                )
        for event in events:
            if event.event_type == "collect_retry_succeeded":
                self.console.emit("COLLECT acknowledged")
            self.logger.event(event.event_type, {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "runtime_state": runtime_state,
                "target_hwnd": self._capture_diagnostics.get("hwnd"),
                "foreground_hwnd": None,
                "os_input_emitted": False,
                "integrity_diagnostics": integrity_diagnostics,
                "visual_acknowledgement_state": (
                    "acknowledged"
                    if event.payload.get("collect_visual_acknowledged")
                    else "pending"
                ),
                **dict(event.payload),
            })
            if event.event_type in {
                "collect_retry_succeeded",
                "collect_retry_exhausted",
                "collect_retry_cancelled",
            }:
                self.logger.event("get_episode_terminal", {
                    "timestamp": timestamp,
                    "frame_index": frame_index,
                    "runtime_state": runtime_state,
                    "terminal_reason": event.payload.get(
                        "cancellation_reason",
                        event.payload.get("outcome"),
                    ),
                    **dict(event.payload),
                })

    def _log_cast_events(
        self,
        events: tuple[CastOpportunityEvent, ...],
        *,
        timestamp: float,
        frame_index: int,
        runtime_state: str,
    ) -> None:
        for event in events:
            self.logger.event(event.event_type, {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "runtime_state": runtime_state,
                "action_applied": False,
                **dict(event.payload),
            })

    def _log_transition(
        self,
        *,
        timestamp: float,
        frame_index: int,
        previous_state: str,
        next_state: str,
        reason: str,
        screenshot_reference: str | None,
    ) -> None:
        self.logger.transition(
            timestamp=timestamp,
            frame_index=frame_index,
            previous_state=previous_state,
            next_state=next_state,
            reason=reason,
            screenshot_reference=screenshot_reference,
        )
        if self.live_config.runtime_profile == "production":
            self.console.emit(
                f"state: {previous_state} -> {next_state} ({reason})"
            )

    def _cast_blockers(
        self,
        *,
        status: PostCycleClearanceStatus,
        runtime_state: RuntimeState,
        prompt_kind: PromptObservationKind | None,
        foreground: bool | None,
        raw_intent: ActionIntent,
        safety_reason: str,
        source_type: str | None = None,
        idle_certificate_valid: bool = False,
        cast_arm_ready: bool = False,
    ) -> tuple[str, ...]:
        blockers: list[CastBlocker] = []
        if runtime_state != RuntimeState.IDLE:
            blockers.append(CastBlocker.RUNTIME_NOT_IDLE)
        if prompt_kind != PromptObservationKind.IDLE_CAST:
            blockers.append(CastBlocker.PROMPT_NOT_IDLE_CAST)
        if (
            runtime_state == RuntimeState.IDLE
            and prompt_kind == PromptObservationKind.IDLE_CAST
            and source_type
            not in {CAST_SOURCE_STARTUP, CAST_SOURCE_RECOVERY}
            and status.stable_idle_frames
            < self.cast_clearance.config.stable_idle_frames_required
        ):
            blockers.append(CastBlocker.IDLE_NOT_STABLE)
        if self.collect_retry.episode_open:
            blockers.append(CastBlocker.PHYSICAL_GET_EPISODE_ACTIVE)
            blockers.append(CastBlocker.COLLECT_NOT_COMPLETED)
        if self.cast_opportunity.waiting_for_acknowledgement:
            blockers.append(CastBlocker.CAST_WAITING_ACK)
        elif self.cast_opportunity.has_scheduled_clearance(
            status.clearance_id
        ):
            blockers.append(CastBlocker.OPPORTUNITY_ALREADY_CONSUMED)

        requires_post_cycle_clearance = source_type not in {
            CAST_SOURCE_STARTUP,
            CAST_SOURCE_RECOVERY,
        }
        if requires_post_cycle_clearance and not status.clearance_available:
            freshness_ms = (
                self.cast_clearance.config.clearance_freshness_seconds * 1000.0
            )
            get_absence_stale = bool(
                status.get_presence_state == PresenceState.ABSENT
                and status.get_evidence_age_ms is not None
                and status.get_evidence_age_ms > freshness_ms
            )
            banner_absence_stale = bool(
                status.result_banner_presence_state == PresenceState.ABSENT
                and status.result_banner_evidence_age_ms is not None
                and status.result_banner_evidence_age_ms > freshness_ms
            )
            if status.clearance_consumed:
                blockers.append(CastBlocker.OPPORTUNITY_ALREADY_CONSUMED)
            elif status.get_presence_state == PresenceState.UNKNOWN:
                blockers.append(CastBlocker.GET_PRESENCE_UNKNOWN)
            elif status.get_presence_state == PresenceState.PRESENT:
                blockers.append(CastBlocker.GET_PRESENT)
            elif status.clearance_expired or get_absence_stale:
                blockers.append(CastBlocker.GET_ABSENCE_STALE)

            if status.result_banner_presence_state == PresenceState.UNKNOWN:
                previous_banner_stale = bool(
                    status.previous_result_banner_presence_state
                    == PresenceState.ABSENT
                    and status.previous_result_banner_evidence_age_ms
                    is not None
                    and status.previous_result_banner_evidence_age_ms
                    > freshness_ms
                )
                blockers.append(
                    CastBlocker.RESULT_BANNER_ABSENCE_STALE
                    if previous_banner_stale
                    else CastBlocker.RESULT_BANNER_UNKNOWN
                )
            elif status.result_banner_presence_state == PresenceState.PRESENT:
                blockers.append(CastBlocker.RESULT_BANNER_PRESENT)
            elif status.clearance_expired or banner_absence_stale:
                blockers.append(CastBlocker.RESULT_BANNER_ABSENCE_STALE)
            if not status.clearance_consumed and not status.clearance_expired:
                blockers.append(CastBlocker.POST_CYCLE_CLEARANCE_MISSING)
        if source_type in {
            CAST_SOURCE_STARTUP,
            CAST_SOURCE_RECOVERY,
        }:
            if not idle_certificate_valid:
                blockers.append(CastBlocker.IDLE_CERTIFICATE_MISSING)
            elif not cast_arm_ready:
                blockers.append(CastBlocker.CAST_COOLDOWN_ACTIVE)

        sink_summary = getattr(self.action_sink, "summary", None)
        panic_latched = bool(
            callable(sink_summary) and sink_summary().get("panic_triggered", False)
        )
        if panic_latched:
            blockers.append(CastBlocker.PANIC_LATCHED)
        if self._foreground_unavailable_event_active:
            blockers.append(CastBlocker.FOREGROUND_UNAVAILABLE)
        elif foreground is not True:
            blockers.append(CastBlocker.FOREGROUND_NOT_CONFIRMED)
        if (
            raw_intent == ActionIntent.CAST
            and safety_reason != "action_emission_disabled"
            and not any(item in blockers for item in {
                CastBlocker.PANIC_LATCHED,
                CastBlocker.FOREGROUND_UNAVAILABLE,
                CastBlocker.FOREGROUND_NOT_CONFIRMED,
            })
        ):
            blockers.append(CastBlocker.SAFETY_NOT_READY)
        return tuple(dict.fromkeys(item.value for item in blockers))

    def _log_cast_blockers(
        self,
        blockers: tuple[str, ...],
        *,
        status: PostCycleClearanceStatus,
        timestamp: float,
        frame_index: int,
        prompt_kind: PromptObservationKind | None,
        raw_intent: ActionIntent,
        cycle_completed: bool,
        transition_from: RuntimeState | None,
        transition_to: RuntimeState | None,
    ) -> None:
        if blockers == self._last_cast_blockers:
            return
        self._last_cast_blockers = blockers
        if not blockers:
            return
        self.logger.event("cast_opportunity_blocked", {
            "timestamp": timestamp,
            "frame_index": frame_index,
            "runtime_state": self.fsm.state.value,
            "prompt_label": prompt_kind.value if prompt_kind else None,
            "raw_intent": raw_intent.value,
            "stable_idle_frames": status.stable_idle_frames,
            "get_presence_state": status.get_presence_state.value,
            "get_evidence_age_ms": status.get_evidence_age_ms,
            "get_absence_source": status.get_absence_source,
            "current_clearance_id": status.clearance_id,
            "current_clearance_source": status.clearance_source,
            "current_clearance_consumed": status.clearance_consumed,
            "previous_consumed_clearance_id": (
                status.previous_consumed_clearance_id
            ),
            "active_cast_opportunity_id": (
                self.cast_opportunity.opportunity_id
                if self.cast_opportunity.opportunity_open else None
            ),
            "active_cast_terminal_outcome": (
                self.cast_opportunity.terminal_outcome.value
            ),
            "result_banner_presence_state": (
                status.result_banner_presence_state.value
            ),
            "result_banner_evidence_age_ms": (
                status.result_banner_evidence_age_ms
            ),
            "previous_result_banner_presence_state": (
                status.previous_result_banner_presence_state.value
            ),
            "previous_result_banner_evidence_age_ms": (
                status.previous_result_banner_evidence_age_ms
            ),
            "active_get_episode": self.collect_retry.episode_open,
            "collect_episode_terminal": self.collect_retry.episode_terminal,
            "cycle_completed": cycle_completed,
            "transition_from": transition_from.value if transition_from else None,
            "transition_to": transition_to.value if transition_to else None,
            "blockers": list(blockers),
            "action_applied": False,
        })

    def _initialize_action_sink(self, *, session_started_at: float) -> None:
        if not self.emit_actions or self.action_sink_name == ACTION_SINK_NONE:
            return
        hwnd = self._capture_diagnostics.get("hwnd")
        title = self._capture_diagnostics.get("window_title")
        process = self._capture_diagnostics.get("process")
        process_id = self._capture_diagnostics.get("process_id")
        client_size = self._capture_diagnostics.get("client_size")
        title_prefix = self._capture_diagnostics.get("window_title_prefix")
        resolution_mode = self._capture_diagnostics.get(
            "window_resolution_mode", "exact_title"
        )
        if not isinstance(hwnd, int) or hwnd <= 0:
            raise LivePreflightError("Action sink requires the exact startup-resolved target HWND")
        if not isinstance(title, str) or not title:
            raise LivePreflightError("Action sink requires the exact startup-resolved window title")
        if not isinstance(process, str) or process.casefold() != EXPECTED_GAME_PROCESS.casefold():
            raise LivePreflightError(
                f"Action sink process must be {EXPECTED_GAME_PROCESS}, got {process!r}"
            )
        if not isinstance(process_id, int) or process_id <= 0:
            raise LivePreflightError("Action sink requires the startup-resolved target PID")
        if tuple(client_size or ()) != EXPECTED_RESOLUTION:
            raise LivePreflightError(
                f"Action sink requires a {EXPECTED_RESOLUTION[0]}x{EXPECTED_RESOLUTION[1]} client"
            )
        try:
            action_values = {
                **dict(self._raw_config.get("action") or {}),
                "press_initial_delay_min_ms": (
                    self.live_config.press_initial_delay_min_ms
                ),
                "press_initial_delay_max_ms": (
                    self.live_config.press_initial_delay_max_ms
                ),
                "press_inter_key_gap_min_ms": (
                    self.live_config.press_inter_key_gap_min_ms
                ),
                "press_inter_key_gap_max_ms": (
                    self.live_config.press_inter_key_gap_max_ms
                ),
                "key_hold_ms": self.live_config.press_key_hold_ms,
            }
            action_config = WindowsActionConfig.from_mapping(
                action_values, panic_key=self.panic_key
            )
            self.action_sink = self.action_sink_factory(
                target_hwnd=hwnd,
                expected_title=title,
                expected_title_prefix=(
                    title_prefix if isinstance(title_prefix, str) else None
                ),
                window_resolution_mode=(
                    resolution_mode
                    if resolution_mode in {"exact_title", "process_name"}
                    else "exact_title"
                ),
                expected_process_id=process_id,
                expected_process_name=EXPECTED_GAME_PROCESS,
                expected_client_size=EXPECTED_RESOLUTION,
                allowlist=self.action_allowlist,
                config=action_config,
                clock=self.clock,
                sleep=self.sleep,
                event_callback=self.logger.event,
                session_started_at=session_started_at,
                enable_live_press_sequence=(
                    self.enable_live_press_sequence
                ),
            )
        except ActionIntegrityPreflightError as exc:
            self._preflight_failure_reason = exc.reason
            self._action_preflight_diagnostics = dict(exc.diagnostics)
            raise LivePreflightError(f"{exc.reason}\n{exc}") from exc
        except Exception as exc:
            raise LivePreflightError(
                f"Action sink initialization failed: {type(exc).__name__}: {exc}"
            ) from exc

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
        roi_config = load_roi_config()
        self._hook_critical_bounds = roi_config.pixel_roi(
            (
                "hook_bar_precise"
                if "hook_bar_precise" in roi_config.rois
                else "hook_bar"
            ),
            width,
            height,
        )
        hook_prompt_bounds = roi_config.pixel_roi(
            (
                "hook_prompt"
                if "hook_prompt" in roi_config.rois
                else "hook_bar"
            ),
            width,
            height,
        )
        self._hook_frame_assembler = HookCriticalFrameAssembler(
            (width, height),
            self._hook_critical_bounds,
            hook_prompt_bounds,
        )
        self._hook_frame_assembler.update_prompt_context(frame)
        if (
            self.evidence_recorder is not None
            or self._press_anomaly_evidence.enabled
        ):
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
        if self.evidence_recorder is not None:
            prepare_video = getattr(self.evidence_recorder, "prepare_video", None)
            if callable(prepare_video):
                try:
                    prepare_video(frame)
                except Exception as exc:
                    reason = (
                        "video_preflight: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    disable = getattr(
                        self.evidence_recorder, "disable", None
                    )
                    if callable(disable):
                        disable(reason)
                    self.logger.event(
                        "diagnostic_evidence_failure",
                        {"timestamp": 0.0, "frame_index": 0, "reason": reason},
                    )
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
        if (
            self.fsm.state == RuntimeState.WAITING
            and self.missed_ready_recovery.active
        ):
            return 1.0 / self.schedule.config.ready_fps
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

    def _write_hook_roi_clip(self) -> dict[str, Any]:
        if (
            self.evidence_recorder is None
            or not self._hook_roi_clip_samples
        ):
            return {
                "hook_roi_clip_path": None,
                "hook_roi_clip_index_path": None,
                "hook_roi_clip_frame_count": 0,
                "hook_roi_clip_fps": 0.0,
            }
        root = self.logger.path / "diagnostic_evidence"
        root.mkdir(parents=True, exist_ok=True)
        video_path = root / "hook_roi_clip.mp4"
        index_path = root / "hook_roi_frames.csv"
        assert self._hook_roi_clip_samples is not None
        first = self._hook_roi_clip_samples[0].pixels
        height, width = first.shape[:2]
        selected: list[HookROIFrame] = []
        last_timestamp: float | None = None
        minimum_interval = 1.0 / 30.0
        for sample in self._hook_roi_clip_samples:
            if (
                last_timestamp is not None
                and sample.timestamp - last_timestamp
                < minimum_interval
            ):
                continue
            selected.append(sample)
            last_timestamp = sample.timestamp
        clip_intervals = [
            right.timestamp - left.timestamp
            for left, right in zip(selected, selected[1:])
            if 0.0 < right.timestamp - left.timestamp <= 0.25
        ]
        sampled_duration = sum(clip_intervals)
        observed_fps = (
            len(clip_intervals) / sampled_duration
            if sampled_duration > 0.0 else 0.0
        )
        output_fps = min(30.0, observed_fps) if observed_fps > 0.0 else 1.0
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            return {
                "hook_roi_clip_path": None,
                "hook_roi_clip_index_path": None,
                "hook_roi_clip_frame_count": 0,
                "hook_roi_clip_fps": output_fps,
                "hook_roi_clip_error": "video_writer_open_failed",
            }
        try:
            for sample in selected:
                writer.write(sample.pixels)
        finally:
            writer.release()
        with index_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = csv.DictWriter(
                handle,
                fieldnames=[
                    "clip_frame_index",
                    "capture_frame_index",
                    "timestamp",
                ],
            )
            rows.writeheader()
            for index, sample in enumerate(selected, start=1):
                rows.writerow({
                    "clip_frame_index": index,
                    "capture_frame_index": sample.frame_index,
                    "timestamp": sample.timestamp,
                })
        return {
            "hook_roi_clip_path": str(video_path),
            "hook_roi_clip_index_path": str(index_path),
            "hook_roi_clip_frame_count": len(selected),
            "hook_roi_clip_fps": output_fps,
        }

    def _start_hook_video_suspension(
        self,
        *,
        timestamp: float,
        frame_index: int,
    ) -> None:
        if self._hook_video_suspension_active is not None:
            return
        self._hook_video_suspension_active = {
            "episode_index": len(self._hook_video_suspensions) + 1,
            "start_timestamp": float(timestamp),
            "start_frame_index": int(frame_index),
        }

    def _finish_hook_video_suspension(
        self,
        *,
        timestamp: float,
        frame_index: int,
    ) -> None:
        active = self._hook_video_suspension_active
        if active is None:
            return
        start = float(active["start_timestamp"])
        end = max(start, float(timestamp))
        samples = [
            item
            for item in (self._hook_roi_clip_samples or ())
            if start - 1e-9 <= item.timestamp <= end + 1e-9
        ]
        first_roi_timestamp = samples[0].timestamp if samples else None
        last_roi_timestamp = samples[-1].timestamp if samples else None
        coverage_tolerance = max(
            0.1,
            2.0 / self.live_config.hook_critical_target_fps,
        )
        self._hook_video_suspensions.append({
            **active,
            "end_timestamp": end,
            "end_frame_index": int(frame_index),
            "duration_seconds": end - start,
            "hook_roi_clip_frame_count": len(samples),
            "hook_roi_clip_first_timestamp": first_roi_timestamp,
            "hook_roi_clip_last_timestamp": last_roi_timestamp,
            "hook_roi_clip_covers_interval": bool(
                samples
                and first_roi_timestamp is not None
                and last_roi_timestamp is not None
                and first_roi_timestamp - start <= coverage_tolerance
                and end - last_roi_timestamp <= coverage_tolerance
            ),
        })
        self._hook_video_suspension_active = None

    def _flush_hook_decision_trace(self, reason: str) -> None:
        if self._hook_decision_trace is None:
            return
        rows = self._hook_decision_trace.drain(reason)
        if not rows or self.evidence_recorder is None:
            return
        path = (
            self.logger.path
            / "diagnostic_evidence"
            / "hook_decision_trace.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        self._hook_decision_trace_path = path
        self._hook_decision_trace_rows_written += len(rows)

    def _finish_hook_episode_telemetry(
        self,
        *,
        timestamp: float,
    ) -> None:
        before = len(self._hook_episode_telemetry.summaries())
        self._hook_episode_telemetry.finish(
            timestamp,
            self._latest_hook_frame.stale_frames_dropped,
        )
        summaries = self._hook_episode_telemetry.summaries()
        if len(summaries) > before:
            self.logger.event("hook_critical_episode_summary", {
                "timestamp": timestamp,
                "runtime_state": self.fsm.state.value,
                **summaries[-1],
            })

    def _hook_recovery_safety_blockers(
        self,
        *,
        foreground: bool | None,
    ) -> tuple[str, ...]:
        blockers: list[str] = []
        sink_summary: Mapping[str, Any] = {}
        if self.action_sink is not None:
            summary = getattr(self.action_sink, "summary", None)
            if callable(summary):
                sink_summary = summary()
        if sink_summary.get("panic_triggered"):
            blockers.append("panic_triggered")
        if foreground is not True:
            blockers.append("foreground_not_confirmed")
        integrity = sink_summary.get(
            "integrity_diagnostics", self._action_preflight_diagnostics
        )
        if isinstance(integrity, Mapping) and integrity.get(
            "suspected_integrity_mismatch"
        ):
            blockers.append("integrity_mismatch")
        if (
            self.action_sink is not None
            and ActionIntent.HOOK_ACTION not in self.action_allowlist
        ):
            blockers.append("action_not_allowlisted")
        return tuple(blockers)

    def _hook_lifecycle_event_payload(
        self,
        *,
        state_age_seconds: float,
        hook_confidence: float | None,
        hook_age_seconds: float | None,
        recovery_reason: str,
    ) -> dict[str, Any]:
        return {
            **self._hook_action_lifecycle.payload(),
            "state_age_seconds": float(state_age_seconds),
            "hook_evidence_confidence": hook_confidence,
            "hook_evidence_age_seconds": hook_age_seconds,
            "recovery_reason": recovery_reason,
        }

    def _panic_latched(self) -> bool:
        if self.action_sink is None:
            return False
        summary = getattr(self.action_sink, "summary", None)
        return bool(
            callable(summary)
            and summary().get("panic_triggered", False)
        )

    def _idle_conflicting_evidence(
        self,
        result: Any,
        *,
        timestamp: float,
    ) -> bool:
        freshness_seconds = (
            self.live_config.idle_recovery_freshness_ms / 1000.0
        )
        if result.fsm.action_request.intent in {
            ActionIntent.HOOK_ACTION,
            ActionIntent.PRESS_SEQUENCE,
            ActionIntent.COLLECT,
        }:
            return True
        bundle = result.qualified.bundle
        for name in ("hook", "press", "get"):
            observation = getattr(bundle, name)
            qualification = getattr(result.qualified, name)
            if (
                observation is not None
                and observation.detected
                and qualification.qualified_detected
                and max(0.0, timestamp - observation.timestamp)
                <= freshness_seconds
            ):
                return True
        banner = bundle.result_banner
        return bool(
            banner is not None
            and banner.detected
            and max(0.0, timestamp - banner.timestamp)
            <= freshness_seconds
        )

    def _idle_certificate_payload(
        self,
        certificate: IdleRecoveryCertificate,
        *,
        timestamp: float,
    ) -> dict[str, Any]:
        return {
            "prompt_certificate": certificate.payload(),
            "prompt_certificate_summary": (
                self._idle_recovery.summary(timestamp)
            ),
        }

    def _request_idle_cast_arm(
        self,
        *,
        source_type: str,
        source_id: str,
        certificate: IdleRecoveryCertificate,
        timestamp: float,
        frame_index: int,
        cooldown_seconds: float,
        previous_state: RuntimeState,
    ) -> None:
        record, events = self._cast_arming.request_cast_opportunity(
            source_type=source_type,
            source_id=source_id,
            physical_idle_id=(
                self._cast_arming.recovery_physical_idle_id(
                    certificate.physical_idle_id
                )
                if source_type == CAST_SOURCE_RECOVERY
                else certificate.physical_idle_id
            ),
            cycle_id=self.deduplicator.cycle_id,
            timestamp=timestamp,
            cooldown_seconds=cooldown_seconds,
        )
        for event in events:
            self.logger.event(event.event_type, {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "previous_state": previous_state.value,
                **self._idle_certificate_payload(
                    certificate, timestamp=timestamp
                ),
                **dict(event.payload),
            })
        if record is not None and cooldown_seconds > 0:
            self.logger.event(
                "idle_recovery_cast_cooldown_started",
                {
                    "timestamp": timestamp,
                    "frame_index": frame_index,
                    "previous_state": previous_state.value,
                    **record.payload(timestamp),
                    **self._idle_certificate_payload(
                        certificate, timestamp=timestamp
                    ),
                },
            )
        if record is not None:
            event_type = (
                "startup_idle_cast_armed"
                if source_type == CAST_SOURCE_STARTUP
                else "idle_recovery_cast_rearmed"
            )
            self.logger.event(event_type, {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "previous_state": previous_state.value,
                **record.payload(timestamp),
                **self._idle_certificate_payload(
                    certificate, timestamp=timestamp
                ),
            })

    def _service_idle_liveness_arm(
        self,
        *,
        certificate: IdleRecoveryCertificate | None,
        timestamp: float,
        frame_index: int,
    ) -> None:
        """Bound a source-less physical IDLE and arm it exactly once."""
        active = self._cast_arming.active
        if (
            active is not None
            and active.source_type
            in {CAST_SOURCE_STARTUP, CAST_SOURCE_RECOVERY}
            and certificate is not None
            and active.physical_idle_id != certificate.physical_idle_id
        ):
            superseded = self._cast_arming.supersede(
                "new_physical_idle_certificate"
            )
            if superseded is not None:
                self.logger.event("cast_arming_superseded", {
                    "timestamp": timestamp,
                    "frame_index": frame_index,
                    "runtime_state": self.fsm.state.value,
                    **superseded.payload(timestamp),
                })
            active = self._cast_arming.active

        if (
            certificate is None
            or self.fsm.state != RuntimeState.IDLE
            or self.cast_opportunity.waiting_for_acknowledgement
            or self._action_emission_in_progress
            or self._panic_latched()
        ):
            self._idle_liveness_started_at = None
            self._idle_liveness_physical_idle_id = None
            return

        physical_idle_id = certificate.physical_idle_id
        if active is not None:
            self._idle_liveness_started_at = None
            self._idle_liveness_physical_idle_id = physical_idle_id
            return
        if physical_idle_id in self._idle_liveness_armed_physical_ids:
            return
        if self._idle_liveness_physical_idle_id != physical_idle_id:
            self._idle_liveness_physical_idle_id = physical_idle_id
            self._idle_liveness_started_at = float(timestamp)
            return
        if self._idle_liveness_started_at is None:
            self._idle_liveness_started_at = float(timestamp)
            return
        if (
            float(timestamp) - self._idle_liveness_started_at
            < self.live_config.idle_cast_liveness_timeout_seconds
        ):
            return

        source_id = f"idle_liveness:{physical_idle_id}"
        self._idle_liveness_armed_physical_ids.add(physical_idle_id)
        self._request_idle_cast_arm(
            source_type=CAST_SOURCE_RECOVERY,
            source_id=source_id,
            certificate=certificate,
            timestamp=timestamp,
            frame_index=frame_index,
            cooldown_seconds=0.0,
            previous_state=RuntimeState.IDLE,
        )

    def _clear_stale_cycle_for_idle_recovery(
        self,
        *,
        timestamp: float,
        frame_index: int,
    ) -> None:
        cancelled_cast_arm = self._cast_arming.cancel(
            "authoritative_idle_prompt_recovery"
        )
        if cancelled_cast_arm is not None:
            self.logger.event("cast_arming_cancelled", {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "runtime_state": self.fsm.state.value,
                **cancelled_cast_arm.payload(timestamp),
            })
        self._hook_action_lifecycle.finish_episode()
        for event in self._press_live_emission.cancel_pending(
            timestamp=timestamp,
            reason="authoritative_idle_prompt_recovery",
        ):
            self.logger.event(event.event_type, {
                "timestamp": timestamp,
                "frame_index": frame_index,
                "runtime_state": self.fsm.state.value,
                **dict(event.payload),
            })
        self._press_shadow.cancel_for_authoritative_idle_recovery()
        self._log_collect_events(
            self.collect_retry.terminate_for_authoritative_idle_recovery(
                timestamp
            ),
            timestamp=timestamp,
            frame_index=frame_index,
            runtime_state=self.fsm.state.value,
        )
        self.cast_clearance.reset_for_authoritative_idle_recovery()
        self._log_cast_events(
            self.cast_opportunity.cancel(
                timestamp=timestamp,
                reason="authoritative_idle_prompt_recovery",
            ),
            timestamp=timestamp,
            frame_index=frame_index,
            runtime_state=self.fsm.state.value,
        )
        self.schedule.reset()
        self.missed_ready_recovery.reset()
        self.recovery_synchronizer.reset(started_at=timestamp)
        self.controller.reset_action_history_for_authoritative_idle()
        self.deduplicator.begin_recovered_cycle()

    def _overlay_lines(
        self,
        *,
        capture_fps: float,
        latency_ms: float,
        prompt: PromptObservation | None,
        result: Any | None,
        activation: DetectorActivationSnapshot,
        actions_applied: int,
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
            f"Get panel={bool(get and get.detected)} | would-fire={request.intent.value} | action_applied={actions_applied > 0}",
            f"Result banner={bool(result_banner and result_banner.detected)} hold-only=true",
        ]

    def run(self, *, max_frames: int | None = None) -> dict[str, Any]:
        captured = processed = actions_applied = 0
        completed_cycles = 0
        missed_ready_recovery_count = 0
        evidence_episode_id = 1
        stop_after_completed_cycle = False
        stop_after_action_commit_failure = False
        evidence_failure_reason: str | None = None
        latencies: list[float] = []
        detector_runs: Counter[str] = Counter()
        result_name = "completed"
        shutdown_reason = "duration_limit"
        started_at_utc = datetime.now(timezone.utc).isoformat()
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
        last_production_heartbeat = 0.0
        started = self.clock()
        next_prompt_due = 0.0
        next_detector_due = 0.0
        next_result_banner_due = 0.0
        initial_bundle = ObservationBundle(0, 0.0)
        activation = self.activation_policy.evaluate(
            RuntimeState.SYNCING, initial_bundle, recorded_observation=True
        )
        try:
            if self.live_config.runtime_profile == "production":
                self.console.emit("startup: production fishing runtime")
            self.logger.event("startup", {
                "timestamp": 0.0,
                "runtime_profile": self.live_config.runtime_profile,
                "duration_seconds": self.live_config.duration_seconds,
                "max_completed_cycles": self.live_config.max_completed_cycles,
            })
            self.preflight()
            self._initialize_action_sink(session_started_at=started)
            self._preflight_passed = True
            if self.live_config.runtime_profile == "production":
                self.console.emit("preflight: passed")
            self.logger.event("preflight_passed", {
                "timestamp": 0.0,
                "resolution": list(EXPECTED_RESOLUTION),
                "approved_roi": list(self.prompt_bundle.roi.pixel),
                "bundle_sha256": self.prompt_bundle.bundle_sha256,
                "emit_actions": self.emit_actions,
                "enable_live_press_sequence": (
                    self.enable_live_press_sequence
                ),
                "action_sink": self.action_sink_name,
                "action_allowlist": sorted(item.value for item in self.action_allowlist),
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
            while (
                self.live_config.duration_seconds == 0
                or self.clock() - started < self.live_config.duration_seconds
            ):
                if max_frames is not None and captured >= max_frames:
                    break
                frame_loop_started = self.clock()
                elapsed = frame_loop_started - started
                hook_critical_mode = bool(
                    self.fsm.state in {
                        RuntimeState.HOOK_PENDING,
                        RuntimeState.HOOK,
                    }
                    and self._hook_critical_bounds is not None
                    and self._hook_frame_assembler is not None
                    and bool(getattr(
                        self.capture,
                        "supports_native_roi_capture",
                        False,
                    ))
                    and callable(getattr(self.capture, "capture_roi", None))
                )
                if hook_critical_mode:
                    if not self._hook_episode_telemetry.active:
                        self.console.emit(
                            "HOOK critical active fps=0.0"
                        )
                    if (
                        self.evidence_recorder is not None
                        and self._hook_video_suspension_active is None
                    ):
                        self._start_hook_video_suspension(
                            timestamp=elapsed,
                            frame_index=captured + 1,
                        )
                    self._hook_episode_telemetry.start(
                        elapsed,
                        self._latest_hook_frame.stale_frames_dropped,
                    )
                    if self._hook_decision_trace is not None:
                        self._hook_decision_trace.start(
                            str(self.deduplicator.cycle_id)
                        )
                elif self._hook_video_suspension_active is not None:
                    self._finish_hook_video_suspension(
                        timestamp=elapsed,
                        frame_index=captured,
                    )
                try:
                    if hook_critical_mode:
                        hook_pixels = validate_bgr_frame(
                            self.capture.capture_roi(
                                self._hook_critical_bounds
                            )
                        )
                        hook_timestamp = self.clock() - started
                        self._latest_hook_frame.publish(HookROIFrame(
                            captured + 1,
                            hook_timestamp,
                            hook_pixels,
                        ))
                        latest_hook_frame = (
                            self._latest_hook_frame.take_latest()
                        )
                        if latest_hook_frame is None:
                            continue
                        frame = self._hook_frame_assembler.compose(
                            latest_hook_frame.pixels
                        )
                        elapsed = latest_hook_frame.timestamp
                        if self.evidence_recorder is not None:
                            assert self._hook_roi_clip_samples is not None
                            self._hook_roi_clip_samples.append(HookROIFrame(
                                latest_hook_frame.frame_index,
                                latest_hook_frame.timestamp,
                                latest_hook_frame.pixels.copy(),
                            ))
                    else:
                        frame = validate_bgr_frame(
                            self.capture.capture()
                        )
                        if self._hook_frame_assembler is not None:
                            self._hook_frame_assembler.update_prompt_context(
                                frame
                            )
                except KeyboardInterrupt:
                    result_name = "interrupted_by_user"
                    shutdown_reason = "ctrl_c"
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
                frame_captured_at = self.clock() - started
                panic_triggered_this_frame = False
                if self.action_sink is not None:
                    poll_panic = getattr(self.action_sink, "poll_panic", None)
                    if callable(poll_panic):
                        panic_triggered_this_frame = bool(poll_panic())
                if (
                    panic_triggered_this_frame
                    and self.live_config.runtime_profile == "production"
                ):
                    for press_event in self._press_live_emission.cancel_pending(
                        timestamp=elapsed,
                        reason="panic_triggered",
                    ):
                        self.logger.event(press_event.event_type, {
                            **dict(press_event.payload),
                            "frame_index": captured,
                            "runtime_state": self.fsm.state.value,
                        })
                    self.logger.event("panic_stop", {
                        "timestamp": elapsed,
                        "frame_index": captured,
                        "runtime_state": self.fsm.state.value,
                        "reason": "panic_key_pressed",
                        "action_applied": False,
                    })
                    self.console.emit(
                        "PANIC: input disabled, shutting down"
                    )
                    result_name = "panic_shutdown"
                    shutdown_reason = "panic_key"
                    break
                height, width = frame.shape[:2]
                if (width, height) != EXPECTED_RESOLUTION:
                    result_name = "safe_stop_resolution_changed"
                    self.logger.event("capture_failure", {
                        "timestamp": elapsed,
                        "frame_index": captured,
                        "reason": f"resolution_changed_to_{width}x{height}",
                    })
                    break
                defer_video_for_hook_fast_path = bool(
                    hook_critical_mode
                    or (
                        self.evidence_recorder is not None
                        and self._press_live_emission.pending is not None
                    )
                    or (
                    self.evidence_recorder is not None
                    and self.action_sink is not None
                    and ActionIntent.HOOK_ACTION in self.action_allowlist
                    and self.fsm.state in {
                        RuntimeState.HOOK_PENDING,
                        RuntimeState.HOOK,
                    }
                    )
                )
                if (
                    self.evidence_recorder is not None
                    and not defer_video_for_hook_fast_path
                ):
                    try:
                        self.evidence_recorder.record_frame(
                            frame, capture_frame_index=captured, timestamp=elapsed
                        )
                    except Exception as exc:
                        evidence_failure_reason = f"video: {type(exc).__name__}: {exc}"
                        disable = getattr(
                            self.evidence_recorder, "disable", None
                        )
                        if callable(disable):
                            disable(evidence_failure_reason)
                        self.logger.event("diagnostic_evidence_failure", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "reason": evidence_failure_reason,
                        })
                prompt_due = (
                    not hook_critical_mode
                    and elapsed >= next_prompt_due
                )
                detector_interval = self._detector_interval(activation)
                detector_due = bool(
                    hook_critical_mode
                    or (
                        detector_interval is not None
                        and elapsed >= next_detector_due
                    )
                )
                post_collect_banner_pending = (
                    self.cast_clearance.post_collect_confirmation_required
                )
                post_collect_banner_due = bool(
                    not hook_critical_mode
                    and post_collect_banner_pending
                    and elapsed >= next_result_banner_due
                )
                should_process = (
                    prompt_due or detector_due or post_collect_banner_due
                )
                deferred_video_recorded = False
                if should_process:
                    processing_started = self.clock()
                    context = FrameContext(captured, elapsed, metadata={"source": "live_detect_only"})
                    ready_burst_update = None
                    missed_ready_update = None
                    if prompt_due:
                        last_prompt = self.prompt_bundle.observer.observe(frame, context)
                        ready_burst_update = self.schedule.observe_prompt(
                            self.fsm.state, last_prompt.kind, elapsed
                        )
                        missed_ready_update = self.missed_ready_recovery.observe(
                            self.fsm.state,
                            last_prompt,
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
                    press = (
                        self.press_detector.observe(frame, context)
                        if (
                            not hook_critical_mode
                            and run_detectors
                            and activation.press
                            != DetectorActivationMode.OFF
                        )
                        else None
                    )
                    if (
                        self._press_v3_shadow is not None
                        and not hook_critical_mode
                        and run_detectors
                        and activation.press != DetectorActivationMode.OFF
                    ):
                        for event_type, payload in self._press_v3_shadow.observe(
                            frame, context, press
                        ):
                            self.logger.event(event_type, {
                                "timestamp": elapsed,
                                **payload,
                            })
                    run_get = bool(
                        not hook_critical_mode
                        and run_detectors
                        and activation.get
                        != DetectorActivationMode.OFF
                    )
                    get = self.get_detector.observe(frame, context) if run_get else None
                    get_detector_executed = get is not None
                    if get is None and self.fsm.state == RuntimeState.IDLE:
                        get = self.cast_clearance.certified_get_absence(
                            timestamp=elapsed,
                            frame_index=captured,
                        )
                    run_result_banner = bool(
                        (
                            not hook_critical_mode
                            and run_detectors
                            and self.fsm.state == RuntimeState.RESULT_PENDING
                        )
                        or post_collect_banner_due
                    )
                    result_banner = (
                        self.result_banner_observer.observe(frame, context)
                        if run_result_banner else None
                    )
                    active_idle_arm = self._cast_arming.active
                    raw_idle_conflict = bool(
                        (hook is not None and hook.detected)
                        or (press is not None and press.detected)
                        or (get is not None and get.detected)
                        or (
                            result_banner is not None
                            and result_banner.detected
                        )
                    )
                    certified_idle = self._idle_recovery.current(
                        timestamp=elapsed,
                        conflicting_evidence=raw_idle_conflict,
                        action_emission_in_progress=(
                            self._action_emission_in_progress
                        ),
                        key_currently_down=False,
                        panic_triggered=self._panic_latched(),
                    )
                    if (
                        get is None
                        and self.fsm.state == RuntimeState.IDLE
                        and certified_idle is not None
                        and active_idle_arm is not None
                        and active_idle_arm.source_type
                        in {CAST_SOURCE_STARTUP, CAST_SOURCE_RECOVERY}
                    ):
                        # This explicit absence is certified by stable, fresh
                        # IDLE_CAST observations. It is never inferred merely
                        # because the GET detector is OFF.
                        get = GetObservation(
                            detected=False,
                            confidence=1.0,
                            frame_index=captured,
                            timestamp=elapsed,
                            source="idle_recovery_certificate",
                            evidence={
                                "certified_absence": True,
                                "certificate_id": (
                                    certified_idle.certificate_id
                                ),
                                "physical_idle_id": (
                                    certified_idle.physical_idle_id
                                ),
                            },
                        )
                    if post_collect_banner_due:
                        next_result_banner_due = (
                            elapsed + 1.0 / self._post_collect_banner_fps
                        )
                    elif not post_collect_banner_pending:
                        next_result_banner_due = elapsed
                    detector_runs.update({
                        "hook": int(hook is not None),
                        "press": int(press is not None),
                        "get": int(get_detector_executed),
                        "result_banner": int(result_banner is not None),
                    })
                    if hook_critical_mode and hook is not None:
                        self._hook_episode_telemetry.record_detector_frame(
                            elapsed
                        )
                        self.console.heartbeat(
                            "hook_critical",
                            "HOOK critical active fps="
                            f"{self._hook_episode_telemetry.current_actual_fps():.1f}",
                            timestamp=elapsed,
                            minimum_interval_seconds=0.5,
                        )
                    raw_bundle = ObservationBundle(
                        captured, elapsed, prompt, hook, press, get, result_banner
                    )
                    sync_reason = None
                    processing_from_sync_required = self.fsm.state == RuntimeState.SYNC_REQUIRED
                    transition_results: list[Any] = []
                    authoritative_idle_applied_this_frame = False
                    if (
                        missed_ready_update is not None
                        and missed_ready_update.recovered
                    ):
                        recovered_cycle_id = (
                            self.deduplicator.begin_recovered_cycle()
                        )
                        self._hook_action_lifecycle.begin_episode(
                            cycle_id=recovered_cycle_id,
                            timestamp=elapsed,
                            start_hook_applied=False,
                        )
                        recovered = self.fsm.recover_missed_ready(
                            elapsed,
                            reason=missed_ready_update.reason,
                        )
                        self._runtime_cycle_started = True
                        transition_results.append(recovered)
                        missed_ready_recovery_count += 1
                        next_prompt_due = min(
                            next_prompt_due,
                            elapsed + self._prompt_interval(),
                        )
                        self.logger.event("missed_ready_recovered", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "cycle_id": recovered_cycle_id,
                            "previous_state": recovered.previous_state.value,
                            "next_state": recovered.next_state.value,
                            "reason": missed_ready_update.reason,
                            "support_frames": (
                                missed_ready_update.support_frames
                            ),
                            "duration_seconds": (
                                missed_ready_update.candidate_age_seconds
                            ),
                            "confidence": (
                                prompt.confidence if prompt is not None else None
                            ),
                            "action_intent": ActionIntent.NONE.value,
                            "action_applied": False,
                        })
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
                            if sync.state != RuntimeState.IDLE:
                                self._runtime_cycle_started = True
                        elif sync.state == RuntimeState.SYNC_REQUIRED:
                            transition_results.append(self.fsm.force_state(
                                RuntimeState.SYNC_REQUIRED, elapsed, sync.reason
                            ))
                    foreground = self.capture.is_foreground()
                    foreground_diagnostics = getattr(self.capture, "diagnostics", None)
                    current_capture_diagnostics = (
                        dict(foreground_diagnostics())
                        if callable(foreground_diagnostics) else {}
                    )
                    self._capture_diagnostics.update(current_capture_diagnostics)
                    foreground_unavailable = bool(
                        current_capture_diagnostics.get(
                            "foreground_window_unavailable", False
                        )
                    )
                    if foreground_unavailable and not self._foreground_unavailable_event_active:
                        self.logger.event("foreground_window_unavailable", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "runtime_state": self.fsm.state.value,
                            "source": "capture_diagnostics",
                            "foreground": False,
                            "action_applied": False,
                        })
                    self._foreground_unavailable_event_active = foreground_unavailable

                    cast_tracking_enabled = bool(
                        self.action_sink is not None
                        and ActionIntent.CAST in self.action_allowlist
                    )
                    if cast_tracking_enabled:
                        cast_visual_events = self.cast_opportunity.observe(
                            timestamp=elapsed,
                            runtime_state=self.fsm.state,
                            prompt_kind=(prompt.kind if prompt is not None else None),
                            prompt_frame_index=(
                                prompt.frame_index if prompt is not None else None
                            ),
                            prompt_timestamp=(
                                prompt.timestamp if prompt is not None else None
                            ),
                        )
                        acknowledged = any(
                            item.event_type == "cast_visual_acknowledged"
                            for item in cast_visual_events
                        )
                        timed_out = any(
                            item.event_type == "cast_visual_timeout"
                            for item in cast_visual_events
                        )
                        if acknowledged:
                            if self.fsm.state == RuntimeState.SYNC_REQUIRED:
                                transition_results.append(
                                    self.fsm.recover_from_sync_required(
                                        RuntimeState.WAITING,
                                        elapsed,
                                        "cast_visual_acknowledged",
                                    )
                                )
                            elif self.fsm.state == RuntimeState.CAST_PENDING:
                                transition_results.append(self.fsm.force_state(
                                    RuntimeState.WAITING,
                                    elapsed,
                                    "cast_visual_acknowledged",
                                ))
                        elif (
                            timed_out
                            and self.fsm.state != RuntimeState.SYNC_REQUIRED
                        ):
                            transition_results.append(self.fsm.force_state(
                                RuntimeState.SYNC_REQUIRED,
                                elapsed,
                                "cast_visual_timeout",
                            ))
                        self._log_cast_events(
                            cast_visual_events,
                            timestamp=elapsed,
                            frame_index=captured,
                            runtime_state=self.fsm.state.value,
                        )

                    runtime_state_before_controller = self.fsm.state
                    last_result = self.controller.process(
                        raw_bundle,
                        foreground=foreground,
                        runtime_environment_supported=True,
                        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
                        preserve_proposal=self.action_sink is not None,
                    )
                    if last_result.action_applied:
                        raise RuntimeError(
                            "Recorded-observation controller invariant violated: action_applied=true"
                        )
                        if timed_out:
                            self._cast_arming.authorize_retry_after_visual_timeout()
                    processed += 1
                    request = last_result.fsm.action_request
                    if (
                        self.fsm.state == RuntimeState.HOOK
                        and self._hook_action_lifecycle.episode is None
                    ):
                        self._hook_action_lifecycle.begin_episode(
                            cycle_id=self.deduplicator.cycle_id,
                            timestamp=self.fsm.state_since,
                            start_hook_applied=False,
                        )
                    current_qualified_hook = (
                        last_result.qualified.bundle.hook
                    )
                    current_hook_age_seconds = (
                        max(
                            0.0,
                            elapsed - current_qualified_hook.timestamp,
                        )
                        if current_qualified_hook is not None else None
                    )
                    hook_stall = None
                    if request.intent != ActionIntent.HOOK_ACTION:
                        hook_stall = (
                            self._hook_action_lifecycle.evaluate_stall(
                                runtime_state=self.fsm.state,
                                state_age_seconds=max(
                                    0.0,
                                    elapsed - self.fsm.state_since,
                                ),
                                qualified_hook_current=bool(
                                    current_qualified_hook is not None
                                    and current_qualified_hook.detected
                                    and last_result.qualified.hook
                                    .qualified_detected
                                    and current_hook_age_seconds is not None
                                    and current_hook_age_seconds
                                    <= max(
                                        0.25,
                                        2.0
                                        / self.live_config
                                        .hook_critical_target_fps,
                                    )
                                ),
                                hook_evidence_confidence=(
                                    current_qualified_hook.confidence
                                    if current_qualified_hook is not None
                                    else None
                                ),
                                hook_evidence_age_seconds=(
                                    current_hook_age_seconds
                                ),
                                safety_blockers=(
                                    self._hook_recovery_safety_blockers(
                                        foreground=foreground
                                    )
                                ),
                            )
                        )
                    if (
                        hook_stall is not None
                        and hook_stall.action != "blocked"
                    ):
                        stall_payload = {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            **self._hook_lifecycle_event_payload(
                                state_age_seconds=(
                                    hook_stall.state_age_seconds
                                ),
                                hook_confidence=(
                                    hook_stall.hook_evidence_confidence
                                ),
                                hook_age_seconds=(
                                    hook_stall.hook_evidence_age_seconds
                                ),
                                recovery_reason=hook_stall.reason,
                            ),
                        }
                        self.logger.event(
                            "hook_action_stall_watchdog_triggered",
                            stall_payload,
                        )
                        if hook_stall.action == "rearm":
                            if self.fsm.rearm_hook_action_opportunity(
                                elapsed
                            ):
                                self.deduplicator.release(
                                    ActionIntent.HOOK_ACTION
                                )
                                self.logger.event(
                                    "hook_action_rearmed_by_watchdog",
                                    stall_payload,
                                )
                        elif hook_stall.action == "sync_required":
                            stalled = (
                                self.fsm
                                .return_hook_stall_to_sync_required(
                                    elapsed
                                )
                            )
                            transition_results.append(stalled)
                            self.logger.event(
                                "hook_stall_returned_to_sync_required",
                                stall_payload,
                            )
                    idle_conflict = self._idle_conflicting_evidence(
                        last_result,
                        timestamp=elapsed,
                    )
                    panic_latched = bool(
                        panic_triggered_this_frame
                        or self._panic_latched()
                    )
                    idle_certificate, idle_events = (
                        self._idle_recovery.observe(
                            prompt,
                            timestamp=elapsed,
                            conflicting_evidence=idle_conflict,
                            action_emission_in_progress=(
                                self._action_emission_in_progress
                            ),
                            key_currently_down=False,
                            panic_triggered=panic_latched,
                        )
                    )
                    for idle_event in idle_events:
                        if idle_event.event_type == "idle_recovery_candidate_started":
                            self._idle_candidate_origin_state = self.fsm.state
                        elif (
                            idle_event.event_type
                            == "idle_recovery_candidate_cancelled"
                        ):
                            self._idle_candidate_origin_state = None
                        self.logger.event(idle_event.event_type, {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "runtime_state": self.fsm.state.value,
                            **dict(idle_event.payload),
                            "prompt_certificate_summary": (
                                self._idle_recovery.summary(elapsed)
                            ),
                        })
                    if self._idle_recovery.candidate_active:
                        observation_interval = max(
                            0.02,
                            min(
                                self.live_config
                                .idle_recovery_freshness_ms
                                / 2000.0,
                                self.live_config
                                .idle_recovery_min_window_seconds
                                / max(
                                    1,
                                    self.live_config
                                    .idle_recovery_window_size - 1,
                                ),
                            ),
                        )
                        next_prompt_due = min(
                            next_prompt_due,
                            elapsed + observation_interval,
                        )

                    cast_pending_protected = bool(
                        self.fsm.state == RuntimeState.CAST_PENDING
                        and self.cast_opportunity
                        .waiting_for_acknowledgement
                        and self.cast_opportunity.visual_ack_deadline
                        is not None
                        and elapsed
                        < self.cast_opportunity.visual_ack_deadline
                    )
                    startup_idle = bool(
                        idle_certificate is not None
                        and not self._startup_idle_handled
                        and not self._runtime_cycle_started
                        and self.fsm.state
                        in {RuntimeState.SYNCING, RuntimeState.IDLE}
                    )
                    idle_arrived_before_certificate = bool(
                        idle_certificate is not None
                        and self.fsm.state == RuntimeState.IDLE
                        and self._runtime_cycle_started
                        and self._idle_candidate_origin_state is not None
                        and self._idle_candidate_origin_state
                        != RuntimeState.IDLE
                        and self._cast_arming.active is None
                        and self.cast_clearance.current(elapsed) is None
                    )
                    if startup_idle:
                        previous_idle_state = self.fsm.state
                        if self.fsm.state != RuntimeState.IDLE:
                            transition_results.append(
                                self.fsm.force_state(
                                    RuntimeState.IDLE,
                                    elapsed,
                                    "startup_idle_confirmation",
                                )
                            )
                            self.controller.reset_action_history_for_authoritative_idle()
                            self.synchronizer.reset(started_at=elapsed)
                        self._startup_idle_handled = True
                        self._idle_candidate_origin_state = None
                        self.logger.event("startup_idle_confirmed", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "previous_state": previous_idle_state.value,
                            "cycle_id": self.deduplicator.cycle_id,
                            **self._idle_certificate_payload(
                                idle_certificate,
                                timestamp=elapsed,
                            ),
                        })
                        self._request_idle_cast_arm(
                            source_type=CAST_SOURCE_STARTUP,
                            source_id=(
                                "startup:"
                                + idle_certificate.certificate_id
                            ),
                            certificate=idle_certificate,
                            timestamp=elapsed,
                            frame_index=captured,
                            cooldown_seconds=(
                                self.live_config
                                .idle_recovery_cast_cooldown_seconds
                            ),
                            previous_state=previous_idle_state,
                        )
                    elif (
                        idle_certificate is not None
                        and (
                            self.fsm.state != RuntimeState.IDLE
                            or idle_arrived_before_certificate
                        )
                        and not cast_pending_protected
                        and not self._action_emission_in_progress
                        and not panic_latched
                    ):
                        previous_idle_state = (
                            self._idle_candidate_origin_state
                            if idle_arrived_before_certificate
                            and self._idle_candidate_origin_state is not None
                            else self.fsm.state
                        )
                        self._clear_stale_cycle_for_idle_recovery(
                            timestamp=elapsed,
                            frame_index=captured,
                        )
                        authoritative = self.fsm.force_state(
                            RuntimeState.IDLE,
                            elapsed,
                            "authoritative_idle_prompt_recovery",
                        )
                        transition_results.append(authoritative)
                        authoritative_idle_applied_this_frame = True
                        self.controller.discard_external_proposal()
                        request = ActionRequest(
                            ActionIntent.NONE,
                            0.0,
                            "authoritative_idle_recovery_frame_blocks_actions",
                        )
                        self._idle_candidate_origin_state = None
                        self._idle_recovery_sequence += 1
                        recovery_source_id = (
                            f"idle_recovery:{idle_certificate.certificate_id}:"
                            f"{self._idle_recovery_sequence}"
                        )
                        self.logger.event(
                            "authoritative_idle_recovery_applied",
                            {
                                "timestamp": elapsed,
                                "frame_index": captured,
                                "previous_state": (
                                    previous_idle_state.value
                                ),
                                "next_state": RuntimeState.IDLE.value,
                                "reason": (
                                    "authoritative_idle_prompt_recovery"
                                ),
                                "cycle_id": self.deduplicator.cycle_id,
                                **self._idle_certificate_payload(
                                    idle_certificate,
                                    timestamp=elapsed,
                                ),
                            },
                        )
                        self._request_idle_cast_arm(
                            source_type=CAST_SOURCE_RECOVERY,
                            source_id=recovery_source_id,
                            certificate=idle_certificate,
                            timestamp=elapsed,
                            frame_index=captured,
                            cooldown_seconds=(
                                self.live_config
                                .idle_recovery_cast_cooldown_seconds
                            ),
                            previous_state=previous_idle_state,
                        )
                    active_cast_arm = self._cast_arming.active
                    if (
                        active_cast_arm is not None
                        and active_cast_arm.source_type
                        in {
                            CAST_SOURCE_STARTUP,
                            CAST_SOURCE_RECOVERY,
                        }
                        and idle_certificate is None
                    ):
                        cancelled_arm = self._cast_arming.cancel(
                            "idle_certificate_no_longer_valid"
                        )
                        if cancelled_arm is not None:
                            self.logger.event(
                                "idle_recovery_candidate_cancelled",
                                {
                                    "timestamp": elapsed,
                                    "frame_index": captured,
                                    "runtime_state": self.fsm.state.value,
                                    "reason": (
                                        "idle_certificate_no_longer_valid"
                                    ),
                                    **cancelled_arm.payload(elapsed),
                                },
                            )
                    qualified_press_for_shadow = (
                        last_result.qualified.bundle.press
                    )
                    press_roi_pixels = None
                    if (
                        (
                            self.evidence_recorder is not None
                            or self._press_anomaly_evidence.enabled
                        )
                        and not hook_critical_mode
                    ):
                        x1, y1, x2, y2 = (
                            self._diagnostic_roi_bounds["press"]
                        )
                        press_roi_pixels = frame[y1:y2, x1:x2]
                    press_completeness = (
                        qualified_press_for_shadow.evidence.get(
                            "press_completeness_certificate"
                        )
                        if qualified_press_for_shadow is not None else None
                    )
                    if (
                        isinstance(press_completeness, Mapping)
                        and not bool(press_completeness.get("complete"))
                        and self.fsm.state == RuntimeState.PRESS
                        and elapsed - self._last_press_completeness_log_at >= 1.0
                    ):
                        self._last_press_completeness_log_at = elapsed
                        self.logger.event("press_completeness_pending", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "runtime_state": self.fsm.state.value,
                            **dict(press_completeness),
                        })
                    press_shadow_proposal = self._press_shadow.observe(
                        timestamp=elapsed,
                        frame_index=captured,
                        fsm_state=self.fsm.state,
                        activation_mode=activation.press,
                        raw=press,
                        qualified=qualified_press_for_shadow,
                        qualification_reason=(
                            last_result.qualified.press
                            .sequence_qualification_reason
                            or last_result.qualified.press
                            .qualification_reason
                        ),
                        safety_reason=last_result.safety.reason,
                        roi_pixels=None,
                        live_emission_enabled=(
                            self.enable_live_press_sequence
                        ),
                    )
                    self._press_shadow.capture_roi_frame(
                        timestamp=elapsed,
                        frame_index=captured,
                        fsm_state=self.fsm.state,
                        activation_mode=activation.press,
                        roi_pixels=press_roi_pixels,
                        panel_disappeared=bool(
                            qualified_press_for_shadow
                            and qualified_press_for_shadow.evidence.get(
                                "panel_disappeared"
                            ) is True
                        ),
                    )
                    self._press_anomaly_evidence.record(
                        episode_index=max(1, self._press_shadow.episode_index),
                        frame_index=captured,
                        timestamp=elapsed,
                        roi_pixels=press_roi_pixels,
                        observation=press,
                        certificate=(
                            press_completeness
                            if isinstance(press_completeness, Mapping)
                            else None
                        ),
                    )
                    press_candidate_request = (
                        ActionRequest(
                            ActionIntent.PRESS_SEQUENCE,
                            qualified_press_for_shadow.confidence,
                            "press_sequence_shadow_verified",
                            payload={
                                "sequence": (
                                    press_shadow_proposal.sequence
                                ),
                                "shadow_only": (
                                    not self.enable_live_press_sequence
                                ),
                                "slot_capacity": (
                                    press_shadow_proposal
                                    .slot_capacity
                                ),
                                "active_press_episode": True,
                                "panel_confirmed": True,
                                "frozen_by_consensus": True,
                                "episode_index": (
                                    press_shadow_proposal
                                    .episode_index
                                ),
                                "stability_frame_count": (
                                    press_shadow_proposal
                                    .stability_frame_count
                                ),
                            },
                        )
                        if (
                            press_shadow_proposal is not None
                            and qualified_press_for_shadow is not None
                        )
                        else None
                    )
                    press_shadow_request = (
                        press_candidate_request
                        if not self.enable_live_press_sequence else None
                    )
                    press_fast_would_fire: dict[str, Any] | None = None
                    press_fast_execution: ActionExecutionResult | None = None
                    press_fast_commit = None
                    press_fast_apply_called = False
                    press_fast_safety_reason: str | None = None
                    press_fast_events = ()
                    if (
                        press_candidate_request is not None
                        and self.enable_live_press_sequence
                    ):
                        scheduled, schedule_events = (
                            self._press_live_emission.schedule(
                                episode_index=(
                                    press_shadow_proposal.episode_index
                                ),
                                timestamp=elapsed,
                                sequence=press_shadow_proposal.sequence,
                                slot_capacity=(
                                    press_shadow_proposal.slot_capacity
                                ),
                                frozen_evidence=last_result.evidence,
                            )
                        )
                        if scheduled is not None:
                            self._press_seen_cycles.add(
                                self.deduplicator.cycle_id
                            )
                            self._press_shadow.record_schedule(
                                deadline=scheduled.deadline,
                                timing=scheduled.timing.payload(),
                            )
                            sequence_text = " ".join(scheduled.sequence)
                            self.logger.event("press_sequence_frozen", {
                                "timestamp": elapsed,
                                "frame_index": captured,
                                "sequence": list(scheduled.sequence),
                                "slot_capacity": scheduled.slot_capacity,
                                "selected_clean_frame": (
                                    qualified_press_for_shadow.evidence.get(
                                        "selected_clean_frame"
                                    )
                                ),
                                **(
                                    dict(press_completeness)
                                    if isinstance(press_completeness, Mapping)
                                    else {}
                                ),
                                "action_applied": False,
                            })
                            self.console.emit(
                                f"PRESS frozen: {sequence_text}"
                            )
                            self.console.emit(
                                scheduled.timing.console_schedule(
                                    scheduled.sequence
                                )
                            )
                        for press_event in schedule_events:
                            self.logger.event(press_event.event_type, {
                                **dict(press_event.payload),
                                "frame_index": captured,
                                "runtime_state": self.fsm.state.value,
                            })

                    pending_press = self._press_live_emission.pending
                    pending_cancel_reason: str | None = None
                    panel_disappeared = bool(
                        qualified_press_for_shadow
                        and qualified_press_for_shadow.evidence.get(
                            "panel_disappeared"
                        ) is True
                    )
                    press_episode_index = self._press_shadow.episode_index
                    if (
                        panel_disappeared
                        and isinstance(press_completeness, Mapping)
                        and not bool(press_completeness.get("complete"))
                        and press_episode_index not in self._press_abstained_episodes
                    ):
                        self._press_abstained_episodes.add(press_episode_index)
                        abstain_payload = {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "runtime_state": self.fsm.state.value,
                            "episode_index": press_episode_index,
                            "reason": "panel_disappeared_before_complete_certificate",
                            **dict(press_completeness),
                        }
                        self.logger.event(
                            "press_completeness_rejected", abstain_payload
                        )
                        self.logger.event(
                            "press_sequence_abstained_incomplete", abstain_payload
                        )
                        self._press_anomaly_evidence.trigger(
                            episode_index=press_episode_index,
                            reason="press_sequence_abstained_incomplete",
                        )
                    if pending_press is not None:
                        pending_cancel_reason = (
                            pending_press_cancellation_reason(
                                pending_press,
                                runtime_state=self.fsm.state,
                                active_episode=self._press_shadow.active,
                                episode_index=self._press_shadow.episode_index,
                                panel_disappeared=panel_disappeared,
                                foreground=foreground,
                                panic_triggered=(
                                    panic_triggered_this_frame
                                ),
                            )
                        )
                    if pending_cancel_reason is not None:
                        cancelled = self._press_live_emission.cancel_pending(
                            timestamp=elapsed,
                            reason=pending_cancel_reason,
                        )
                        self._press_shadow.record_emission_result(
                            safety_reason=pending_cancel_reason,
                            action_sink_called=False,
                            terminal_outcome="cancelled",
                            total_key_count=len(pending_press.sequence),
                        )
                        self.console.emit("PRESS cancelled")
                        for press_event in cancelled:
                            self.logger.event(press_event.event_type, {
                                **dict(press_event.payload),
                                "frame_index": captured,
                                "runtime_state": self.fsm.state.value,
                            })
                        pending_press = None

                    if (
                        pending_press is not None
                        and self._press_live_emission.due(elapsed)
                    ):
                        if pending_cancel_reason is None:
                            frozen_evidence = (
                                pending_press.frozen_evidence
                                or last_result.evidence
                            )
                            press_shadow_request = ActionRequest(
                                ActionIntent.PRESS_SEQUENCE,
                                frozen_evidence.confidence,
                                "press_sequence_scheduled_emission",
                                payload={
                                    "sequence": pending_press.sequence,
                                    "shadow_only": False,
                                    "slot_capacity": pending_press.slot_capacity,
                                    "active_press_episode": True,
                                    "panel_confirmed": True,
                                    "frozen_by_consensus": True,
                                    "episode_index": pending_press.episode_index,
                                    "press_timing_plan": (
                                        pending_press.timing.payload()
                                    ),
                                },
                            )
                            press_safety = self.controller.evaluate_external_action_safety(
                                press_shadow_request,
                                frozen_evidence,
                                timestamp=elapsed,
                                state=self.fsm.state,
                                foreground=foreground,
                                runtime_environment_supported=True,
                            )
                            press_fast_safety_reason = press_safety.reason
                            if (
                                press_safety.reason
                                == "action_emission_disabled"
                                and self.action_sink is not None
                                and ActionIntent.PRESS_SEQUENCE
                                in self.action_allowlist
                                and self.controller.stage_external_press_sequence(
                                    press_shadow_request
                                )
                            ):
                                press_fast_would_fire = self.deduplicator.observe(
                                    press_shadow_request,
                                    safety_reason="action_emission_disabled",
                                    frame_index=captured,
                                    timestamp=elapsed,
                                    runtime_state=self.fsm.state.value,
                                    prompt_evidence=None,
                                    specialized_evidence={},
                                )
                                scheduled_attempt, start_events = (
                                    self._press_live_emission.begin_scheduled_attempt(
                                        timestamp=elapsed
                                    )
                                )
                                press_fast_events = start_events
                                if (
                                    scheduled_attempt is not None
                                    and press_fast_would_fire is not None
                                ):
                                    action_id = str(
                                        press_fast_would_fire["deduplication_key"]
                                    )
                                    press_fast_would_fire.update({
                                        "action_id": action_id,
                                        "episode_id": str(
                                            scheduled_attempt.episode_index
                                        ),
                                    })
                                    press_fast_apply_called = True
                                    self.console.emit(
                                        "PRESS emitting: "
                                        + " ".join(scheduled_attempt.sequence)
                                    )
                                    press_fast_execution = self.action_sink.apply(
                                        press_shadow_request,
                                        ActionExecutionContext(
                                            action_id=action_id,
                                            episode_id=str(
                                                scheduled_attempt.episode_index
                                            ),
                                            requested_at=elapsed,
                                            capture_frame_index=captured,
                                            runtime_state=RuntimeState.PRESS.value,
                                            target_hwnd=self._capture_diagnostics.get(
                                                "hwnd"
                                            ),
                                        ),
                                    )
                                    execution_events = (
                                        self._press_live_emission.record_execution(
                                            episode_index=(
                                                scheduled_attempt.episode_index
                                            ),
                                            timestamp=(
                                                press_fast_execution.completed_at
                                            ),
                                            execution=press_fast_execution,
                                        )
                                    )
                                    press_fast_events = (
                                        *press_fast_events,
                                        *execution_events,
                                    )
                                    outcome = (
                                        "completed"
                                        if press_fast_execution.applied
                                        else (
                                            "partial"
                                            if press_fast_execution.partial_execution
                                            else "failed"
                                        )
                                    )
                                    self.console.emit(f"PRESS {outcome}")
                                    self._press_shadow.record_emission_result(
                                        safety_reason=press_safety.reason,
                                        action_sink_called=True,
                                        terminal_outcome=outcome,
                                        attempted_count=(
                                            press_fast_execution.attempted_count
                                        ),
                                        completed_key_count=(
                                            press_fast_execution.completed_key_count
                                        ),
                                        total_key_count=(
                                            press_fast_execution.total_key_count
                                        ),
                                        timestamp=elapsed,
                                        frame_index=captured,
                                    )
                                    if press_fast_execution.applied:
                                        actions_applied += 1
                                        press_fast_commit = (
                                            self.controller.commit_external_action(
                                                press_shadow_request,
                                                elapsed,
                                            )
                                        )
                                        if not press_fast_commit.action_applied:
                                            result_name = (
                                                "safe_stop_action_commit_failure"
                                            )
                                            stop_after_action_commit_failure = True
                                    else:
                                        self.controller.discard_external_proposal()
                                    for press_event in press_fast_events:
                                        self.logger.event(
                                            press_event.event_type,
                                            {
                                                **dict(press_event.payload),
                                                "frame_index": captured,
                                                "runtime_state": self.fsm.state.value,
                                            },
                                        )
                            else:
                                pending_cancel_reason = (
                                    press_safety.reason
                                    if press_safety.reason
                                    != "action_emission_disabled"
                                    else "press_emission_stage_rejected"
                                )
                                self.controller.discard_external_proposal()
                        if pending_cancel_reason is not None:
                            cancelled = self._press_live_emission.cancel_pending(
                                timestamp=elapsed,
                                reason=pending_cancel_reason,
                            )
                            self._press_shadow.record_emission_result(
                                safety_reason=pending_cancel_reason,
                                action_sink_called=False,
                                terminal_outcome="cancelled",
                                total_key_count=len(pending_press.sequence),
                            )
                            self.console.emit("PRESS cancelled")
                            for press_event in cancelled:
                                self.logger.event(press_event.event_type, {
                                    **dict(press_event.payload),
                                    "frame_index": captured,
                                    "runtime_state": self.fsm.state.value,
                                })
                    press_visual_events = (
                        self._press_live_emission.observe_panel(
                            timestamp=elapsed,
                            panel_observed=(
                                qualified_press_for_shadow is not None
                            ),
                            panel_disappeared=bool(
                                qualified_press_for_shadow
                                and qualified_press_for_shadow
                                .evidence.get(
                                    "panel_disappeared"
                                )
                                is True
                            ),
                        )
                    )
                    for press_event in press_visual_events:
                        if (
                            press_event.event_type
                            == "press_visual_acknowledged"
                        ):
                            self._press_shadow.record_visual_ack(
                                timestamp=elapsed,
                                frame_index=captured,
                            )
                            if self._press_v3_shadow is not None:
                                self._press_v3_shadow.record_visual_ack(
                                    "acknowledged"
                                )
                        elif (
                            press_event.event_type
                            == "press_visual_ack_timeout"
                        ):
                            self._press_anomaly_evidence.trigger(
                                episode_index=int(
                                    press_event.payload.get(
                                        "episode_index",
                                        self._press_shadow.episode_index,
                                    )
                                ),
                                reason="press_visual_ack_timeout",
                            )
                            if self._press_v3_shadow is not None:
                                self._press_v3_shadow.record_visual_ack(
                                    "timeout"
                                )
                        self.logger.event(
                            press_event.event_type,
                            {
                                **dict(press_event.payload),
                                "frame_index": captured,
                                "runtime_state": self.fsm.state.value,
                            },
                        )
                    if (
                        hook_critical_mode
                        and self._hook_decision_trace is not None
                    ):
                        raw_hook_evidence = (
                            hook.evidence if hook is not None else {}
                        )
                        qualified_hook_result = (
                            last_result.qualified.bundle.hook
                        )
                        hook_qualification = (
                            last_result.qualified.hook
                        )
                        hook_decision = (
                            self.fsm.last_hook_action_decision
                        )
                        decision_payload = (
                            hook_decision.payload()
                            if hook_decision is not None else {}
                        )
                        self._hook_decision_trace.record({
                            "timestamp": elapsed,
                            "capture_frame_index": captured,
                            "raw_detector_detected": bool(
                                hook and hook.detected
                            ),
                            "raw_detector_confidence": (
                                float(hook.confidence)
                                if hook is not None else None
                            ),
                            "divider_line_x": raw_hook_evidence.get(
                                "divider_line_x"
                            ),
                            "divider_confidence": (
                                raw_hook_evidence.get(
                                    "divider_confidence"
                                )
                            ),
                            "fill_endpoint_x": raw_hook_evidence.get(
                                "fill_endpoint_x"
                            ),
                            "current_hook_geometry_is_usable": (
                                decision_payload.get(
                                    "current_hook_geometry_is_usable",
                                    False,
                                )
                            ),
                            "hook_episode_active": (
                                decision_payload.get(
                                    "hook_episode_active",
                                    self.fsm.hook_episode_active,
                                )
                            ),
                            "qualified_hook_observation_retained": (
                                qualified_hook_result is not None
                            ),
                            "qualified_hook_detected": bool(
                                qualified_hook_result
                                and qualified_hook_result.detected
                            ),
                            "hook_qualification_reason": (
                                hook_qualification.qualification_reason
                            ),
                            "hook_used_by_fusion": (
                                hook_qualification.used_by_fusion
                            ),
                            "hook_action_policy_reason": (
                                decision_payload.get(
                                    "reason",
                                    "not_evaluated_before_"
                                    + last_result.fsm.transition_reason,
                                )
                            ),
                            "hook_action_policy_action_ready": bool(
                                decision_payload.get(
                                    "action_ready",
                                    False,
                                )
                            ),
                            "fsm_state_before": (
                                runtime_state_before_controller.value
                            ),
                            "fsm_committed_state": (
                                last_result.fsm.next_state.value
                            ),
                            "fsm_transition_reason": (
                                last_result.fsm.transition_reason
                            ),
                            "safety_decision": (
                                last_result.safety.decision.value
                            ),
                            "safety_reason": last_result.safety.reason,
                            "proposal_created": (
                                request.intent
                                == ActionIntent.HOOK_ACTION
                            ),
                        })
                    hook_fast_would_fire: dict[str, Any] | None = None
                    hook_fast_execution: ActionExecutionResult | None = None
                    hook_fast_commit = None
                    hook_fast_apply_called = False
                    hook_fast_blockers: list[str] = []
                    hook_fast_opportunity_id: str | None = None
                    hook_fast_already_consumed = False
                    hook_geometry_ready_at: float | None = None
                    safety_completed_at: float | None = None
                    if request.intent == ActionIntent.HOOK_ACTION:
                        self._hook_action_lifecycle.begin_episode(
                            cycle_id=self.deduplicator.cycle_id,
                            timestamp=self.fsm.state_since,
                            start_hook_applied=False,
                        )
                        self._hook_action_lifecycle.mark_opportunity_created()
                        hook_geometry_ready_at = self.clock() - started
                        safety_completed_at = self.clock() - started
                        self.deduplicator.record_raw_proposal(request)
                        hook_fast_opportunity_id = (
                            self.deduplicator.opportunity_id(
                                ActionIntent.HOOK_ACTION
                            )
                        )
                        hook_fast_already_consumed = (
                            self.deduplicator.already_consumed(
                                ActionIntent.HOOK_ACTION
                            )
                        )
                        if (
                            self.action_sink is not None
                            and ActionIntent.HOOK_ACTION
                            not in self.action_allowlist
                        ):
                            hook_fast_blockers.append(
                                "action_not_allowlisted"
                            )
                        if (
                            last_result.safety.reason
                            != "action_emission_disabled"
                        ):
                            hook_fast_blockers.append(
                                last_result.safety.reason
                            )
                        if hook_fast_already_consumed:
                            hook_fast_blockers.append(
                                "hook_opportunity_already_consumed"
                            )
                        if not hook_fast_blockers:
                            hook_fast_would_fire = (
                                self.deduplicator.observe(
                                    request,
                                    safety_reason=(
                                        last_result.safety.reason
                                    ),
                                    frame_index=captured,
                                    timestamp=elapsed,
                                    runtime_state=self.fsm.state.value,
                                    prompt_evidence=None,
                                    specialized_evidence={},
                                    count_raw=False,
                                )
                            )
                            if hook_fast_would_fire is None:
                                hook_fast_already_consumed = True
                                hook_fast_blockers.append(
                                    "hook_opportunity_already_consumed"
                                )
                        if (
                            hook_fast_would_fire is not None
                            and self.action_sink is not None
                        ):
                            self._hook_action_lifecycle.mark_action_started()
                            action_id = str(
                                hook_fast_would_fire[
                                    "deduplication_key"
                                ]
                            )
                            hook_fast_would_fire["action_id"] = action_id
                            hook_fast_would_fire["episode_id"] = str(
                                hook_fast_would_fire["cycle_id"]
                            )
                            hook_fast_apply_called = True
                            hook_fast_execution = self.action_sink.apply(
                                request,
                                ActionExecutionContext(
                                    action_id=action_id,
                                    episode_id=str(
                                        hook_fast_would_fire["episode_id"]
                                    ),
                                    requested_at=elapsed,
                                    capture_frame_index=captured,
                                    runtime_state=self.fsm.state.value,
                                    target_hwnd=(
                                        self._capture_diagnostics.get(
                                            "hwnd"
                                        )
                                    ),
                                ),
                            )
                            self._hook_action_lifecycle.mark_emission_result(
                                emission_started=(
                                    hook_fast_execution.started_at is not None
                                ),
                                applied=hook_fast_execution.applied,
                            )
                            if hook_fast_execution.applied:
                                self.console.emit("HOOK_ACTION applied")
                                actions_applied += 1
                                hook_fast_commit = (
                                    self.controller.commit_external_action(
                                        request,
                                        elapsed,
                                    )
                                )
                                if not hook_fast_commit.action_applied:
                                    result_name = (
                                        "safe_stop_action_commit_failure"
                                    )
                                    stop_after_action_commit_failure = True
                            else:
                                self.controller.discard_external_proposal()
                            self._flush_hook_decision_trace(
                                "action_apply_completed"
                            )
                    if (
                        defer_video_for_hook_fast_path
                        and not hook_critical_mode
                        and self.evidence_recorder is not None
                    ):
                        try:
                            self.evidence_recorder.record_frame(
                                frame,
                                capture_frame_index=captured,
                                timestamp=elapsed,
                            )
                            deferred_video_recorded = True
                        except Exception as exc:
                            evidence_failure_reason = (
                                f"video: {type(exc).__name__}: {exc}"
                            )
                            disable = getattr(
                                self.evidence_recorder, "disable", None
                            )
                            if callable(disable):
                                disable(evidence_failure_reason)
                            self.logger.event(
                                "diagnostic_evidence_failure",
                                {
                                    "timestamp": elapsed,
                                    "frame_index": captured,
                                    "reason": evidence_failure_reason,
                                },
                            )
                    collect_retry_enabled = bool(
                        self.action_sink is not None
                        and ActionIntent.COLLECT in self.action_allowlist
                    )
                    cast_opportunity_enabled = bool(
                        cast_tracking_enabled
                    )
                    qualified_get = last_result.qualified.bundle.get
                    qualified_banner = last_result.qualified.bundle.result_banner
                    if (
                        cast_opportunity_enabled
                        and self.cast_opportunity.waiting_for_acknowledgement
                    ):
                        qualified_hook_for_cast = last_result.qualified.bundle.hook
                        qualified_press_for_cast = last_result.qualified.bundle.press
                        cast_cancel_reason = None
                        if qualified_get is not None and qualified_get.detected:
                            cast_cancel_reason = "qualified_get_during_cast_pending"
                        elif (
                            qualified_hook_for_cast is not None
                            and qualified_hook_for_cast.detected
                        ):
                            cast_cancel_reason = "qualified_hook_during_cast_pending"
                        elif (
                            qualified_press_for_cast is not None
                            and qualified_press_for_cast.detected
                        ):
                            cast_cancel_reason = "qualified_press_during_cast_pending"
                        if cast_cancel_reason is not None:
                            self._log_cast_events(
                                self.cast_opportunity.cancel(
                                    timestamp=elapsed,
                                    reason=cast_cancel_reason,
                                ),
                                timestamp=elapsed,
                                frame_index=captured,
                                runtime_state=self.fsm.state.value,
                            )
                    if collect_retry_enabled:
                        get_confirmation_frames = int(
                            qualified_get.evidence.get("get_confirmation_frames", 0)
                            if qualified_get is not None else 0
                        )
                        collect_observation_events = self.collect_retry.observe_panel(
                            opportunity_id=(
                                f"cycle:{self.deduplicator.cycle_id}:COLLECT"
                            ),
                            timestamp=elapsed,
                            panel_observed=qualified_get is not None,
                            panel_visible=bool(qualified_get and qualified_get.detected),
                            get_confidence=(
                                qualified_get.confidence if qualified_get is not None else 0.0
                            ),
                            get_confirmation_frames=get_confirmation_frames,
                        )
                        self._log_collect_events(
                            collect_observation_events,
                            timestamp=elapsed,
                            frame_index=captured,
                            runtime_state=self.fsm.state.value,
                        )
                        if any(
                            item.event_type == "collect_retry_exhausted"
                            for item in collect_observation_events
                        ) and self.fsm.state != RuntimeState.SYNC_REQUIRED:
                            transition_results.append(self.fsm.force_state(
                                RuntimeState.SYNC_REQUIRED,
                                elapsed,
                                "collect_visual_ack_timeout",
                            ))
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
                    if cast_opportunity_enabled:
                        sink_summary = getattr(self.action_sink, "summary", None)
                        panic_latched_for_clearance = bool(
                            callable(sink_summary)
                            and sink_summary().get("panic_triggered", False)
                        )
                        clearance_events = self.cast_clearance.observe(
                            timestamp=elapsed,
                            previous_state=last_result.fsm.previous_state,
                            current_state=self.fsm.state,
                            prompt_kind=(
                                prompt.kind if prompt is not None else None
                            ),
                            prompt_frame_index=(
                                prompt.frame_index if prompt is not None else None
                            ),
                            get_observation=qualified_get,
                            get_activation_mode=(
                                last_result.qualified.get.activation_mode
                            ),
                            result_banner=qualified_banner,
                            physical_get_episode_open=(
                                self.collect_retry.episode_open
                            ),
                            physical_get_episode_id=(
                                self.collect_retry.physical_episode_id
                            ),
                            physical_get_episode_terminal=(
                                self.collect_retry.episode_terminal
                            ),
                            physical_get_panel_visible=(
                                self.collect_retry.panel_visible
                            ),
                            collect_visual_acknowledged=(
                                self.collect_retry.visual_acknowledged
                            ),
                            collect_complete_emission_count=(
                                self.collect_retry.complete_emission_count
                            ),
                            collect_terminal_reason=(
                                self.collect_retry.terminal_reason
                            ),
                            runtime_cycle_id=(
                                f"cycle:{self.deduplicator.cycle_id}"
                            ),
                            foreground_confirmed=(foreground is True),
                            panic_latched=panic_latched_for_clearance,
                        )
                        self._log_cast_events(
                            clearance_events,
                            timestamp=elapsed,
                            frame_index=captured,
                            runtime_state=self.fsm.state.value,
                        )

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
                        preserve_hook_cycle = (
                            self._hook_action_lifecycle.can_rearm_after_sync()
                        )
                        self.controller.reset_for_sync_recovery(elapsed)
                        self.recovery_synchronizer.reset(started_at=elapsed)
                        self.deduplicator.reset_for_sync_recovery(
                            preserve_cycle=preserve_hook_cycle
                        )
                        if preserve_hook_cycle:
                            self.deduplicator.release(
                                ActionIntent.HOOK_ACTION
                            )
                        self.logger.event("sync_recovery_started", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "recovery_target": None,
                            "support_frames": 0,
                            "duration_seconds": 0.0,
                            "rejection_reason": "fresh_window_after_sync_required",
                        })
                    elif (
                        processing_from_sync_required
                        and self.fsm.state == RuntimeState.SYNC_REQUIRED
                    ):
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
                            if (
                                recovered.previous_state
                                == RuntimeState.SYNC_REQUIRED
                                and recovered.next_state == RuntimeState.HOOK
                                and recovery.reason
                                == "qualified_hook_sync_recovery"
                            ):
                                recovery_blockers = (
                                    self._hook_recovery_safety_blockers(
                                        foreground=foreground
                                    )
                                )
                                if (
                                    not recovery_blockers
                                    and self._hook_action_lifecycle.rearm_after_sync()
                                    and self.fsm.rearm_hook_action_opportunity(
                                        elapsed
                                    )
                                ):
                                    self.deduplicator.release(
                                        ActionIntent.HOOK_ACTION
                                    )
                                    recovered_hook = (
                                        last_result.qualified.bundle.hook
                                    )
                                    hook_age = (
                                        max(
                                            0.0,
                                            elapsed
                                            - recovered_hook.timestamp,
                                        )
                                        if recovered_hook is not None else None
                                    )
                                    self.logger.event(
                                        "hook_action_rearmed_after_sync_recovery",
                                        {
                                            "timestamp": elapsed,
                                            "frame_index": captured,
                                            **self._hook_lifecycle_event_payload(
                                                state_age_seconds=0.0,
                                                hook_confidence=(
                                                    recovered_hook.confidence
                                                    if recovered_hook is not None
                                                    else recovery.confidence
                                                ),
                                                hook_age_seconds=hook_age,
                                                recovery_reason=(
                                                    recovery.reason
                                                ),
                                            ),
                                        },
                                    )

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

                    if (
                        self.evidence_recorder is not None
                        and not hook_critical_mode
                    ):
                        executed = {
                            "prompt": prompt_due,
                            "hook": hook is not None,
                            "press": press is not None,
                            "get": get_detector_executed,
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
                            evidence_failure_reason = f"roi: {type(exc).__name__}: {exc}"
                            disable = getattr(
                                self.evidence_recorder, "disable", None
                            )
                            if callable(disable):
                                disable(evidence_failure_reason)
                            self.logger.event("diagnostic_evidence_failure", {
                                "timestamp": elapsed,
                                "frame_index": captured,
                                "reason": evidence_failure_reason,
                            })

                    if prompt and prompt.kind.value != last_prompt_kind:
                        self.logger.event("prompt_label_change", {
                            "timestamp": elapsed,
                            "frame_index": captured,
                            "previous_label": last_prompt_kind,
                            "next_label": prompt.kind.value,
                            "prompt_evidence": dict(prompt.evidence),
                        })
                        if prompt.kind == PromptObservationKind.READY_BITE:
                            self.console.emit("READY detected")
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
                        if not self.enable_live_press_sequence:
                            completeness_payload = qualified_press.evidence.get(
                                "press_completeness_certificate", {}
                            )
                            self.logger.event("press_sequence_frozen", {
                                "timestamp": elapsed,
                                "frame_index": captured,
                                "selected_clean_frame": selected_clean,
                                "sequence": list(qualified_press.sequence_candidate),
                                **(
                                    dict(completeness_payload)
                                    if isinstance(completeness_payload, Mapping)
                                    else {}
                                ),
                            })

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
                            self.console.emit("GET detected")
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

                    cycle_completed_this_frame = any(
                        item.next_state == RuntimeState.IDLE
                        and item.previous_state not in {
                            RuntimeState.IDLE,
                            RuntimeState.SYNCING,
                            RuntimeState.SYNC_REQUIRED,
                        }
                        for item in transition_results
                    )
                    diagnostic_transition = next(
                        (
                            item for item in reversed(transition_results)
                            if item.previous_state != item.next_state
                        ),
                        None,
                    )
                    for transition_result in transition_results:
                        if transition_result.previous_state == transition_result.next_state:
                            continue
                        screenshot = (
                            self.logger.save_screenshot(frame, captured, "runtime_transition")
                            if self.live_config.save_transition_frames else None
                        )
                        self._log_transition(
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
                            transition_result.previous_state
                            == RuntimeState.RESULT_PENDING
                            and transition_result.next_state
                            == RuntimeState.GET
                            and self.deduplicator.cycle_id
                            not in self._press_seen_cycles
                        ):
                            self.console.emit(
                                "PERFECT or no-PRESS path inferred: "
                                "RESULT_PENDING -> GET"
                            )
                        if (
                            transition_result.next_state == RuntimeState.IDLE
                            and transition_result.previous_state not in {
                                RuntimeState.IDLE,
                                RuntimeState.SYNCING,
                                RuntimeState.SYNC_REQUIRED,
                            }
                        ):
                            self.deduplicator.finish_cycle()
                            self._hook_action_lifecycle.finish_episode()
                            completed_cycles += 1
                            if self.evidence_recorder is not None:
                                self.logger.event("diagnostic_cycle_completed", {
                                    "timestamp": elapsed,
                                    "frame_index": captured,
                                    "episode_id": evidence_episode_id,
                                    "completed_cycles": completed_cycles,
                                    "actions_applied": actions_applied,
                                })
                            evidence_episode_id += 1
                            stop_after_completed_cycle = bool(
                                self.live_config.max_completed_cycles is not None
                                and self.live_config.max_completed_cycles > 0
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
                    request = (
                        ActionRequest(
                            ActionIntent.NONE,
                            0.0,
                            "authoritative_idle_recovery_frame_blocks_actions",
                        )
                        if authoritative_idle_applied_this_frame
                        else last_result.fsm.action_request
                    )
                    collect_attempt: CollectAttempt | None = None
                    cast_attempt: CastAttempt | None = None
                    cast_blockers: tuple[str, ...] = ()
                    if cast_opportunity_enabled:
                        clearance_status = self.cast_clearance.status(elapsed)
                        post_clearance = self.cast_clearance.current(
                            elapsed
                        )
                        if (
                            self.fsm.state == RuntimeState.IDLE
                            and post_clearance is not None
                        ):
                            _, arming_events = (
                                self._cast_arming
                                .request_cast_opportunity(
                                    source_type=(
                                        CAST_SOURCE_POST_CYCLE
                                    ),
                                    source_id=(
                                        "post_cycle:"
                                        + post_clearance.clearance_id
                                    ),
                                    physical_idle_id=(
                                        idle_certificate.physical_idle_id
                                        if idle_certificate is not None
                                        else (
                                            "post_idle:"
                                            + post_clearance.clearance_id
                                        )
                                    ),
                                    cycle_id=(
                                        self.deduplicator.cycle_id
                                    ),
                                    timestamp=elapsed,
                                    cooldown_seconds=0.0,
                                )
                            )
                            for arming_event in arming_events:
                                self.logger.event(
                                    arming_event.event_type,
                                    {
                                        "timestamp": elapsed,
                                        "frame_index": captured,
                                        "previous_state": (
                                            self.fsm.state.value
                                        ),
                                        "clearance_id": (
                                            post_clearance.clearance_id
                                        ),
                                        **dict(arming_event.payload),
                                    },
                                )
                        expired_cast_arm = (
                            self._cast_arming.expire_if_overdue(
                                timestamp=elapsed,
                                service_timeout_seconds=(
                                    self.live_config
                                    .idle_cast_liveness_timeout_seconds
                                ),
                            )
                        )
                        if expired_cast_arm is not None:
                            self.logger.event("cast_arming_expired", {
                                "timestamp": elapsed,
                                "frame_index": captured,
                                "runtime_state": self.fsm.state.value,
                                **expired_cast_arm.payload(elapsed),
                            })
                        self._service_idle_liveness_arm(
                            certificate=idle_certificate,
                            timestamp=elapsed,
                            frame_index=captured,
                        )
                        active_cast_arm = self._cast_arming.active
                        arm_uses_idle_certificate = bool(
                            active_cast_arm is not None
                            and active_cast_arm.source_type
                            in {
                                CAST_SOURCE_STARTUP,
                                CAST_SOURCE_RECOVERY,
                            }
                        )
                        cast_arm_ready = self._cast_arming.ready(
                            timestamp=elapsed,
                            runtime_idle=(
                                self.fsm.state == RuntimeState.IDLE
                            ),
                            idle_certificate_valid=(
                                idle_certificate is not None
                                if arm_uses_idle_certificate else True
                            ),
                        )
                        cast_safety_reason = last_result.safety.reason
                        if (
                            cast_arm_ready
                            and request.intent == ActionIntent.NONE
                            and active_cast_arm is not None
                        ):
                            request = ActionRequest(
                                ActionIntent.CAST,
                                last_result.evidence.confidence,
                                "armed_cast_lifecycle_ready",
                                payload={
                                    "source_type": (
                                        active_cast_arm.source_type
                                    ),
                                    "source_id": active_cast_arm.source_id,
                                    "physical_idle_id": (
                                        active_cast_arm.physical_idle_id
                                    ),
                                },
                            )
                            cast_safety = (
                                self.controller
                                .evaluate_external_action_safety(
                                    request,
                                    last_result.evidence,
                                    timestamp=elapsed,
                                    state=self.fsm.state,
                                    foreground=foreground,
                                    runtime_environment_supported=True,
                                    get_panel_present=False,
                                )
                            )
                            cast_safety_reason = cast_safety.reason
                            if (
                                cast_safety_reason
                                == "action_emission_disabled"
                                and not self.controller.stage_external_cast(
                                    request
                                )
                            ):
                                cast_safety_reason = (
                                    "cast_external_proposal_not_staged"
                                )
                        cast_blockers = self._cast_blockers(
                            status=clearance_status,
                            runtime_state=self.fsm.state,
                            prompt_kind=(
                                prompt.kind if prompt is not None else None
                            ),
                            foreground=foreground,
                            raw_intent=request.intent,
                            safety_reason=cast_safety_reason,
                            source_type=(
                                active_cast_arm.source_type
                                if active_cast_arm is not None else None
                            ),
                            idle_certificate_valid=(
                                idle_certificate is not None
                            ),
                            cast_arm_ready=cast_arm_ready,
                        )
                        self._log_cast_blockers(
                            cast_blockers,
                            status=clearance_status,
                            timestamp=elapsed,
                            frame_index=captured,
                            prompt_kind=(
                                prompt.kind if prompt is not None else None
                            ),
                            raw_intent=request.intent,
                            cycle_completed=cycle_completed_this_frame,
                            transition_from=(
                                diagnostic_transition.previous_state
                                if diagnostic_transition is not None else None
                            ),
                            transition_to=(
                                diagnostic_transition.next_state
                                if diagnostic_transition is not None else None
                            ),
                        )
                    if request.intent == ActionIntent.HOOK_ACTION:
                        would_fire = hook_fast_would_fire
                        if would_fire is not None:
                            would_fire["prompt_evidence"] = (
                                dict(prompt.evidence)
                                if prompt is not None else {}
                            )
                            would_fire["specialized_evidence"] = specialized
                        if hook_fast_blockers:
                            blocker_key = (
                                str(hook_fast_opportunity_id),
                                tuple(hook_fast_blockers),
                            )
                            if (
                                blocker_key
                                not in self._logged_hook_action_blockers
                            ):
                                self._logged_hook_action_blockers.add(
                                    blocker_key
                                )
                                self.logger.event(
                                    "hook_action_blocked",
                                    {
                                        "timestamp": elapsed,
                                        "frame_index": captured,
                                        "runtime_state_before": (
                                            last_result.fsm
                                            .previous_state.value
                                        ),
                                        "committed_runtime_state": (
                                            self.fsm.state.value
                                        ),
                                        "transition_from": (
                                            last_result.fsm
                                            .previous_state.value
                                        ),
                                        "transition_to": (
                                            last_result.fsm
                                            .next_state.value
                                        ),
                                        "raw_intent": (
                                            request.intent.value
                                        ),
                                        "hook_evidence": (
                                            specialized.get("hook")
                                        ),
                                        "eligibility_blockers": (
                                            hook_fast_blockers
                                        ),
                                        "hook_opportunity_id": (
                                            hook_fast_opportunity_id
                                        ),
                                        "already_consumed": (
                                            hook_fast_already_consumed
                                        ),
                                        "action_applied": False,
                                    },
                                )
                    elif request.intent == ActionIntent.PRESS_SEQUENCE:
                        # The verified shadow proposal owns guarded Live
                        # dispatch. Never send the earlier raw FSM proposal.
                        if self.action_sink is not None:
                            self.controller.discard_external_proposal()
                        would_fire = None
                    elif cast_opportunity_enabled and request.intent == ActionIntent.CAST:
                        self.deduplicator.record_raw_proposal(request)
                        if (
                            cast_safety_reason == "action_emission_disabled"
                            and not cast_blockers
                            and active_cast_arm is not None
                        ):
                            clearance = self.cast_clearance.current(elapsed)
                            cast_attempt, cast_schedule_events = (
                                self.cast_opportunity.schedule(
                                    timestamp=elapsed,
                                    clearance_id=(
                                        clearance.clearance_id
                                        if (
                                            clearance is not None
                                            and active_cast_arm.source_type
                                            == CAST_SOURCE_POST_CYCLE
                                        )
                                        else None
                                    ),
                                    runtime_state=self.fsm.state,
                                    prompt_kind=(
                                        prompt.kind if prompt is not None else None
                                    ),
                                    physical_get_episode_open=(
                                        self.collect_retry.episode_open
                                    ),
                                    source_type=(
                                        active_cast_arm.source_type
                                    ),
                                    source_id=active_cast_arm.source_id,
                                    cycle_id=active_cast_arm.cycle_id,
                                )
                            )
                            if cast_attempt is not None:
                                self._cast_arming.mark_cast_started(
                                    cast_attempt.opportunity_id
                                )
                                if (
                                    active_cast_arm.source_type
                                    == CAST_SOURCE_POST_CYCLE
                                ):
                                    consumed, consume_events = (
                                        self.cast_clearance.consume_with_events(
                                            cast_attempt.clearance_id
                                        )
                                    )
                                    if not consumed:
                                        raise RuntimeError(
                                            "CAST opportunity consumed an invalid clearance"
                                        )
                                    self._log_cast_events(
                                        consume_events,
                                        timestamp=elapsed,
                                        frame_index=captured,
                                        runtime_state=self.fsm.state.value,
                                    )
                            self._log_cast_events(
                                cast_schedule_events,
                                timestamp=elapsed,
                                frame_index=captured,
                                runtime_state=self.fsm.state.value,
                            )
                        would_fire = (
                            self.deduplicator.observe(
                                request,
                                safety_reason=cast_safety_reason,
                                frame_index=captured,
                                timestamp=elapsed,
                                runtime_state=self.fsm.state.value,
                                prompt_evidence=prompt.evidence if prompt else None,
                                specialized_evidence=specialized,
                                identity_suffix=(
                                    cast_attempt.opportunity_id
                                    if cast_attempt is not None else None
                                ),
                                count_raw=False,
                            )
                            if cast_attempt is not None else None
                        )
                        if would_fire is not None and cast_attempt is not None:
                            would_fire["deduplication_key"] = cast_attempt.action_id
                            would_fire["cast_opportunity_id"] = (
                                cast_attempt.opportunity_id
                            )
                            would_fire["cast_source_type"] = (
                                cast_attempt.source_type
                            )
                            would_fire["cast_source_id"] = (
                                cast_attempt.source_id
                            )
                    elif collect_retry_enabled and request.intent == ActionIntent.COLLECT:
                        self.deduplicator.record_raw_proposal(request)
                        if last_result.safety.reason == "action_emission_disabled":
                            collect_attempt, collect_schedule_events = (
                                self.collect_retry.schedule_attempt(
                                    timestamp=elapsed,
                                    get_confidence=(
                                        qualified_get.confidence
                                        if qualified_get is not None else 0.0
                                    ),
                                    get_confirmation_frames=int(
                                        qualified_get.evidence.get(
                                            "get_confirmation_frames", 0
                                        ) if qualified_get is not None else 0
                                    ),
                                )
                            )
                            self._log_collect_events(
                                collect_schedule_events,
                                timestamp=elapsed,
                                frame_index=captured,
                                runtime_state=self.fsm.state.value,
                            )
                        elif last_result.safety.reason in {
                            "foreground_window_not_confirmed",
                            "unsupported_runtime_resolution",
                            "sync_required_blocks_actions",
                        } and self.collect_retry.active:
                            self._log_collect_events(
                                (self.collect_retry.cancel(
                                    elapsed, last_result.safety.reason
                                ),),
                                timestamp=elapsed,
                                frame_index=captured,
                                runtime_state=self.fsm.state.value,
                            )
                        would_fire = (
                            self.deduplicator.observe(
                                request,
                                safety_reason=last_result.safety.reason,
                                frame_index=captured,
                                timestamp=elapsed,
                                runtime_state=self.fsm.state.value,
                                prompt_evidence=prompt.evidence if prompt else None,
                                specialized_evidence=specialized,
                                identity_suffix=(
                                    f"attempt:{collect_attempt.attempt_number}"
                                    if collect_attempt is not None else None
                                ),
                                count_raw=False,
                            )
                            if collect_attempt is not None else None
                        )
                        if would_fire is not None and collect_attempt is not None:
                            would_fire["deduplication_key"] = collect_attempt.attempt_id
                            would_fire["collect_opportunity_id"] = (
                                collect_attempt.opportunity_id
                            )
                            would_fire["physical_get_episode_id"] = (
                                self.collect_retry.physical_episode_id
                            )
                    else:
                        would_fire = self.deduplicator.observe(
                            request,
                            safety_reason=last_result.safety.reason,
                            frame_index=captured,
                            timestamp=elapsed,
                            runtime_state=self.fsm.state.value,
                            prompt_evidence=prompt.evidence if prompt else None,
                            specialized_evidence=specialized,
                        )
                    if press_shadow_request is not None:
                        request = press_shadow_request
                        if self.enable_live_press_sequence:
                            would_fire = press_fast_would_fire
                            if would_fire is not None:
                                would_fire["prompt_evidence"] = (
                                    dict(prompt.evidence)
                                    if prompt is not None else {}
                                )
                                would_fire[
                                    "specialized_evidence"
                                ] = specialized
                                would_fire["safety_decision"] = (
                                    "ALLOW_LIVE_SINK"
                                )
                                would_fire["safety_reason"] = (
                                    press_fast_safety_reason
                                )
                        else:
                            would_fire = self.deduplicator.observe(
                                press_shadow_request,
                                safety_reason="action_emission_disabled",
                                frame_index=captured,
                                timestamp=elapsed,
                                runtime_state=self.fsm.state.value,
                                prompt_evidence=(
                                    prompt.evidence if prompt else None
                                ),
                                specialized_evidence=specialized,
                            )
                            if would_fire is not None:
                                would_fire["safety_decision"] = "DENY"
                                would_fire["safety_reason"] = (
                                    "press_sequence_shadow_only_not_live_allowlisted"
                                )
                    if would_fire:
                        event_type = would_fire.pop("event_type")
                        action_id = str(would_fire["deduplication_key"])
                        would_fire["action_id"] = action_id
                        would_fire["episode_id"] = (
                            str(self.collect_retry.physical_episode_id)
                            if collect_attempt is not None
                            else (
                                str(cast_attempt.opportunity_id)
                                if cast_attempt is not None
                                else (
                                    str(request.payload.get("episode_index"))
                                    if (
                                        event_type
                                        == "WOULD_PRESS_SEQUENCE"
                                    )
                                    else str(would_fire["cycle_id"])
                                )
                            )
                        )
                        if event_type == "WOULD_HOOK_ACTION":
                            sendinput_started_at = (
                                hook_fast_execution.started_at
                                if hook_fast_execution is not None
                                else None
                            )
                            sendinput_completed_at = (
                                hook_fast_execution.completed_at
                                if hook_fast_execution is not None
                                else None
                            )
                            would_fire.update({
                                "frame_captured_at": frame_captured_at,
                                "hook_geometry_ready_at": (
                                    hook_geometry_ready_at
                                ),
                                "safety_completed_at": (
                                    safety_completed_at
                                ),
                                "sendinput_started_at": (
                                    sendinput_started_at
                                ),
                                "sendinput_completed_at": (
                                    sendinput_completed_at
                                ),
                                "capture_to_sendinput_start_ms": (
                                    (
                                        sendinput_started_at
                                        - frame_captured_at
                                    )
                                    * 1000.0
                                    if sendinput_started_at is not None
                                    else None
                                ),
                                "action_ready_to_sendinput_start_ms": (
                                    (
                                        sendinput_started_at
                                        - hook_geometry_ready_at
                                    )
                                    * 1000.0
                                    if (
                                        sendinput_started_at is not None
                                        and hook_geometry_ready_at is not None
                                    )
                                    else None
                                ),
                            })
                        screenshot = self.logger.save_screenshot(frame, captured, event_type)
                        would_fire["screenshot_reference"] = screenshot
                        if event_type == "WOULD_HOOK_ACTION" and hook_crossed_at is not None:
                            would_fire["threshold_crossing_to_would_fire_ms"] = (
                                elapsed - hook_crossed_at
                            ) * 1000.0
                        self.logger.event(event_type, would_fire)
                        self.logger.update_cycle(self.deduplicator.cycle_id, event_type, would_fire)
                        if (
                            self.action_sink is not None
                            and hook_fast_apply_called
                        ):
                            execution = hook_fast_execution
                            assert execution is not None
                            if execution.applied:
                                commit = hook_fast_commit
                                assert commit is not None
                                if not commit.action_applied:
                                    self.logger.event("action_failed", {
                                        "timestamp": elapsed,
                                        "frame_index": captured,
                                        "action_id": action_id,
                                        "intent": request.intent.value,
                                        "reason": (
                                            "runtime_commit_failed_after_complete_input"
                                        ),
                                        "commit_reason": commit.reason,
                                        "action_applied": True,
                                    })
                                elif (
                                    commit.previous_state
                                    != commit.next_state
                                ):
                                    self._log_transition(
                                        timestamp=elapsed,
                                        frame_index=captured,
                                        previous_state=(
                                            commit.previous_state.value
                                        ),
                                        next_state=(
                                            commit.next_state.value
                                        ),
                                        reason=commit.reason,
                                        screenshot_reference=screenshot,
                                    )
                        elif event_type == "WOULD_PRESS_SEQUENCE":
                            # Shadow-only never reaches the sink. With the
                            # explicit Live opt-in, dispatch and commit already
                            # completed on the pre-diagnostics fast path.
                            if (
                                press_fast_apply_called
                                and press_fast_execution is not None
                                and press_fast_execution.applied
                            ):
                                assert press_fast_commit is not None
                                if not press_fast_commit.action_applied:
                                    self.logger.event("action_failed", {
                                        "timestamp": elapsed,
                                        "frame_index": captured,
                                        "action_id": action_id,
                                        "intent": request.intent.value,
                                        "reason": (
                                            "runtime_commit_failed_"
                                            "after_complete_input"
                                        ),
                                        "commit_reason": (
                                            press_fast_commit.reason
                                        ),
                                        "action_applied": True,
                                    })
                                elif (
                                    press_fast_commit.previous_state
                                    != press_fast_commit.next_state
                                ):
                                    self._log_transition(
                                        timestamp=elapsed,
                                        frame_index=captured,
                                        previous_state=(
                                            press_fast_commit
                                            .previous_state.value
                                        ),
                                        next_state=(
                                            press_fast_commit
                                            .next_state.value
                                        ),
                                        reason=(
                                            press_fast_commit.reason
                                        ),
                                        screenshot_reference=(
                                            screenshot
                                        ),
                                    )
                        elif self.action_sink is not None:
                            if collect_attempt is not None:
                                self._log_collect_events(
                                    (CollectRetryEvent(
                                        "collect_attempt_started",
                                        self.collect_retry.attempt_payload(collect_attempt),
                                    ),),
                                    timestamp=elapsed,
                                    frame_index=captured,
                                    runtime_state=self.fsm.state.value,
                                )
                            if cast_attempt is not None:
                                if (
                                    cast_attempt.source_type
                                    == CAST_SOURCE_STARTUP
                                ):
                                    self.logger.event(
                                        "startup_idle_cast_started",
                                        {
                                            "timestamp": elapsed,
                                            "frame_index": captured,
                                            "previous_state": (
                                                self.fsm.state.value
                                            ),
                                            "source_type": (
                                                cast_attempt.source_type
                                            ),
                                            "source_id": cast_attempt.source_id,
                                            "cycle_id": cast_attempt.cycle_id,
                                            "opportunity_id": (
                                                cast_attempt.opportunity_id
                                            ),
                                        },
                                    )
                                self._log_cast_events(
                                    (CastOpportunityEvent(
                                        "cast_attempt_started",
                                        {
                                            "opportunity_id": cast_attempt.opportunity_id,
                                            "action_id": cast_attempt.action_id,
                                            "os_input_emitted": False,
                                        },
                                    ),),
                                    timestamp=elapsed,
                                    frame_index=captured,
                                    runtime_state=self.fsm.state.value,
                                )
                            self._action_emission_in_progress = True
                            try:
                                execution = self.action_sink.apply(
                                    request,
                                    ActionExecutionContext(
                                        action_id=action_id,
                                        episode_id=str(would_fire["episode_id"]),
                                        requested_at=elapsed,
                                        capture_frame_index=captured,
                                        runtime_state=self.fsm.state.value,
                                        target_hwnd=self._capture_diagnostics.get("hwnd"),
                                    ),
                                )
                            finally:
                                self._action_emission_in_progress = False
                            if collect_attempt is not None:
                                self._log_collect_events(
                                    self.collect_retry.record_execution(
                                        collect_attempt,
                                        execution,
                                        timestamp=execution.completed_at,
                                    ),
                                    timestamp=execution.completed_at,
                                    frame_index=captured,
                                    runtime_state=self.fsm.state.value,
                                )
                            if cast_attempt is not None:
                                self._log_cast_events(
                                    self.cast_opportunity.record_execution(
                                        cast_attempt,
                                        execution,
                                        timestamp=execution.completed_at,
                                    ),
                                    timestamp=execution.completed_at,
                                    frame_index=captured,
                                    runtime_state=self.fsm.state.value,
                                )
                                self._cast_arming.record_execution(
                                    timestamp=execution.completed_at,
                                    emission_started=(
                                        execution.started_at is not None
                                    ),
                                    applied=execution.applied,
                                    terminal_outcome=(
                                        "applied"
                                        if execution.applied
                                        else (
                                            "partial"
                                            if execution.partial_execution
                                            else "failed"
                                        )
                                    ),
                                )
                            if execution.applied:
                                if request.intent == ActionIntent.START_HOOK:
                                    self.console.emit("START_HOOK applied")
                                actions_applied += 1
                                if request.intent == ActionIntent.CAST:
                                    self._runtime_cycle_started = True
                                commit = self.controller.commit_external_action(
                                    request,
                                    elapsed,
                                )
                                if not commit.action_applied:
                                    if collect_attempt is not None and self.collect_retry.active:
                                        self._log_collect_events(
                                            (self.collect_retry.cancel(
                                                elapsed,
                                                "runtime_commit_failed_after_complete_input",
                                                attempt=collect_attempt,
                                            ),),
                                            timestamp=elapsed,
                                            frame_index=captured,
                                            runtime_state=self.fsm.state.value,
                                        )
                                    self.logger.event("action_failed", {
                                        "timestamp": elapsed,
                                        "frame_index": captured,
                                        "action_id": action_id,
                                        "intent": last_result.fsm.action_request.intent.value,
                                        "reason": "runtime_commit_failed_after_complete_input",
                                        "commit_reason": commit.reason,
                                        "action_applied": True,
                                    })
                                    result_name = "safe_stop_action_commit_failure"
                                    stop_after_action_commit_failure = True
                                else:
                                    if request.intent == ActionIntent.START_HOOK:
                                        self._runtime_cycle_started = True
                                        self._hook_action_lifecycle.begin_episode(
                                            cycle_id=self.deduplicator.cycle_id,
                                            timestamp=elapsed,
                                            start_hook_applied=True,
                                        )
                                    if commit.previous_state != commit.next_state:
                                        self._log_transition(
                                            timestamp=elapsed,
                                            frame_index=captured,
                                            previous_state=commit.previous_state.value,
                                            next_state=commit.next_state.value,
                                            reason=commit.reason,
                                            screenshot_reference=screenshot,
                                        )
                                        activation = self.activation_policy.evaluate(
                                            self.fsm.state,
                                            raw_bundle,
                                            recorded_observation=True,
                                        )
                            else:
                                self.controller.discard_external_proposal()
                        else:
                            self.controller.discard_external_proposal()
                    elif self.action_sink is not None:
                        self.controller.discard_external_proposal()

                if (
                    defer_video_for_hook_fast_path
                    and not hook_critical_mode
                    and not deferred_video_recorded
                    and self.evidence_recorder is not None
                ):
                    try:
                        self.evidence_recorder.record_frame(
                            frame,
                            capture_frame_index=captured,
                            timestamp=elapsed,
                        )
                    except Exception as exc:
                        evidence_failure_reason = (
                            f"video: {type(exc).__name__}: {exc}"
                        )
                        disable = getattr(
                            self.evidence_recorder, "disable", None
                        )
                        if callable(disable):
                            disable(evidence_failure_reason)
                        self.logger.event(
                            "diagnostic_evidence_failure",
                            {
                                "timestamp": elapsed,
                                "frame_index": captured,
                                "reason": evidence_failure_reason,
                            },
                        )
                capture_fps = captured / max(1e-9, self.clock() - started)
                latency_ms = latencies[-1] if latencies else 0.0
                if not hook_critical_mode:
                    self.overlay.show(frame, self._overlay_lines(
                        capture_fps=capture_fps,
                        latency_ms=latency_ms,
                        prompt=last_prompt,
                        result=last_result,
                        activation=activation,
                        actions_applied=actions_applied,
                    ))
                if (
                    self.live_config.runtime_profile == "production"
                    and elapsed - last_production_heartbeat >= 30.0
                ):
                    sink_summary = (
                        self.action_sink.summary()
                        if self.action_sink is not None
                        and callable(getattr(self.action_sink, "summary", None))
                        else {}
                    )
                    hook_episodes = self._hook_episode_telemetry.summaries()
                    hook_fps = (
                        float(hook_episodes[-1].get(
                            "hook_detector_actual_fps", 0.0
                        ))
                        if hook_episodes else 0.0
                    )
                    self.console.heartbeat(
                        "production_runtime",
                        (
                            f"state={self.fsm.state.value} cycles={completed_cycles} "
                            f"capture_fps={capture_fps:.1f} hook_fps={hook_fps:.1f} "
                            f"actions_failed={sum(sink_summary.get('failed_action_counts', {}).values())} "
                            f"focus={not self._foreground_unavailable_event_active} "
                            f"logging={'degraded' if self.logger.logging_disabled else 'healthy'}"
                        ),
                        timestamp=elapsed,
                        minimum_interval_seconds=30.0,
                    )
                    last_production_heartbeat = elapsed
                elif (
                    self.live_config.runtime_profile == "diagnostic"
                    and not hook_critical_mode
                    and elapsed - last_terminal >= 1.0
                ):
                    print(
                        f"frame={captured} capture_fps={capture_fps:.1f} latency_ms={latency_ms:.1f} "
                        f"prompt={last_prompt.kind.value if last_prompt else 'N/A'} state={self.fsm.state.value} "
                        f"activation={activation.hook.value}/{activation.press.value}/{activation.get.value} "
                        f"intent={last_result.fsm.action_request.intent.value if last_result else 'NONE'} "
                        f"action_applied={bool(actions_applied)}",
                        flush=True,
                    )
                    last_terminal = elapsed
                if (
                    self._hook_episode_telemetry.active
                    and self.fsm.state not in {
                        RuntimeState.HOOK_PENDING,
                        RuntimeState.HOOK,
                    }
                ):
                    self._finish_hook_episode_telemetry(
                        timestamp=self.clock() - started,
                    )
                    self._flush_hook_decision_trace("episode_ended")
                if stop_after_completed_cycle:
                    result_name = "completed_target_cycles"
                    break
                if stop_after_action_commit_failure:
                    break
                loop_target_fps = (
                    self.live_config.hook_critical_target_fps
                    if hook_critical_mode
                    else self.live_config.max_fps
                )
                loop_interval = 1.0 / loop_target_fps
                remaining = (
                    loop_interval - (self.clock() - frame_loop_started)
                )
                if remaining > 0.0:
                    self.sleep(remaining)
        except KeyboardInterrupt:
            result_name = "interrupted_by_user"
            shutdown_reason = "ctrl_c"
        except LivePreflightError as exc:
            result_name = "preflight_failed"
            self._preflight_failure_reason = (
                self._preflight_failure_reason or "preflight_error"
            )
            self._preflight_failure_message = str(exc)
            python_integrity = self._action_preflight_diagnostics.get(
                "python_process", {}
            )
            target_integrity = self._action_preflight_diagnostics.get(
                "target_process", {}
            )
            self.logger.event("preflight_failed", {
                "timestamp": 0.0,
                "frame_index": 0,
                "preflight_passed": False,
                "preflight_failure_reason": self._preflight_failure_reason,
                "preflight_failure_message": self._preflight_failure_message,
                "reason": str(exc),
                "python_integrity": dict(python_integrity),
                "target_integrity": dict(target_integrity),
                "suspected_integrity_mismatch": self._action_preflight_diagnostics.get(
                    "suspected_integrity_mismatch"
                ),
            })
        finally:
            elapsed_total = max(0.0, self.clock() - started)
            if result_name == "interrupted_by_user":
                for press_event in self._press_live_emission.cancel_pending(
                    timestamp=elapsed_total,
                    reason="ctrl_c",
                ):
                    self.logger.event(press_event.event_type, {
                        **dict(press_event.payload),
                        "frame_index": captured,
                        "runtime_state": self.fsm.state.value,
                    })
            if self._hook_episode_telemetry.active:
                self._finish_hook_episode_telemetry(
                    timestamp=elapsed_total,
                )
            if self._hook_video_suspension_active is not None:
                self._finish_hook_video_suspension(
                    timestamp=elapsed_total,
                    frame_index=captured,
                )
            self._flush_hook_decision_trace("session_ended")
            press_shadow_summary: dict[str, Any] = {
                "press_decision_trace_path": None,
                "press_decision_trace_rows": 0,
                "press_roi_clip_path": None,
                "press_roi_frames_path": None,
                "press_roi_frame_count": 0,
                "press_roi_clip_fps": 0.0,
                "press_roi_pre_roll_seconds": 1.0,
                "press_roi_post_roll_seconds": 0.5,
                "press_roi_frames_dropped": 0,
                "press_review_items_path": None,
                "press_episode_review_items": [],
                "press_shadow_proposal_count": 0,
            }
            if self.evidence_recorder is not None:
                with ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="live-artifact-writer",
                ) as artifact_executor:
                    hook_future = artifact_executor.submit(
                        self._write_hook_roi_clip
                    )
                    press_future = artifact_executor.submit(
                        self._press_shadow.write_artifacts,
                        self.logger.path / "diagnostic_evidence",
                    )
                    hook_roi_clip_summary = hook_future.result()
                    press_shadow_summary = press_future.result()
            else:
                self._press_shadow.finish_session()
                hook_roi_clip_summary = self._write_hook_roi_clip()
            press_v3_summary: dict[str, Any] = {
                "press_v3_shadow_processed_frames": 0,
                "press_v3_shadow_dropped_busy_frames": 0,
                "press_v3_processing_latency_mean_ms": 0.0,
                "press_v3_processing_latency_p95_ms": 0.0,
                "press_v3_episode_count": 0,
                "press_v3_episode_summaries": [],
                "press_v3_debug_evidence_path": None,
                "press_v3_debug_evidence_frames": 0,
            }
            if self._press_v3_shadow is not None:
                press_v3_events, press_v3_summary = (
                    self._press_v3_shadow.finish(elapsed_total)
                )
                for event_type, payload in press_v3_events:
                    self.logger.event(event_type, {
                        "timestamp": elapsed_total,
                        **payload,
                    })
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
            press_anomaly_summary = self._press_anomaly_evidence.close()
            if self.live_config.runtime_profile == "production":
                self.console.emit(
                    f"shutdown: result={result_name} cycles={completed_cycles}"
                )
            self.overlay.close()
            self.console.close()
            if self._opened:
                self.capture.close()
                self._opened = False
            action_summary = {
                "action_sink_type": ACTION_SINK_NONE,
                "action_allowlist": sorted(item.value for item in self.action_allowlist),
                "attempted_action_counts": {},
                "applied_action_counts": {},
                "rejected_action_counts": {},
                "partial_action_counts": {},
                "failed_action_counts": {},
                "os_input_emitted_counts": {},
                "action_applied_semantics": "complete_os_input_not_visual_acknowledgement",
                "rejection_counts_by_reason": {},
                "panic_triggered": False,
                "focus_loss_count": 0,
                "foreground_unavailable_count": 0,
                "integrity_diagnostics": self._action_preflight_diagnostics,
            }
            if self.action_sink is not None:
                sink_summary = getattr(self.action_sink, "summary", None)
                if callable(sink_summary):
                    action_summary.update(sink_summary())
            if shutdown_reason == "duration_limit" and result_name != "completed":
                shutdown_reason = {
                    "completed_target_cycles": "completed_cycle_limit",
                    "preflight_failed": "fatal_preflight_failure",
                    "interrupted_by_user": "ctrl_c",
                }.get(result_name, result_name)
            hook_episode_summaries = self._hook_episode_telemetry.summaries()
            summary = {
                "result": result_name,
                "started_at": started_at_utc,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "shutdown_reason": shutdown_reason,
                "runtime_profile": self.live_config.runtime_profile,
                "visual_evidence_enabled": self.evidence_recorder is not None,
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
                "proposed_action_counts": dict(self.deduplicator.raw_proposals),
                "unique_would_fire": dict(self.deduplicator.unique_events),
                "actions_applied": actions_applied,
                "final_state": self.fsm.state.value,
                "emit_actions": self.emit_actions,
                "action_sink": (
                    self.action_sink_name if self.action_sink is not None else None
                ),
                "capture_backend": self._capture_diagnostics.get(
                    "backend", getattr(self.capture, "backend_name", "unknown")
                ),
                "capture_fallback_used": bool(self._capture_diagnostics.get("fallback_used", False)),
                "capture_diagnostics": self._capture_diagnostics,
                "completed_cycles": completed_cycles,
                "missed_ready_recovery_count": (
                    missed_ready_recovery_count
                ),
                "press_initial_delay_range_ms": [
                    self.live_config.press_initial_delay_min_ms,
                    self.live_config.press_initial_delay_max_ms,
                ],
                "press_inter_key_gap_range_ms": [
                    self.live_config.press_inter_key_gap_min_ms,
                    self.live_config.press_inter_key_gap_max_ms,
                ],
                "press_key_hold_ms": int(
                    self.live_config.press_key_hold_ms
                ),
                "hook_critical_target_fps": (
                    self.live_config.hook_critical_target_fps
                ),
                "hook_critical_episodes": (
                    hook_episode_summaries
                ),
                "hook_critical_fps_aggregate": (
                    float(np.mean([
                        float(item.get("hook_detector_actual_fps", 0.0))
                        for item in hook_episode_summaries
                    ])) if hook_episode_summaries else 0.0
                ),
                "full_video_suspended_during_hook_critical": bool(
                    self._hook_video_suspensions
                ),
                "full_video_suspension_episode_count": len(
                    self._hook_video_suspensions
                ),
                "full_video_suspensions": list(
                    self._hook_video_suspensions
                ),
                "stale_hook_frames_dropped": (
                    self._latest_hook_frame.stale_frames_dropped
                ),
                "hook_decision_trace_path": (
                    str(self._hook_decision_trace_path)
                    if self._hook_decision_trace_path is not None
                    else None
                ),
                "hook_decision_trace_rows": (
                    self._hook_decision_trace_rows_written
                ),
                "max_completed_cycles": self.live_config.max_completed_cycles,
                **self.cast_clearance.summary(),
                **self.cast_opportunity.summary(),
                **self.collect_retry.summary(),
                **self._press_live_emission.summary(),
                **self.console.summary(),
                **action_summary,
                **evidence_summary,
                **hook_roi_clip_summary,
                **press_shadow_summary,
                **press_v3_summary,
                **press_anomaly_summary,
                "preflight_passed": self._preflight_passed,
                "preflight_failure_reason": self._preflight_failure_reason,
                "preflight_failure_message": self._preflight_failure_message,
                "python_integrity": dict(
                    action_summary.get("integrity_diagnostics", {}).get(
                        "python_process", {}
                    )
                ),
                "target_integrity": dict(
                    action_summary.get("integrity_diagnostics", {}).get(
                        "target_process", {}
                    )
                ),
                "suspected_integrity_mismatch": action_summary.get(
                    "integrity_diagnostics", {}
                ).get("suspected_integrity_mismatch"),
                "foreground_unavailable_count": max(
                    int(action_summary.get("foreground_unavailable_count", 0)),
                    int(self._capture_diagnostics.get(
                        "foreground_unavailable_count", 0
                    )),
                ),
            }
            summary["action_attempted_count"] = sum(
                action_summary.get("attempted_action_counts", {}).values()
            )
            summary["action_applied_count"] = sum(
                action_summary.get("applied_action_counts", {}).values()
            )
            summary["action_rejected_count"] = sum(
                action_summary.get("rejected_action_counts", {}).values()
            )
            summary["action_partial_count"] = sum(
                action_summary.get("partial_action_counts", {}).values()
            )
            summary["action_failed_count"] = sum(
                action_summary.get("failed_action_counts", {}).values()
            )
            summary["logging_disabled"] = self.logger.logging_disabled
            summary["logging_failure_reason"] = (
                self.logger.logging_failure_reason
            )
            if self.live_config.runtime_profile == "production":
                for diagnostic_key in (
                    "video_path", "actual_video_path",
                    "detector_evidence_path", "event_windows_path",
                    "hook_roi_clip_path", "hook_roi_clip_index_path",
                    "press_roi_clip_path", "press_roi_frames_path",
                    "press_decision_trace_path", "press_review_items_path",
                    "hook_decision_trace_path",
                    "press_v3_debug_evidence_path",
                ):
                    summary.pop(diagnostic_key, None)
            self.logger.finalize(summary)
        return summary
