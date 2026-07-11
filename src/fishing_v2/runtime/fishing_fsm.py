"""Action-aware hybrid fishing FSM with specialized-panel confirmation."""

from __future__ import annotations

from dataclasses import dataclass

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.observations import PromptObservationKind
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import StateEvidence
from src.fishing_v2.fusion.transition_policy import TransitionPolicy
from src.fishing_v2.perception.observation_bundle import ObservationBundle


@dataclass(frozen=True)
class FSMConfig:
    stable_frames: int = 2
    cast_pending_timeout_sec: float = 4.0
    hook_pending_timeout_sec: float = 3.0
    result_pending_timeout_sec: float = 5.0
    collect_pending_timeout_sec: float = 4.0
    sync_lost_timeout_sec: float = 2.0
    hook_safe_zone_start: float = 0.65
    hook_safe_zone_end: float = 0.85
    get_retry_interval_seconds: float = 0.4
    get_max_attempts: int = 12
    get_max_duration_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.hook_safe_zone_start < self.hook_safe_zone_end <= 1.0:
            raise ValueError("HOOK safe zone must be an ordered ratio within 0..1")
        if not 0.3 <= self.get_retry_interval_seconds <= 0.5:
            raise ValueError("GET retry interval must remain within the reviewed 0.3..0.5 second range")
        if self.get_max_attempts < 1 or self.get_max_duration_seconds <= 0:
            raise ValueError("GET retry limits must be positive")


@dataclass(frozen=True)
class FSMResult:
    previous_state: RuntimeState
    next_state: RuntimeState
    action_request: ActionRequest
    transition_reason: str
    changed: bool
    telemetry: tuple[str, ...] = ()


class FishingFSM:
    def __init__(
        self,
        config: FSMConfig | None = None,
        *,
        initial_state: RuntimeState = RuntimeState.SYNCING,
        initial_timestamp: float = 0.0,
    ) -> None:
        self.config = config or FSMConfig()
        self.state = initial_state
        self.state_since = float(initial_timestamp)
        self._candidate: RuntimeState | None = None
        self._candidate_frames = 0
        self._conflict_since: float | None = None
        self._actions_sent: set[tuple[RuntimeState, ActionIntent]] = set()
        self._press_action_proposed = False
        self._press_waiting_for_clear = False
        self._get_started_at: float | None = initial_timestamp if initial_state == RuntimeState.GET else None
        self._get_last_attempt_at: float | None = None
        self._get_attempts = 0
        self.policy = TransitionPolicy()

    @staticmethod
    def _none(reason: str = "no_action") -> ActionRequest:
        return ActionRequest(ActionIntent.NONE, 0.0, reason)

    def _held(self, previous: RuntimeState, reason: str, telemetry: tuple[str, ...] = ()) -> FSMResult:
        return FSMResult(previous, previous, self._none(reason), reason, False, telemetry)

    def force_state(self, state: RuntimeState, timestamp: float, reason: str = "explicit_override") -> FSMResult:
        previous = self.state
        self.state = state
        self.state_since = float(timestamp)
        self._candidate = None
        self._candidate_frames = 0
        self._conflict_since = None
        if state == RuntimeState.GET:
            self._reset_get_retry(timestamp)
        return FSMResult(previous, state, self._none(), reason, previous != state)

    def _transition(self, target: RuntimeState, timestamp: float, reason: str) -> FSMResult:
        previous = self.state
        self.policy.require_legal(previous, target)
        self.state = target
        self.state_since = float(timestamp)
        self._candidate = None
        self._candidate_frames = 0
        self._conflict_since = None
        if target == RuntimeState.GET:
            self._reset_get_retry(timestamp)
        if target == RuntimeState.IDLE and previous != RuntimeState.IDLE:
            self._actions_sent.clear()
            self._press_action_proposed = False
            self._press_waiting_for_clear = False
        return FSMResult(previous, target, self._none(), reason, previous != target)

    def _emit_once(
        self,
        intent: ActionIntent,
        confidence: float,
        reason: str,
        *,
        payload: dict | None = None,
    ) -> ActionRequest:
        key = (self.state, intent)
        if key in self._actions_sent:
            return self._none("action_already_proposed_in_state")
        self._actions_sent.add(key)
        return ActionRequest(intent, confidence, reason, payload or {})

    def _timeout(self, timestamp: float) -> float:
        return float(timestamp) - self.state_since

    def _reset_get_retry(self, timestamp: float) -> None:
        self._get_started_at = float(timestamp)
        self._get_last_attempt_at = None
        self._get_attempts = 0

    @staticmethod
    def _prompt_kind(bundle: ObservationBundle | None) -> PromptObservationKind:
        return bundle.prompt.kind if bundle and bundle.prompt else PromptObservationKind.UNKNOWN

    @staticmethod
    def _specialized_confirmed(target: RuntimeState, bundle: ObservationBundle | None) -> bool:
        if target == RuntimeState.HOOK:
            return bool(bundle and bundle.hook and bundle.hook.detected)
        if target == RuntimeState.PRESS:
            return bool(bundle and bundle.press and bundle.press.detected)
        if target == RuntimeState.GET:
            return bool(bundle and bundle.get and bundle.get.detected)
        return True

    def _stable_transition(self, target: RuntimeState, timestamp: float, reason: str) -> FSMResult | None:
        if target == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = target
            self._candidate_frames = 1
        if self._candidate_frames < max(1, self.config.stable_frames):
            return None
        return self._transition(target, timestamp, reason)

    def _advance_get(
        self, evidence: StateEvidence, timestamp: float, bundle: ObservationBundle | None
    ) -> FSMResult:
        previous = self.state
        panel = bundle.get if bundle else None
        if panel is not None and not panel.detected:
            return self._transition(RuntimeState.COLLECT_PENDING, timestamp, "get_panel_disappeared")
        if panel is None or not panel.detected:
            return self._held(previous, "waiting_for_get_panel_observation")
        started = self._get_started_at if self._get_started_at is not None else self.state_since
        duration = float(timestamp) - started
        if duration >= self.config.get_max_duration_seconds:
            return self._transition(RuntimeState.SYNC_REQUIRED, timestamp, "get_retry_duration_exceeded")
        if self._get_attempts >= self.config.get_max_attempts:
            return self._transition(RuntimeState.SYNC_REQUIRED, timestamp, "get_retry_attempts_exceeded")
        due = (
            self._get_last_attempt_at is None
            or float(timestamp) - self._get_last_attempt_at >= self.config.get_retry_interval_seconds
        )
        if not due:
            return self._held(previous, "get_retry_interval_not_elapsed")
        self._get_attempts += 1
        self._get_last_attempt_at = float(timestamp)
        action = ActionRequest(
            ActionIntent.COLLECT,
            max(evidence.confidence, panel.confidence),
            "get_panel_present_collect_retry",
            {
                "attempt": self._get_attempts,
                "max_attempts": self.config.get_max_attempts,
                "elapsed_seconds": duration,
                "max_duration_seconds": self.config.get_max_duration_seconds,
            },
        )
        return FSMResult(previous, previous, action, "get_collect_retry_due", False)

    def advance(
        self,
        evidence: StateEvidence,
        timestamp: float,
        bundle: ObservationBundle | None = None,
    ) -> FSMResult:
        previous = self.state
        failed_telemetry = ("FAILED",) if "FAILED" in evidence.reason.upper() else ()
        if self.state == RuntimeState.SYNC_REQUIRED:
            return self._held(previous, "sync_required_blocks_actions", failed_telemetry)

        timeout_limits = {
            RuntimeState.CAST_PENDING: self.config.cast_pending_timeout_sec,
            RuntimeState.HOOK_PENDING: self.config.hook_pending_timeout_sec,
            RuntimeState.RESULT_PENDING: self.config.result_pending_timeout_sec,
            RuntimeState.COLLECT_PENDING: self.config.collect_pending_timeout_sec,
        }
        if self.state in timeout_limits and self._timeout(timestamp) >= timeout_limits[self.state]:
            return self._transition(RuntimeState.SYNC_REQUIRED, timestamp, f"{self.state.value.lower()}_timeout")

        if self.state == RuntimeState.GET:
            return self._advance_get(evidence, timestamp, bundle)

        # A visible GET panel has priority over every prompt, including IDLE_CAST.
        if bundle and bundle.get and bundle.get.detected and self.policy.is_legal(self.state, RuntimeState.GET):
            moved = self._stable_transition(RuntimeState.GET, timestamp, "get_panel_priority")
            return moved or self._held(previous, "get_panel_candidate_not_stable")

        if self.state == RuntimeState.PRESS:
            panel = bundle.press if bundle else None
            if self._press_action_proposed:
                self._press_waiting_for_clear = bool(panel and panel.detected)
                return self._transition(RuntimeState.RESULT_PENDING, timestamp, "press_action_proposed")
            if panel and panel.detected and panel.sequence:
                action = self._emit_once(
                    ActionIntent.PRESS_SEQUENCE,
                    max(evidence.confidence, panel.confidence),
                    "press_panel_sequence_confirmed",
                    payload={"sequence": panel.sequence},
                )
                self._press_action_proposed = action.intent == ActionIntent.PRESS_SEQUENCE
                return FSMResult(previous, previous, action, "press_sequence_ready", False)
            return self._held(previous, "press_panel_active_awaiting_sequence", failed_telemetry)

        target = evidence.recommended_state
        if self.state == RuntimeState.WAITING and target == RuntimeState.IDLE:
            # Residual/single-frame IDLE hints never pull WAITING backward.
            target = None
        if target in {RuntimeState.HOOK, RuntimeState.PRESS, RuntimeState.GET} and not self._specialized_confirmed(target, bundle):
            target = None
        if self.state == RuntimeState.RESULT_PENDING and target == RuntimeState.IDLE:
            if self._prompt_kind(bundle) != PromptObservationKind.IDLE_CAST:
                target = None
        if self.state == RuntimeState.RESULT_PENDING and self._press_waiting_for_clear:
            if bundle and bundle.press and bundle.press.detected and target == RuntimeState.PRESS:
                target = None
            elif bundle and bundle.press and not bundle.press.detected:
                self._press_waiting_for_clear = False

        if evidence.has_conflict or (target is not None and not self.policy.is_legal(self.state, target)):
            self._conflict_since = self._conflict_since or float(timestamp)
            if float(timestamp) - self._conflict_since >= self.config.sync_lost_timeout_sec:
                if self.policy.is_legal(self.state, RuntimeState.SYNC_REQUIRED):
                    return self._transition(RuntimeState.SYNC_REQUIRED, timestamp, "persistent_conflicting_or_illegal_evidence")
            return self._held(previous, "conflict_not_stable", failed_telemetry)
        self._conflict_since = None

        if target is not None and target != self.state:
            moved = self._stable_transition(target, timestamp, f"stable_{evidence.reason}")
            if moved is None:
                return self._held(previous, "candidate_not_stable", failed_telemetry)
            if target == RuntimeState.PRESS:
                panel = bundle.press if bundle else None
                action = self._none("press_panel_has_no_sequence")
                if panel and panel.sequence:
                    action = self._emit_once(
                        ActionIntent.PRESS_SEQUENCE,
                        max(evidence.confidence, panel.confidence),
                        "press_panel_sequence_confirmed",
                        payload={"sequence": panel.sequence},
                    )
                    self._press_action_proposed = action.intent == ActionIntent.PRESS_SEQUENCE
                return FSMResult(moved.previous_state, moved.next_state, action, moved.transition_reason, True)
            return moved
        self._candidate = None
        self._candidate_frames = 0

        prompt = self._prompt_kind(bundle)
        if self.state == RuntimeState.IDLE:
            get_guard_passed = bool(bundle and bundle.get is not None and not bundle.get.detected)
            if prompt == PromptObservationKind.IDLE_CAST and get_guard_passed:
                action = self._emit_once(ActionIntent.CAST, evidence.confidence, "idle_cast_prompt_and_get_guard_passed")
                if action.intent != ActionIntent.NONE:
                    moved = self._transition(RuntimeState.CAST_PENDING, timestamp, "cast_intent_proposed")
                    return FSMResult(previous, moved.next_state, action, moved.transition_reason, True)
            return self._held(previous, "idle_waiting_for_prompt_or_get_guard", failed_telemetry)

        if self.state == RuntimeState.READY:
            if prompt == PromptObservationKind.READY_BITE:
                action = self._emit_once(ActionIntent.START_HOOK, evidence.confidence, "ready_bite_confirmed")
                if action.intent != ActionIntent.NONE:
                    moved = self._transition(RuntimeState.HOOK_PENDING, timestamp, "start_hook_intent_proposed")
                    return FSMResult(previous, moved.next_state, action, moved.transition_reason, True)
            return self._held(previous, "ready_waiting_for_confirmed_bite", failed_telemetry)

        if self.state == RuntimeState.HOOK:
            hook = bundle.hook if bundle else None
            position = hook.fill_ratio if hook and hook.detected else None
            if position is not None and self.config.hook_safe_zone_start <= position <= self.config.hook_safe_zone_end:
                action = self._emit_once(
                    ActionIntent.HOOK_ACTION,
                    max(evidence.confidence, hook.confidence),
                    "hook_cursor_entered_configured_safe_zone",
                    payload={
                        "position_ratio": position,
                        "safe_zone_start": self.config.hook_safe_zone_start,
                        "safe_zone_end": self.config.hook_safe_zone_end,
                    },
                )
                if action.intent != ActionIntent.NONE:
                    moved = self._transition(RuntimeState.RESULT_PENDING, timestamp, "hook_action_proposed")
                    return FSMResult(previous, moved.next_state, action, moved.transition_reason, True)
            return self._held(previous, "hook_waiting_for_configured_safe_zone", failed_telemetry)

        return self._held(previous, "state_held", failed_telemetry)
