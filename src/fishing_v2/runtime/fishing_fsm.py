"""Action-aware hybrid fishing FSM with explicit propose/apply causality."""

from __future__ import annotations

from dataclasses import dataclass

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.observations import HookObservation, PromptObservationKind
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import StateEvidence
from src.fishing_v2.fusion.transition_policy import TransitionPolicy
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.hook_action_policy import (
    HookActionDecision,
    HookActionPolicy,
    HookActionPolicyConfig,
)


@dataclass(frozen=True)
class FSMConfig:
    stable_frames: int = 2
    prompt_min_confidence: float = 0.80
    cast_pending_timeout_sec: float = 4.0
    hook_pending_timeout_sec: float = 3.0
    result_pending_timeout_sec: float = 5.0
    collect_pending_timeout_sec: float = 4.0
    sync_lost_timeout_sec: float = 2.0
    hook_divider_safety_margin_px: int = 10
    hook_fallback_trigger_threshold: float = 0.70
    hook_episode_timeout_sec: float = 12.0
    hook_disappearance_frames_required: int = 2
    get_retry_interval_seconds: float = 0.4
    get_max_attempts: int = 12
    get_max_duration_seconds: float = 5.0
    recorded_press_exit_idle_frames: int = 4
    result_minimum_pending_sec: float = 1.5
    result_maximum_pending_sec: float = 10.0

    def __post_init__(self) -> None:
        HookActionPolicyConfig(
            self.hook_divider_safety_margin_px,
            self.hook_fallback_trigger_threshold,
        )
        if self.hook_episode_timeout_sec <= 0:
            raise ValueError("Hook episode timeout must be positive")
        if not 0.0 <= self.prompt_min_confidence <= 1.0:
            raise ValueError("Prompt minimum confidence must be within 0..1")
        if self.hook_disappearance_frames_required < 1:
            raise ValueError("Hook disappearance frames must be positive")
        if not 0.3 <= self.get_retry_interval_seconds <= 0.5:
            raise ValueError("GET retry interval must remain within the reviewed 0.3..0.5 second range")
        if self.get_max_attempts < 1 or self.get_max_duration_seconds <= 0:
            raise ValueError("GET retry limits must be positive")
        if self.recorded_press_exit_idle_frames < 2:
            raise ValueError("recorded_press_exit_idle_frames must be at least two")
        if self.result_minimum_pending_sec <= 0:
            raise ValueError("result_minimum_pending_sec must be positive")
        if self.result_maximum_pending_sec <= self.result_minimum_pending_sec:
            raise ValueError("result_maximum_pending_sec must exceed the minimum grace")


@dataclass(frozen=True)
class FSMResult:
    previous_state: RuntimeState
    next_state: RuntimeState
    action_request: ActionRequest
    transition_reason: str
    changed: bool
    telemetry: tuple[str, ...] = ()
    visual_acknowledgement: str | None = None


@dataclass(frozen=True)
class ActionCommitResult:
    action_applied: bool
    previous_state: RuntimeState
    next_state: RuntimeState
    reason: str


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
        self._actions_applied: set[tuple[RuntimeState, ActionIntent]] = set()
        self._cast_opportunities_applied: set[str] = set()
        self._start_hook_opportunity_id: str | None = None
        self._start_hook_consumed_opportunity_id: str | None = None
        self._authoritative_physical_idle_id: str | None = None
        self._pending_request: ActionRequest | None = None
        self._press_waiting_for_clear = False
        self._press_intent_proposed = False
        self._hook_intent_proposed = False
        self._hook_episode_active = initial_state == RuntimeState.HOOK
        self._hook_episode_started_at = float(initial_timestamp) if self._hook_episode_active else None
        self._hook_absent_frames = 0
        self._last_hook_action_decision: HookActionDecision | None = None
        self._get_started_at: float | None = initial_timestamp if initial_state == RuntimeState.GET else None
        self._get_last_applied_at: float | None = None
        self._get_attempts = 0
        self.policy = TransitionPolicy()
        self.hook_action_policy = HookActionPolicy(HookActionPolicyConfig(
            self.config.hook_divider_safety_margin_px,
            self.config.hook_fallback_trigger_threshold,
        ))

    @property
    def actions_applied(self) -> frozenset[tuple[RuntimeState, ActionIntent]]:
        return frozenset(self._actions_applied)

    @property
    def cast_opportunities_applied(self) -> frozenset[str]:
        return frozenset(self._cast_opportunities_applied)

    @property
    def start_hook_opportunity_id(self) -> str | None:
        return self._start_hook_opportunity_id

    @property
    def start_hook_consumed_opportunity_id(self) -> str | None:
        return self._start_hook_consumed_opportunity_id

    @property
    def last_hook_action_decision(
        self,
    ) -> HookActionDecision | None:
        return self._last_hook_action_decision

    @property
    def hook_episode_active(self) -> bool:
        return self._hook_episode_active

    @property
    def pending_request(self) -> ActionRequest | None:
        """Expose proposal presence for lifecycle diagnostics only."""
        return self._pending_request

    @staticmethod
    def _none(reason: str = "no_action") -> ActionRequest:
        return ActionRequest(ActionIntent.NONE, 0.0, reason)

    def _held(
        self,
        previous: RuntimeState,
        reason: str,
        telemetry: tuple[str, ...] = (),
        visual_acknowledgement: str | None = None,
    ) -> FSMResult:
        return FSMResult(
            previous, previous, self._none(reason), reason, False,
            telemetry, visual_acknowledgement,
        )

    def force_state(self, state: RuntimeState, timestamp: float, reason: str = "explicit_override") -> FSMResult:
        previous = self.state
        self.state = state
        self.state_since = float(timestamp)
        self._candidate = None
        self._candidate_frames = 0
        self._conflict_since = None
        self._pending_request = None
        self._press_intent_proposed = False
        self._hook_intent_proposed = False
        self._hook_episode_active = state == RuntimeState.HOOK
        self._hook_episode_started_at = float(timestamp) if self._hook_episode_active else None
        self._hook_absent_frames = 0
        if state == RuntimeState.GET:
            self._reset_get_retry(timestamp)
        return FSMResult(previous, state, self._none(), reason, previous != state)

    def recover_to_authoritative_idle(
        self,
        timestamp: float,
        *,
        physical_idle_id: str,
        reason: str = "authoritative_idle_prompt_recovery",
    ) -> FSMResult:
        """Commit a certified physical-IDLE boundary, not a state-pointer move.

        A repeated certificate for the same physical IDLE episode is
        intentionally idempotent.  Only a new certificate-owned identity may
        close the legacy state-owned CAST marker left by an older episode.
        Per-opportunity CAST identities remain retained for session-wide
        exactly-once protection.
        """
        resolved_physical_idle_id = str(physical_idle_id).strip()
        if not resolved_physical_idle_id:
            raise ValueError(
                "Authoritative IDLE recovery requires a physical idle identity"
            )
        previous = self.state
        if (
            self._authoritative_physical_idle_id
            == resolved_physical_idle_id
            and self.state == RuntimeState.IDLE
        ):
            return self._held(
                previous,
                "authoritative_idle_episode_already_open",
            )

        # Close proposal and state-owned legacy CAST ownership before opening
        # the newly certified physical IDLE episode.  Do not clear the
        # opportunity-owned set: an already emitted CAST must stay consumed.
        self._pending_request = None
        self._actions_applied.discard((RuntimeState.IDLE, ActionIntent.CAST))
        self._start_hook_opportunity_id = None
        result = self.force_state(RuntimeState.IDLE, timestamp, reason)
        self._authoritative_physical_idle_id = resolved_physical_idle_id
        return result

    def begin_sync_recovery(self, timestamp: float) -> None:
        """Clear all episode/action state before accepting recovery evidence."""
        if self.state != RuntimeState.SYNC_REQUIRED:
            raise RuntimeError("Sync recovery can only begin from SYNC_REQUIRED")
        self.state_since = float(timestamp)
        self._candidate = None
        self._candidate_frames = 0
        self._conflict_since = None
        self._pending_request = None
        self._actions_applied.clear()
        self._start_hook_opportunity_id = None
        self._start_hook_consumed_opportunity_id = None
        self._press_waiting_for_clear = False
        self._press_intent_proposed = False
        self._hook_intent_proposed = False
        self._hook_episode_active = False
        self._hook_episode_started_at = None
        self._hook_absent_frames = 0
        self._get_started_at = None
        self._get_last_applied_at = None
        self._get_attempts = 0

    def recover_from_sync_required(
        self,
        state: RuntimeState,
        timestamp: float,
        reason: str,
    ) -> FSMResult:
        """Apply an independently qualified recovery without relaxing FSM edges."""
        if self.state != RuntimeState.SYNC_REQUIRED:
            raise RuntimeError("Runtime is not awaiting synchronization")
        concrete_states = {
            RuntimeState.IDLE,
            RuntimeState.WAITING,
            RuntimeState.READY,
            RuntimeState.HOOK,
            RuntimeState.PRESS,
            RuntimeState.GET,
        }
        if state not in concrete_states:
            raise ValueError("Recovery target must be a concrete runtime state")
        return self.force_state(state, timestamp, reason)

    def rearm_hook_action_opportunity(self, timestamp: float) -> bool:
        """Reopen only the current HOOK proposal latch before any emission."""
        if self.state != RuntimeState.HOOK:
            return False
        if (RuntimeState.HOOK, ActionIntent.HOOK_ACTION) in self._actions_applied:
            return False
        self._pending_request = None
        self._hook_intent_proposed = False
        self._hook_episode_active = True
        self._hook_episode_started_at = float(timestamp)
        self._hook_absent_frames = 0
        return True

    def return_hook_stall_to_sync_required(
        self,
        timestamp: float,
        *,
        reason: str = "hook_action_stall_without_current_hook_evidence",
    ) -> FSMResult:
        if self.state != RuntimeState.HOOK:
            raise RuntimeError("Hook stall recovery requires HOOK state")
        return self._transition(
            RuntimeState.SYNC_REQUIRED,
            timestamp,
            reason,
        )

    def recover_missed_ready(
        self,
        timestamp: float,
        *,
        reason: str,
    ) -> FSMResult:
        """Acknowledge a user START_HOOK that occurred between prompt samples."""
        if self.state != RuntimeState.WAITING:
            raise RuntimeError(
                "Missed READY recovery is only valid from WAITING"
            )
        return self._transition(
            RuntimeState.HOOK_PENDING,
            timestamp,
            reason,
            visual_acknowledgement="HOOK_INSTRUCTION",
        )

    def reconcile_start_hook_opportunity(self, opportunity_id: str) -> bool:
        """Bind READY proposal ownership to one physical READY certificate.

        Legacy state/action history remains available for diagnostics, but it
        cannot suppress a distinct physical READY opportunity.  Only the
        current and most recently consumed identities are retained.
        """
        resolved = str(opportunity_id).strip()
        if self.state != RuntimeState.READY or not resolved:
            return False
        if self._start_hook_opportunity_id == resolved:
            return False
        self._pending_request = None
        self._start_hook_opportunity_id = resolved
        return True

    def _transition(
        self,
        target: RuntimeState,
        timestamp: float,
        reason: str,
        *,
        visual_acknowledgement: str | None = None,
    ) -> FSMResult:
        previous = self.state
        self.policy.require_legal(previous, target)
        self.state = target
        self.state_since = float(timestamp)
        self._candidate = None
        self._candidate_frames = 0
        self._conflict_since = None
        if target == RuntimeState.GET:
            self._reset_get_retry(timestamp)
        if target == RuntimeState.PRESS and previous != RuntimeState.PRESS:
            self._press_intent_proposed = False
        elif previous == RuntimeState.PRESS and target != RuntimeState.PRESS:
            self._press_intent_proposed = False
        if target == RuntimeState.HOOK and previous != RuntimeState.HOOK:
            self._hook_intent_proposed = False
            self._hook_episode_active = True
            self._hook_episode_started_at = float(timestamp)
            self._hook_absent_frames = 0
        elif previous == RuntimeState.HOOK and target != RuntimeState.HOOK:
            self._hook_episode_active = False
            self._hook_episode_started_at = None
            self._hook_absent_frames = 0
        if target == RuntimeState.IDLE and previous != RuntimeState.IDLE:
            self._actions_applied.clear()
            self._authoritative_physical_idle_id = None
            self._press_waiting_for_clear = False
            self._hook_intent_proposed = False
        return FSMResult(
            previous, target, self._none(), reason, previous != target,
            visual_acknowledgement=visual_acknowledgement,
        )

    def _propose(
        self,
        intent: ActionIntent,
        confidence: float,
        reason: str,
        *,
        payload: dict | None = None,
    ) -> ActionRequest:
        if intent == ActionIntent.PRESS_SEQUENCE and self._press_intent_proposed:
            return self._none("press_sequence_already_proposed_in_episode")
        if intent == ActionIntent.HOOK_ACTION and self._hook_intent_proposed:
            return self._none("hook_action_already_proposed_in_episode")
        resolved_payload = dict(payload or {})
        key = (self.state, intent)
        if intent == ActionIntent.START_HOOK:
            opportunity_id = self._start_hook_opportunity_id
            if opportunity_id is not None:
                if opportunity_id == self._start_hook_consumed_opportunity_id:
                    return self._none(
                        "start_hook_opportunity_already_consumed"
                    )
                resolved_payload["start_hook_opportunity_id"] = (
                    opportunity_id
                )
            elif key in self._actions_applied:
                return self._none("action_already_applied_in_state")
        elif key in self._actions_applied and intent != ActionIntent.COLLECT:
            return self._none("action_already_applied_in_state")
        request = ActionRequest(intent, confidence, reason, resolved_payload)
        self._pending_request = request
        if intent == ActionIntent.PRESS_SEQUENCE:
            self._press_intent_proposed = True
        elif intent == ActionIntent.HOOK_ACTION:
            self._hook_intent_proposed = True
            self._hook_episode_active = False
            self._hook_episode_started_at = None
            self._hook_absent_frames = 0
        return request

    def _refresh_hook_episode_latch(
        self,
        hook: HookObservation | None,
        timestamp: float,
    ) -> None:
        if not self._hook_episode_active:
            return
        if (
            self._hook_episode_started_at is not None
            and float(timestamp) - self._hook_episode_started_at >= self.config.hook_episode_timeout_sec
        ):
            self._hook_episode_active = False
            self._hook_episode_started_at = None
            self._hook_absent_frames = 0
            return
        evidence = hook.evidence if hook is not None else {}
        current_bar_present = bool(
            hook
            and (
                hook.detected
                or (
                    evidence.get("crossing_geometry_version") == 1
                    and evidence.get("divider_line_detected")
                )
            )
        )
        self._hook_absent_frames = 0 if current_bar_present else self._hook_absent_frames + 1
        if self._hook_absent_frames >= self.config.hook_disappearance_frames_required:
            self._hook_episode_active = False
            self._hook_episode_started_at = None

    def _hook_action_request(
        self,
        evidence: StateEvidence,
        timestamp: float,
        bundle: ObservationBundle | None,
        hook_action_observation: HookObservation | None = None,
    ) -> tuple[ActionRequest, str]:
        hook = (
            hook_action_observation
            if hook_action_observation is not None
            else bundle.hook if bundle else None
        )
        self._refresh_hook_episode_latch(hook, timestamp)
        decision = self.hook_action_policy.evaluate(
            hook,
            action_already_proposed=self._hook_intent_proposed,
            hook_episode_active=self._hook_episode_active,
        )
        self._last_hook_action_decision = decision
        if not decision.action_ready:
            return self._none(decision.reason), decision.reason
        assert hook is not None
        return (
            self._propose(
                ActionIntent.HOOK_ACTION,
                max(evidence.confidence, hook.confidence),
                "hook_fill_safely_crossed_threshold",
                payload=decision.payload(),
            ),
            "hook_action_proposed",
        )

    def discard_proposal(self) -> None:
        self._pending_request = None

    def stage_external_press_sequence(
        self,
        request: ActionRequest,
    ) -> bool:
        """Stage a verified ACTIVE-frame PRESS proposal for external commit."""
        if (
            self.state != RuntimeState.PRESS
            or request.intent != ActionIntent.PRESS_SEQUENCE
            or (
                self._pending_request is not None
                and self._pending_request.intent
                != ActionIntent.PRESS_SEQUENCE
            )
        ):
            return False
        self._pending_request = request
        self._press_intent_proposed = True
        return True

    def stage_external_cast(self, request: ActionRequest) -> bool:
        """Stage a certified IDLE lifecycle CAST for the normal commit path."""
        physical_idle_id = str(
            request.payload.get("physical_idle_id", "")
        ).strip()
        if (
            self.state != RuntimeState.IDLE
            or request.intent != ActionIntent.CAST
            or (
                self._authoritative_physical_idle_id is not None
                and physical_idle_id
                != self._authoritative_physical_idle_id
            )
            or (
                self._pending_request is not None
                and self._pending_request.intent != ActionIntent.CAST
            )
        ):
            return False
        self._pending_request = request
        return True

    @staticmethod
    def _cast_opportunity_identity(request: ActionRequest) -> str | None:
        if request.intent != ActionIntent.CAST:
            return None
        value = request.payload.get("cast_opportunity_id")
        if value is None:
            return None
        resolved = str(value).strip()
        return resolved or None

    @staticmethod
    def _start_hook_opportunity_identity(
        request: ActionRequest,
    ) -> str | None:
        if request.intent != ActionIntent.START_HOOK:
            return None
        value = request.payload.get("start_hook_opportunity_id")
        if value is None:
            return None
        resolved = str(value).strip()
        return resolved or None

    def commit_action(self, request: ActionRequest, timestamp: float) -> ActionCommitResult:
        previous = self.state
        if request.intent == ActionIntent.NONE or self._pending_request != request:
            return ActionCommitResult(False, previous, previous, "no_matching_proposed_action")
        key = (self.state, request.intent)
        cast_opportunity_id = self._cast_opportunity_identity(request)
        start_hook_opportunity_id = (
            self._start_hook_opportunity_identity(request)
        )
        duplicate_cast_opportunity = bool(
            cast_opportunity_id is not None
            and cast_opportunity_id in self._cast_opportunities_applied
        )
        duplicate_legacy_action = bool(
            cast_opportunity_id is None
            and start_hook_opportunity_id is None
            and key in self._actions_applied
            and request.intent != ActionIntent.COLLECT
        )
        duplicate_start_hook_opportunity = bool(
            start_hook_opportunity_id is not None
            and start_hook_opportunity_id
            == self._start_hook_consumed_opportunity_id
        )
        if (
            duplicate_cast_opportunity
            or duplicate_start_hook_opportunity
            or duplicate_legacy_action
        ):
            self._pending_request = None
            return ActionCommitResult(
                False,
                previous,
                previous,
                (
                    "cast_opportunity_already_applied"
                    if duplicate_cast_opportunity
                    else (
                        "start_hook_opportunity_already_consumed"
                        if duplicate_start_hook_opportunity
                        else "action_already_applied_in_state"
                    )
                ),
            )
        if cast_opportunity_id is not None:
            self._cast_opportunities_applied.add(cast_opportunity_id)
        elif start_hook_opportunity_id is not None:
            self._start_hook_consumed_opportunity_id = (
                start_hook_opportunity_id
            )
            self._actions_applied.add(key)
        else:
            self._actions_applied.add(key)
        self._pending_request = None
        target = {
            ActionIntent.CAST: RuntimeState.CAST_PENDING,
            ActionIntent.START_HOOK: RuntimeState.HOOK_PENDING,
            ActionIntent.HOOK_ACTION: RuntimeState.RESULT_PENDING,
            ActionIntent.PRESS_SEQUENCE: RuntimeState.RESULT_PENDING,
        }.get(request.intent)
        if request.intent == ActionIntent.COLLECT:
            self._get_attempts += 1
            self._get_last_applied_at = float(timestamp)
            return ActionCommitResult(True, previous, previous, "collect_action_applied")
        if target is None or not self.policy.is_legal(self.state, target):
            return ActionCommitResult(False, previous, previous, "action_not_applicable_in_current_state")
        if request.intent == ActionIntent.PRESS_SEQUENCE:
            self._press_waiting_for_clear = True
        moved = self._transition(target, timestamp, f"{request.intent.value.lower()}_action_applied")
        return ActionCommitResult(True, previous, moved.next_state, moved.transition_reason)

    def _timeout(self, timestamp: float) -> float:
        return float(timestamp) - self.state_since

    def _reset_get_retry(self, timestamp: float) -> None:
        self._get_started_at = float(timestamp)
        self._get_last_applied_at = None
        self._get_attempts = 0

    @staticmethod
    def _prompt_kind(bundle: ObservationBundle | None) -> PromptObservationKind:
        return bundle.prompt.kind if bundle and bundle.prompt else PromptObservationKind.UNKNOWN

    @staticmethod
    def _specialized_confirmed(target: RuntimeState, bundle: ObservationBundle | None) -> bool:
        observation = {
            RuntimeState.HOOK: bundle.hook if bundle else None,
            RuntimeState.PRESS: bundle.press if bundle else None,
            RuntimeState.GET: bundle.get if bundle else None,
        }.get(target)
        return bool(observation and observation.detected) if target in {
            RuntimeState.HOOK, RuntimeState.PRESS, RuntimeState.GET,
        } else True

    def _stable_transition(
        self,
        target: RuntimeState,
        timestamp: float,
        reason: str,
        *,
        visual_acknowledgement: str | None = None,
    ) -> FSMResult | None:
        if target == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = target
            self._candidate_frames = 1
        if self._candidate_frames < max(1, self.config.stable_frames):
            return None
        return self._transition(
            target, timestamp, reason,
            visual_acknowledgement=visual_acknowledgement,
        )

    def clear_transition_candidate(self) -> None:
        """Discard stale stability support after a scheduler cancellation."""
        self._candidate = None
        self._candidate_frames = 0

    def _advance_get(
        self,
        evidence: StateEvidence,
        timestamp: float,
        bundle: ObservationBundle | None,
        *,
        recorded_observation: bool,
    ) -> FSMResult:
        previous = self.state
        panel = bundle.get if bundle else None
        if panel is not None and not panel.detected:
            return self._transition(
                RuntimeState.COLLECT_PENDING,
                timestamp,
                "get_panel_disappeared",
                visual_acknowledgement="qualified_get_panel_disappeared",
            )
        if panel is None or not panel.detected:
            return self._held(previous, "waiting_for_get_panel_observation")
        started = self._get_started_at if self._get_started_at is not None else self.state_since
        duration = float(timestamp) - started
        if not recorded_observation and duration >= self.config.get_max_duration_seconds:
            return self._transition(RuntimeState.SYNC_REQUIRED, timestamp, "get_retry_duration_exceeded")
        if not recorded_observation and self._get_attempts >= self.config.get_max_attempts:
            return self._transition(RuntimeState.SYNC_REQUIRED, timestamp, "get_retry_attempts_exceeded")
        due = recorded_observation or (
            self._get_last_applied_at is None
            or float(timestamp) - self._get_last_applied_at >= self.config.get_retry_interval_seconds
        )
        if not due:
            return self._held(previous, "get_retry_interval_not_elapsed")
        action = self._propose(
            ActionIntent.COLLECT,
            max(evidence.confidence, panel.confidence),
            "get_panel_present_collect_retry",
            payload={
                "attempt": self._get_attempts + 1,
                "max_attempts": self.config.get_max_attempts,
                "elapsed_seconds": duration,
                "max_duration_seconds": self.config.get_max_duration_seconds,
            },
        )
        return FSMResult(previous, previous, action, "get_collect_proposed", False)

    def advance(
        self,
        evidence: StateEvidence,
        timestamp: float,
        bundle: ObservationBundle | None = None,
        *,
        recorded_observation: bool = False,
        hook_action_observation: HookObservation | None = None,
    ) -> FSMResult:
        previous = self.state
        self._pending_request = None
        self._last_hook_action_decision = None
        failed_telemetry = ("FAILED",) if "FAILED" in evidence.reason.upper() else ()
        if self.state == RuntimeState.SYNC_REQUIRED:
            return self._held(previous, "sync_required_blocks_actions", failed_telemetry)
        if self.state == RuntimeState.SYNCING:
            return self._held(previous, "startup_synchronizer_owns_transition", failed_telemetry)

        if self.state == RuntimeState.GET:
            return self._advance_get(
                evidence, timestamp, bundle,
                recorded_observation=recorded_observation,
            )

        prompt = self._prompt_kind(bundle)

        # A qualified GET panel has priority over every prompt, including
        # recorded PRESS_INSTRUCTION and IDLE_CAST acknowledgements.
        if bundle and bundle.get and bundle.get.detected and self.policy.is_legal(self.state, RuntimeState.GET):
            return self._transition(
                RuntimeState.GET,
                timestamp,
                "get_panel_priority",
                visual_acknowledgement="qualified_get_panel_present",
            )

        # RESULT_PENDING is shared by the short HOOK->PRESS hand-off and the
        # final result. Qualified PRESS must therefore remain eligible during
        # the grace window. All other exits wait for result evidence to settle.
        qualified_press_pending = bool(
            self.state == RuntimeState.RESULT_PENDING
            and bundle and bundle.press and bundle.press.detected
        )
        if self.state == RuntimeState.RESULT_PENDING and not qualified_press_pending:
            result_elapsed = self._timeout(timestamp)
            banner_present = bool(
                bundle and bundle.result_banner and bundle.result_banner.detected
            )
            if result_elapsed >= self.config.result_maximum_pending_sec:
                return self._transition(
                    RuntimeState.SYNC_REQUIRED,
                    timestamp,
                    "result_pending_maximum_timeout",
                )
            if banner_present:
                return self._held(
                    previous,
                    "result_banner_holds_result_pending",
                    failed_telemetry,
                    visual_acknowledgement="RESULT_BANNER_PRESENT",
                )
            if result_elapsed < self.config.result_minimum_pending_sec:
                return self._held(
                    previous, "result_pending_minimum_grace", failed_telemetry
                )

        press_panel = bundle.press if bundle else None
        press_disappeared = bool(
            press_panel
            and press_panel.evidence.get("panel_disappeared") is True
        )
        if (
            self.state == RuntimeState.PRESS
            and press_panel is not None
            and not press_panel.detected
            and (
                press_disappeared if recorded_observation
                else not press_panel.panel_candidate
            )
        ):
            return self._transition(
                RuntimeState.RESULT_PENDING,
                timestamp,
                "press_panel_disappeared",
                visual_acknowledgement="qualified_press_panel_disappeared",
            )

        if recorded_observation:
            if self.state == RuntimeState.READY and prompt == PromptObservationKind.HOOK_INSTRUCTION:
                return self._transition(
                    RuntimeState.HOOK_PENDING,
                    timestamp,
                    "recorded_hook_instruction_acknowledgement",
                    visual_acknowledgement="HOOK_INSTRUCTION",
                )
            if self.state == RuntimeState.HOOK and prompt in {
                PromptObservationKind.PRESS_INSTRUCTION,
                PromptObservationKind.IDLE_CAST,
            }:
                return self._transition(
                    RuntimeState.RESULT_PENDING,
                    timestamp,
                    "recorded_hook_result_prompt_acknowledgement",
                    visual_acknowledgement=prompt.value,
                )
            if (
                self.state == RuntimeState.PRESS
                and prompt == PromptObservationKind.IDLE_CAST
                and press_disappeared
            ):
                return self._transition(
                    RuntimeState.RESULT_PENDING,
                    timestamp,
                    "recorded_press_result_prompt_acknowledgement",
                    visual_acknowledgement=prompt.value,
                )

        # A threshold-qualified HOOK_INSTRUCTION is the visual acknowledgement
        # for a completed START_HOOK. During the READY-to-HOOK hand-off it must
        # clear transient prompt conflict before the generic conflict timer can
        # force SYNC_REQUIRED. It is still only an activation hint: transition
        # to HOOK and HOOK_ACTION remain gated by qualified Hook evidence.
        if (
            recorded_observation
            and self.state == RuntimeState.HOOK_PENDING
            and prompt == PromptObservationKind.HOOK_INSTRUCTION
            and bundle is not None
            and bundle.prompt is not None
            and bundle.prompt.confidence
            >= self.config.prompt_min_confidence
        ):
            self._conflict_since = None
            if evidence.recommended_state != RuntimeState.HOOK:
                if (
                    self._timeout(timestamp)
                    >= self.config.hook_pending_timeout_sec
                ):
                    return self._transition(
                        RuntimeState.SYNC_REQUIRED,
                        timestamp,
                        "hook_pending_timeout",
                    )
                self._candidate = None
                self._candidate_frames = 0
                return self._held(
                    previous,
                    "hook_instruction_acknowledged_waiting_for_hook_evidence",
                    failed_telemetry,
                    visual_acknowledgement="HOOK_INSTRUCTION",
                )

        # A completed physical CAST has its own bounded visual acknowledgement
        # window. The old IDLE_CAST frame commonly remains on screen during the
        # cast animation and is not an illegal transition by itself. Strong
        # specialized evidence still bypasses this grace and follows the normal
        # conflict/safety path below.
        if (
            self.state == RuntimeState.CAST_PENDING
            and self._timeout(timestamp) < self.config.cast_pending_timeout_sec
            and prompt == PromptObservationKind.IDLE_CAST
            and evidence.recommended_state in {None, RuntimeState.IDLE}
            and not evidence.has_conflict
        ):
            self._candidate = None
            self._candidate_frames = 0
            self._conflict_since = None
            return self._held(previous, "cast_pending_visual_ack_grace")

        timeout_limits = {
            RuntimeState.CAST_PENDING: self.config.cast_pending_timeout_sec,
            RuntimeState.HOOK_PENDING: self.config.hook_pending_timeout_sec,
            RuntimeState.COLLECT_PENDING: self.config.collect_pending_timeout_sec,
        }
        if self.state in timeout_limits and self._timeout(timestamp) >= timeout_limits[self.state]:
            return self._transition(RuntimeState.SYNC_REQUIRED, timestamp, f"{self.state.value.lower()}_timeout")

        # Explicit same-frame divider/fill geometry is an action-specific
        # contract. Evaluate it before a stale semantic prompt can turn the
        # qualified Fusion recommendation into an illegal HOOK transition.
        # Safety still evaluates the resulting request normally.
        action_hook = (
            hook_action_observation
            if hook_action_observation is not None
            else bundle.hook if bundle is not None else None
        )
        hook_action_evidence = (
            action_hook.evidence
            if action_hook is not None else {}
        )
        explicit_hook_geometry_present = bool(
            self.state == RuntimeState.HOOK
            and action_hook is not None
            and hook_action_evidence.get("crossing_geometry_version") == 1
            and hook_action_evidence.get("divider_line_detected") is True
            and hook_action_evidence.get("divider_line_x") is not None
            and hook_action_evidence.get("fill_endpoint_x") is not None
        )
        if explicit_hook_geometry_present:
            action, _ = self._hook_action_request(
                evidence,
                timestamp,
                bundle,
                action_hook,
            )
            if action.intent == ActionIntent.HOOK_ACTION:
                return FSMResult(
                    previous,
                    previous,
                    action,
                    "hook_action_proposed",
                    False,
                )

        target = evidence.recommended_state
        if self.state == RuntimeState.WAITING and target == RuntimeState.IDLE:
            target = None
        if target in {RuntimeState.HOOK, RuntimeState.PRESS, RuntimeState.GET} and not self._specialized_confirmed(target, bundle):
            target = None
        if self.state == RuntimeState.RESULT_PENDING and target == RuntimeState.IDLE:
            if prompt != PromptObservationKind.IDLE_CAST:
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
            visual = None
            if target == RuntimeState.HOOK:
                visual = "qualified_active_hook_bar"
            elif target == RuntimeState.PRESS:
                visual = "qualified_press_panel"
            elif target == RuntimeState.GET:
                visual = "qualified_get_panel_present"
            elif bundle and bundle.prompt:
                visual = bundle.prompt.kind.value
            moved = self._stable_transition(
                target,
                timestamp,
                f"stable_{evidence.reason}",
                visual_acknowledgement=visual,
            )
            if moved is None:
                return self._held(previous, "candidate_not_stable", failed_telemetry)
            if target == RuntimeState.HOOK:
                action, _ = self._hook_action_request(
                    evidence,
                    timestamp,
                    bundle,
                    action_hook,
                )
                if action.intent == ActionIntent.HOOK_ACTION:
                    return FSMResult(
                        moved.previous_state,
                        moved.next_state,
                        action,
                        moved.transition_reason,
                        True,
                        visual_acknowledgement=(
                            moved.visual_acknowledgement
                        ),
                    )
            if target == RuntimeState.PRESS:
                panel = bundle.press if bundle else None
                action = self._none("press_panel_has_no_sequence")
                if panel and panel.sequence:
                    action = self._propose(
                        ActionIntent.PRESS_SEQUENCE,
                        max(evidence.confidence, panel.confidence),
                        "press_panel_sequence_confirmed",
                        payload={"sequence": panel.sequence},
                    )
                return FSMResult(
                    moved.previous_state, moved.next_state, action,
                    moved.transition_reason, True,
                    visual_acknowledgement=moved.visual_acknowledgement,
                )
            return moved
        self._candidate = None
        self._candidate_frames = 0

        if self.state == RuntimeState.IDLE:
            get_guard_passed = bool(bundle and bundle.get is not None and not bundle.get.detected)
            if prompt == PromptObservationKind.IDLE_CAST and get_guard_passed:
                action = self._propose(
                    ActionIntent.CAST,
                    evidence.confidence,
                    "idle_cast_prompt_and_get_guard_passed",
                )
                return FSMResult(previous, previous, action, "cast_intent_proposed", False)
            return self._held(previous, "idle_waiting_for_prompt_or_get_guard", failed_telemetry)

        if self.state == RuntimeState.READY:
            if prompt == PromptObservationKind.READY_BITE:
                action = self._propose(
                    ActionIntent.START_HOOK,
                    evidence.confidence,
                    "ready_bite_confirmed",
                )
                return FSMResult(previous, previous, action, "start_hook_intent_proposed", False)
            return self._held(previous, "ready_waiting_for_confirmed_bite", failed_telemetry)

        if self.state == RuntimeState.HOOK:
            action, reason = self._hook_action_request(
                evidence,
                timestamp,
                bundle,
                action_hook,
            )
            if action.intent == ActionIntent.HOOK_ACTION:
                return FSMResult(
                    previous,
                    previous,
                    action,
                    "hook_action_proposed",
                    False,
                )
            return self._held(previous, reason, failed_telemetry)

        if self.state == RuntimeState.PRESS:
            panel = bundle.press if bundle else None
            if panel and panel.detected and panel.sequence:
                action = self._propose(
                    ActionIntent.PRESS_SEQUENCE,
                    max(evidence.confidence, panel.confidence),
                    "press_panel_sequence_confirmed",
                    payload={"sequence": panel.sequence},
                )
                return FSMResult(previous, previous, action, "press_sequence_proposed", False)
            return self._held(previous, "press_panel_active_awaiting_sequence", failed_telemetry)

        return self._held(previous, "state_held", failed_telemetry)
