"""Action-aware, stateful fishing workflow FSM."""

from __future__ import annotations

from dataclasses import dataclass

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import StateEvidence
from src.fishing_v2.fusion.transition_policy import TransitionPolicy


@dataclass(frozen=True)
class FSMConfig:
    stable_frames: int = 2
    cast_pending_timeout_sec: float = 4.0
    hook_pending_timeout_sec: float = 3.0
    post_catch_timeout_sec: float = 5.0
    collect_pending_timeout_sec: float = 4.0
    sync_lost_timeout_sec: float = 2.0


@dataclass(frozen=True)
class FSMResult:
    previous_state: RuntimeState
    next_state: RuntimeState
    action_request: ActionRequest
    transition_reason: str
    changed: bool


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
        self.policy = TransitionPolicy()

    @staticmethod
    def _none(reason: str = "no_action") -> ActionRequest:
        return ActionRequest(ActionIntent.NONE, 0.0, reason)

    def force_state(self, state: RuntimeState, timestamp: float, reason: str = "explicit_override") -> FSMResult:
        previous = self.state
        self.state = state
        self.state_since = float(timestamp)
        self._candidate = None
        self._candidate_frames = 0
        self._conflict_since = None
        return FSMResult(previous, state, self._none(), reason, previous != state)

    def _transition(self, target: RuntimeState, timestamp: float, reason: str) -> FSMResult:
        previous = self.state
        self.policy.require_legal(previous, target)
        self.state = target
        self.state_since = float(timestamp)
        self._candidate = None
        self._candidate_frames = 0
        self._conflict_since = None
        return FSMResult(previous, target, self._none(), reason, previous != target)

    def _emit_once(self, intent: ActionIntent, confidence: float, reason: str) -> ActionRequest:
        key = (self.state, intent)
        if key in self._actions_sent:
            return self._none("action_already_proposed_in_state")
        self._actions_sent.add(key)
        return ActionRequest(intent, confidence, reason)

    def _timeout(self, timestamp: float) -> float:
        return float(timestamp) - self.state_since

    def advance(self, evidence: StateEvidence, timestamp: float) -> FSMResult:
        previous = self.state
        if self.state in {RuntimeState.STOPPED, RuntimeState.SYNC_REQUIRED}:
            return FSMResult(previous, previous, self._none("state_blocks_actions"), "state_blocks_transition", False)

        timeout_limits = {
            RuntimeState.CAST_PENDING: self.config.cast_pending_timeout_sec,
            RuntimeState.HOOK_PENDING: self.config.hook_pending_timeout_sec,
            RuntimeState.POST_CATCH: self.config.post_catch_timeout_sec,
            RuntimeState.COLLECT_PENDING: self.config.collect_pending_timeout_sec,
        }
        if self.state in timeout_limits and self._timeout(timestamp) >= timeout_limits[self.state]:
            return self._transition(RuntimeState.SYNC_REQUIRED, timestamp, f"{self.state.value.lower()}_timeout")

        target = evidence.recommended_state
        if self.state == RuntimeState.HOOK and target in {
            RuntimeState.IDLE, RuntimeState.WAITING, RuntimeState.READY,
        }:
            target = RuntimeState.POST_CATCH
        elif self.state == RuntimeState.PRESS and target in {
            RuntimeState.IDLE, RuntimeState.WAITING, RuntimeState.READY,
        }:
            target = RuntimeState.POST_CATCH
        if evidence.has_conflict or (target is not None and not self.policy.is_legal(self.state, target)):
            self._conflict_since = self._conflict_since or float(timestamp)
            if float(timestamp) - self._conflict_since >= self.config.sync_lost_timeout_sec:
                if self.policy.is_legal(self.state, RuntimeState.SYNC_REQUIRED):
                    return self._transition(RuntimeState.SYNC_REQUIRED, timestamp, "persistent_conflicting_or_illegal_evidence")
            return FSMResult(previous, previous, self._none("waiting_on_conflict"), "conflict_not_stable", False)
        self._conflict_since = None

        if target is not None and target != self.state:
            if target == self._candidate:
                self._candidate_frames += 1
            else:
                self._candidate = target
                self._candidate_frames = 1
            if self._candidate_frames < max(1, self.config.stable_frames):
                return FSMResult(previous, previous, self._none("candidate_not_stable"), "candidate_not_stable", False)
            result = self._transition(target, timestamp, f"stable_{evidence.reason}")
            if target == RuntimeState.PRESS:
                return FSMResult(result.previous_state, result.next_state, self._emit_once(ActionIntent.PRESS_SEQUENCE, evidence.confidence, "press_observation_stable"), result.transition_reason, True)
            return result
        self._candidate = None
        self._candidate_frames = 0

        if self.state == RuntimeState.IDLE:
            action = self._emit_once(ActionIntent.CAST, evidence.confidence, "idle_ready_to_cast")
            if action.intent != ActionIntent.NONE:
                moved = self._transition(RuntimeState.CAST_PENDING, timestamp, "cast_intent_proposed")
                return FSMResult(previous, moved.next_state, action, moved.transition_reason, True)
        if self.state == RuntimeState.READY:
            action = self._emit_once(ActionIntent.START_HOOK, evidence.confidence, "ready_to_start_hook")
            if action.intent != ActionIntent.NONE:
                moved = self._transition(RuntimeState.HOOK_PENDING, timestamp, "start_hook_intent_proposed")
                return FSMResult(previous, moved.next_state, action, moved.transition_reason, True)
        if self.state == RuntimeState.GET:
            action = self._emit_once(ActionIntent.COLLECT, evidence.confidence, "get_ready_to_collect")
            if action.intent != ActionIntent.NONE:
                moved = self._transition(RuntimeState.COLLECT_PENDING, timestamp, "collect_intent_proposed")
                return FSMResult(previous, moved.next_state, action, moved.transition_reason, True)
        return FSMResult(previous, previous, self._none(), "state_held", False)
