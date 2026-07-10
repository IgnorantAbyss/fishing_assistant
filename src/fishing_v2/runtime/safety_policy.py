from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import StateEvidence


class SafetyDecision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    WAIT = "WAIT"
    REQUIRE_SYNC = "REQUIRE_SYNC"


@dataclass(frozen=True)
class SafetyConfig:
    enabled: bool = True
    emit_actions: bool = False
    minimum_evidence_confidence: float = 0.8
    cooldown_sec: float = 0.25


@dataclass(frozen=True)
class SafetyResult:
    decision: SafetyDecision
    reason: str


class SafetyPolicy:
    ALLOWED_STATES = {
        ActionIntent.CAST: {RuntimeState.CAST_PENDING},
        ActionIntent.START_HOOK: {RuntimeState.HOOK_PENDING},
        ActionIntent.PRESS_SEQUENCE: {RuntimeState.PRESS},
        ActionIntent.COLLECT: {RuntimeState.COLLECT_PENDING},
        ActionIntent.PAUSE: set(RuntimeState),
        ActionIntent.STOP: set(RuntimeState),
    }

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def evaluate(
        self,
        request: ActionRequest,
        state: RuntimeState,
        evidence: StateEvidence,
        *,
        foreground: bool | None,
        already_sent: bool,
        elapsed_since_action: float | None,
    ) -> SafetyResult:
        if state == RuntimeState.SYNC_REQUIRED:
            return SafetyResult(SafetyDecision.DENY, "sync_required_blocks_actions")
        if state == RuntimeState.SYNCING:
            return SafetyResult(SafetyDecision.REQUIRE_SYNC, "startup_sync_incomplete")
        if request.intent == ActionIntent.NONE:
            return SafetyResult(SafetyDecision.WAIT, "no_action_intent")
        if request.intent not in self.ALLOWED_STATES or state not in self.ALLOWED_STATES[request.intent]:
            return SafetyResult(SafetyDecision.DENY, "action_not_allowed_in_runtime_state")
        if already_sent:
            return SafetyResult(SafetyDecision.DENY, "duplicate_action_in_state")
        if foreground is False:
            return SafetyResult(SafetyDecision.DENY, "foreground_window_not_confirmed")
        if foreground is None:
            return SafetyResult(SafetyDecision.WAIT, "foreground_window_status_unknown")
        if evidence.has_conflict:
            return SafetyResult(SafetyDecision.WAIT, "conflicting_evidence")
        if evidence.confidence < self.config.minimum_evidence_confidence:
            return SafetyResult(SafetyDecision.WAIT, "evidence_confidence_too_low")
        if elapsed_since_action is not None and elapsed_since_action < self.config.cooldown_sec:
            return SafetyResult(SafetyDecision.WAIT, "action_cooldown")
        if not self.config.emit_actions:
            return SafetyResult(SafetyDecision.WAIT, "action_emission_disabled")
        return SafetyResult(SafetyDecision.ALLOW, "safety_checks_passed")
