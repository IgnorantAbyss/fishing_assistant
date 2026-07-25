from __future__ import annotations

import math
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
        ActionIntent.CAST: {RuntimeState.IDLE},
        ActionIntent.START_HOOK: {RuntimeState.READY},
        ActionIntent.HOOK_ACTION: {RuntimeState.HOOK},
        ActionIntent.PRESS_SEQUENCE: {RuntimeState.PRESS},
        ActionIntent.COLLECT: {RuntimeState.GET},
    }

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def _hook_explicit_geometry_meets_confidence(
        self,
        request: ActionRequest,
    ) -> bool:
        if request.intent != ActionIntent.HOOK_ACTION:
            return False
        payload = request.payload
        required_flags = (
            "action_ready",
            "current_hook_geometry_is_usable",
            "divider_line_detected",
            "divider_margin_passed",
        )
        if not all(payload.get(name) is True for name in required_flags):
            return False
        numeric_values = (
            payload.get("divider_line_x"),
            payload.get("fill_endpoint_x"),
            payload.get("divider_confidence"),
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in numeric_values
        ):
            return False
        divider_x, fill_endpoint_x, divider_confidence = (
            float(value) for value in numeric_values
        )
        return bool(
            fill_endpoint_x >= divider_x
            and divider_confidence
            >= self.config.minimum_evidence_confidence
        )

    def evaluate(
        self,
        request: ActionRequest,
        state: RuntimeState,
        evidence: StateEvidence,
        *,
        foreground: bool | None,
        already_sent: bool,
        elapsed_since_action: float | None,
        runtime_environment_supported: bool = True,
        get_panel_present: bool | None = None,
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
        if not runtime_environment_supported:
            return SafetyResult(SafetyDecision.DENY, "unsupported_runtime_resolution")
        if foreground is False:
            return SafetyResult(SafetyDecision.DENY, "foreground_window_not_confirmed")
        if foreground is None:
            return SafetyResult(SafetyDecision.WAIT, "foreground_window_status_unknown")
        if evidence.has_conflict:
            return SafetyResult(SafetyDecision.WAIT, "conflicting_evidence")
        if (
            evidence.confidence < self.config.minimum_evidence_confidence
            and not self._hook_explicit_geometry_meets_confidence(request)
        ):
            return SafetyResult(SafetyDecision.WAIT, "evidence_confidence_too_low")
        if request.intent == ActionIntent.CAST:
            if get_panel_present is True:
                return SafetyResult(SafetyDecision.DENY, "get_panel_guard_cancelled_cast")
            if get_panel_present is None:
                return SafetyResult(SafetyDecision.WAIT, "get_panel_guard_not_observed")
        if request.intent == ActionIntent.COLLECT:
            if get_panel_present is not True:
                return SafetyResult(SafetyDecision.DENY, "collect_requires_visible_get_panel")
            attempt = int(request.payload.get("attempt", 0))
            maximum = int(request.payload.get("max_attempts", 0))
            elapsed = float(request.payload.get("elapsed_seconds", float("inf")))
            max_duration = float(request.payload.get("max_duration_seconds", 0.0))
            if attempt < 1 or maximum < 1 or attempt > maximum:
                return SafetyResult(SafetyDecision.DENY, "collect_attempt_limit_exceeded")
            if elapsed >= max_duration:
                return SafetyResult(SafetyDecision.DENY, "collect_duration_limit_exceeded")
        if elapsed_since_action is not None and elapsed_since_action < self.config.cooldown_sec:
            return SafetyResult(SafetyDecision.WAIT, "action_cooldown")
        if not self.config.emit_actions:
            return SafetyResult(SafetyDecision.WAIT, "action_emission_disabled")
        return SafetyResult(SafetyDecision.ALLOW, "safety_checks_passed")
