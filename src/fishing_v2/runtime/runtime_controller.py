from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.fusion.observation_fusion import ObservationFusion, StateEvidence
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.ports.action_sink import (
    ActionExecutionContext,
    ActionExecutionResult,
    ActionSink,
)
from src.fishing_v2.runtime.detector_activation import (
    DetectorActivationPolicy,
    DetectorActivationSnapshot,
)
from src.fishing_v2.runtime.detector_evidence import (
    DetectorEvidenceQualifier,
    QualifiedObservationBundle,
)
from src.fishing_v2.runtime.fishing_fsm import (
    ActionCommitResult,
    FSMResult,
    FishingFSM,
)
from src.fishing_v2.runtime.safety_policy import SafetyDecision, SafetyPolicy, SafetyResult


class ActionExecutionMode(str, Enum):
    STANDARD = "standard"
    RECORDED_OBSERVATION = "recorded_observation"


@dataclass(frozen=True)
class ControllerResult:
    evidence: StateEvidence
    fsm: FSMResult
    safety: SafetyResult
    activation: DetectorActivationSnapshot
    next_activation: DetectorActivationSnapshot
    qualified: QualifiedObservationBundle
    action_applied: bool
    action_commit: ActionCommitResult
    action_execution: ActionExecutionResult | None


class RuntimeController:
    def __init__(
        self,
        fusion: ObservationFusion,
        fsm: FishingFSM,
        safety: SafetyPolicy,
        *,
        action_sink: ActionSink | None = None,
        activation_policy: DetectorActivationPolicy | None = None,
        evidence_qualifier: DetectorEvidenceQualifier | None = None,
    ) -> None:
        self.fusion = fusion
        self.fsm = fsm
        self.safety = safety
        self.action_sink = action_sink
        self.activation_policy = activation_policy or DetectorActivationPolicy()
        self.evidence_qualifier = evidence_qualifier or DetectorEvidenceQualifier()
        self._last_action_at: float | None = None
        self._sent: set[tuple[str, ActionIntent, int]] = set()

    def reset_for_sync_recovery(self, timestamp: float) -> None:
        """Drop stale episode, qualifier and one-shot state at sync loss."""
        self.fsm.begin_sync_recovery(timestamp)
        self.evidence_qualifier.reset_temporal_state()
        self._sent.clear()
        self._last_action_at = None

    def qualify_raw_bundle(
        self,
        raw_bundle: ObservationBundle,
        *,
        action_mode: ActionExecutionMode = ActionExecutionMode.STANDARD,
    ) -> tuple[DetectorActivationSnapshot, QualifiedObservationBundle]:
        recorded = action_mode == ActionExecutionMode.RECORDED_OBSERVATION
        activation = self.activation_policy.evaluate(
            self.fsm.state,
            raw_bundle,
            recorded_observation=recorded,
        )
        return activation, self.evidence_qualifier.qualify(raw_bundle, activation)

    def process(
        self,
        raw_bundle: ObservationBundle,
        *,
        foreground: bool | None = None,
        runtime_environment_supported: bool | None = None,
        action_mode: ActionExecutionMode = ActionExecutionMode.STANDARD,
        preserve_proposal: bool = False,
    ) -> ControllerResult:
        recorded = action_mode == ActionExecutionMode.RECORDED_OBSERVATION
        activation, qualified = self.qualify_raw_bundle(
            raw_bundle,
            action_mode=action_mode,
        )
        bundle = qualified.bundle
        evidence = self.fusion.fuse(bundle, self.fsm.state)
        fsm_result = self.fsm.advance(
            evidence,
            bundle.timestamp,
            bundle,
            recorded_observation=recorded,
        )
        attempt = int(fsm_result.action_request.payload.get("attempt", 0))
        key = (fsm_result.previous_state.value, fsm_result.action_request.intent, attempt)
        get_panel_present = bundle.get.detected if bundle.get is not None else None
        safety = self.safety.evaluate(
            fsm_result.action_request,
            fsm_result.next_state,
            evidence,
            foreground=foreground,
            already_sent=key in self._sent,
            elapsed_since_action=(
                None if self._last_action_at is None
                else bundle.timestamp - self._last_action_at
            ),
            runtime_environment_supported=runtime_environment_supported is True,
            get_panel_present=get_panel_present,
        )
        applied = False
        execution: ActionExecutionResult | None = None
        commit = ActionCommitResult(
            False, self.fsm.state, self.fsm.state,
            "action_not_applied",
        )
        if (
            safety.decision == SafetyDecision.ALLOW
            and self.safety.config.emit_actions
            and self.action_sink is not None
            and not recorded
        ):
            context = ActionExecutionContext(
                action_id=(
                    f"runtime:{fsm_result.previous_state.value}:"
                    f"{fsm_result.action_request.intent.value}:{attempt}"
                ),
                episode_id=fsm_result.previous_state.value,
                requested_at=bundle.timestamp,
                capture_frame_index=bundle.frame_index,
                runtime_state=fsm_result.next_state.value,
                target_hwnd=None,
            )
            execution = self.action_sink.apply(fsm_result.action_request, context)
            if execution.applied:
                commit = self.fsm.commit_action(fsm_result.action_request, bundle.timestamp)
            applied = bool(execution.applied and commit.action_applied)
            if applied:
                self._sent.add(key)
                self._last_action_at = bundle.timestamp
                fsm_result = replace(
                    fsm_result,
                    next_state=commit.next_state,
                    transition_reason=commit.reason,
                    changed=commit.previous_state != commit.next_state,
                )
        if not applied and not preserve_proposal:
            self.fsm.discard_proposal()
        next_activation = self.activation_policy.evaluate(
            self.fsm.state,
            raw_bundle,
            recorded_observation=recorded,
        )
        return ControllerResult(
            evidence,
            fsm_result,
            safety,
            activation,
            next_activation,
            qualified,
            applied,
            commit,
            execution,
        )

    def commit_external_action(
        self,
        request: ActionRequest,
        timestamp: float,
        *,
        action_key: tuple[str, ActionIntent, int] | None = None,
    ) -> ActionCommitResult:
        """Commit an already-completed external sink result exactly once."""
        commit = self.fsm.commit_action(request, timestamp)
        if commit.action_applied:
            if action_key is not None:
                self._sent.add(action_key)
            self._last_action_at = float(timestamp)
        return commit

    def discard_external_proposal(self) -> None:
        self.fsm.discard_proposal()
