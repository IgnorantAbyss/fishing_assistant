from __future__ import annotations

from dataclasses import dataclass

from src.fishing_v2.domain.action_intent import ActionIntent
from src.fishing_v2.fusion.observation_fusion import ObservationFusion, StateEvidence
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.ports.action_sink import ActionSink
from src.fishing_v2.runtime.fishing_fsm import FSMResult, FishingFSM
from src.fishing_v2.runtime.safety_policy import SafetyDecision, SafetyPolicy, SafetyResult


@dataclass(frozen=True)
class ControllerResult:
    evidence: StateEvidence
    fsm: FSMResult
    safety: SafetyResult


class RuntimeController:
    def __init__(
        self,
        fusion: ObservationFusion,
        fsm: FishingFSM,
        safety: SafetyPolicy,
        *,
        action_sink: ActionSink | None = None,
    ) -> None:
        self.fusion = fusion
        self.fsm = fsm
        self.safety = safety
        self.action_sink = action_sink
        self._last_action_at: float | None = None
        self._sent: set[tuple[str, ActionIntent]] = set()

    def process(self, bundle: ObservationBundle, *, foreground: bool | None = None) -> ControllerResult:
        evidence = self.fusion.fuse(bundle, self.fsm.state)
        fsm_result = self.fsm.advance(evidence, bundle.timestamp)
        key = (fsm_result.next_state.value, fsm_result.action_request.intent)
        safety = self.safety.evaluate(
            fsm_result.action_request,
            fsm_result.next_state,
            evidence,
            foreground=foreground,
            already_sent=key in self._sent,
            elapsed_since_action=(None if self._last_action_at is None else bundle.timestamp - self._last_action_at),
        )
        if (
            safety.decision == SafetyDecision.ALLOW
            and self.safety.config.emit_actions
            and self.action_sink is not None
        ):
            self.action_sink.emit(fsm_result.action_request)
            self._sent.add(key)
            self._last_action_at = bundle.timestamp
        return ControllerResult(evidence, fsm_result, safety)
