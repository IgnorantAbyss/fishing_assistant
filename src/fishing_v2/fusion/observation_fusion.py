"""Rule-based evidence fusion without detector priority overwrite."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from src.fishing_v2.domain.observations import PromptObservationKind
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.perception.observation_bundle import ObservationBundle


@dataclass(frozen=True)
class FusionConfig:
    hook_strong_threshold: float = 0.85
    press_strong_threshold: float = 0.85
    get_strong_threshold: float = 0.85
    prompt_min_confidence: float = 0.80
    conflict_timeout_sec: float = 2.0


@dataclass(frozen=True)
class StateEvidence:
    candidate_states: Mapping[RuntimeState, float]
    supporting_observations: tuple[str, ...]
    conflicting_observations: tuple[str, ...]
    recommended_state: RuntimeState | None
    confidence: float
    reason: str
    frame_index: int
    timestamp: float

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicting_observations)


class ObservationFusion:
    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()

    def fuse(self, bundle: ObservationBundle, current: RuntimeState) -> StateEvidence:
        scores: dict[RuntimeState, float] = {}
        support: list[str] = []
        conflicts: list[str] = []

        def offer(state: RuntimeState, score: float, source: str) -> None:
            scores[state] = max(scores.get(state, 0.0), float(score))
            support.append(f"{source}:{score:.3f}")

        special: list[tuple[RuntimeState, float, str, float]] = []
        if bundle.hook and bundle.hook.detected:
            special.append((RuntimeState.HOOK, bundle.hook.confidence, "hook", self.config.hook_strong_threshold))
        if bundle.press and bundle.press.detected:
            special.append((RuntimeState.PRESS, bundle.press.confidence, "press", self.config.press_strong_threshold))
        if bundle.get and bundle.get.detected:
            special.append((RuntimeState.GET, bundle.get.confidence, "get", self.config.get_strong_threshold))
        for state, confidence, source, _ in special:
            bonus = 0.08 if state == current else 0.0
            offer(state, min(1.0, confidence + bonus), source)

        prompt_state: RuntimeState | None = None
        prompt = bundle.prompt
        prompt_mapping = {
            PromptObservationKind.IDLE_CAST: RuntimeState.IDLE,
            PromptObservationKind.WAITING_IN_PROGRESS: RuntimeState.WAITING,
            PromptObservationKind.READY_BITE: RuntimeState.READY,
        }
        if prompt is not None:
            prompt_state = prompt_mapping.get(prompt.kind)
            if prompt_state is not None and prompt.confidence >= self.config.prompt_min_confidence:
                offer(prompt_state, prompt.confidence, f"prompt_{prompt.kind.value}")
            elif prompt.kind == PromptObservationKind.UNKNOWN:
                support.append(f"prompt_{prompt.kind.value}:non_state_evidence")
            elif prompt.kind in {
                PromptObservationKind.HOOK_INSTRUCTION,
                PromptObservationKind.PRESS_INSTRUCTION,
            }:
                support.append(f"prompt_{prompt.kind.value}:activation_hint_only")

        strong_special = [item for item in special if item[1] >= item[3]]
        if strong_special:
            priority = {RuntimeState.GET: 3, RuntimeState.PRESS: 2, RuntimeState.HOOK: 1}
            state, confidence, source, _ = max(
                strong_special, key=lambda item: (priority[item[0]], item[1])
            )
            if prompt_state is not None and prompt_state != state:
                compatible = (
                    state == RuntimeState.GET
                    or (state == RuntimeState.HOOK and prompt.kind == PromptObservationKind.READY_BITE)
                )
                if compatible:
                    support.append(f"compatible_residual_prompt:{prompt.kind.value}")
                else:
                    conflicts.append(f"prompt_{prompt.kind.value}_vs_{source}")
            elif prompt is not None and (
                (state == RuntimeState.HOOK and prompt.kind == PromptObservationKind.HOOK_INSTRUCTION)
                or (state == RuntimeState.PRESS and prompt.kind == PromptObservationKind.PRESS_INSTRUCTION)
            ):
                support.append(f"compatible_activation_hint:{prompt.kind.value}")
            return StateEvidence(
                scores, tuple(support), tuple(conflicts), state, confidence,
                f"strong_{source}_evidence", bundle.frame_index, bundle.timestamp,
            )

        if not scores:
            return StateEvidence(
                scores, tuple(support), tuple(conflicts), None, 0.0,
                "insufficient_observation_evidence_no_fallback", bundle.frame_index, bundle.timestamp,
            )
        recommended, confidence = max(scores.items(), key=lambda item: item[1])
        if recommended != current and current not in {
            RuntimeState.SYNCING, RuntimeState.SYNC_REQUIRED,
        }:
            support.append(f"transition_candidate:{current.value}_to_{recommended.value}")
        return StateEvidence(
            scores, tuple(support), tuple(conflicts), recommended, confidence,
            "highest_supported_candidate", bundle.frame_index, bundle.timestamp,
        )
