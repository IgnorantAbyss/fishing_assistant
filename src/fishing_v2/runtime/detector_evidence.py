"""Qualification boundary between diagnostic raw detector output and Fusion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from src.fishing_v2.domain.observations import GetObservation, HookObservation, PressObservation
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.detector_activation import (
    DetectorActivationMode,
    DetectorActivationSnapshot,
)
from src.fishing_v2.runtime.press_sequence_aggregator import (
    PressSequenceAggregationConfig,
    PressSequenceTemporalAggregator,
)


class HookEvidenceKind(str, Enum):
    REJECTED = "REJECTED"
    RECTANGLE_CANDIDATE = "RECTANGLE_CANDIDATE"
    ACTIVE_HOOK_BAR = "ACTIVE_HOOK_BAR"


class PressEvidenceKind(str, Enum):
    REJECTED = "REJECTED"
    PRESS_PANEL_CANDIDATE = "PRESS_PANEL_CANDIDATE"
    PRESS_PANEL_PRESENT = "PRESS_PANEL_PRESENT"
    PRESS_SEQUENCE_READY = "PRESS_SEQUENCE_READY"


@dataclass(frozen=True)
class EvidenceQualification:
    detector: str
    activation_mode: DetectorActivationMode
    raw_detected: bool
    qualified_detected: bool
    qualification_reason: str
    used_by_fusion: bool
    diagnostic_only: bool
    hook_evidence_kind: HookEvidenceKind | None = None
    press_evidence_kind: PressEvidenceKind | None = None
    sequence_ready: bool = False
    sequence_qualification_reason: str | None = None


@dataclass(frozen=True)
class QualifiedObservationBundle:
    bundle: ObservationBundle
    hook: EvidenceQualification
    press: EvidenceQualification
    get: EvidenceQualification


@dataclass(frozen=True)
class EvidenceQualificationConfig:
    hook_strong_confidence: float = 0.85
    press_strong_confidence: float = 0.85
    get_strong_confidence: float = 0.85
    press_panel_confirmation_frames: int = 2
    press_panel_geometry_tolerance: float = 0.12
    press_sequence_window_frames: int = 5
    press_sequence_consensus_frames: int = 3
    press_per_key_min_aggregated_confidence: float = 0.68
    press_sequence_min_aggregated_confidence: float = 0.68


class DetectorEvidenceQualifier:
    def __init__(self, config: EvidenceQualificationConfig | None = None) -> None:
        self.config = config or EvidenceQualificationConfig()
        self.press_aggregator = PressSequenceTemporalAggregator(
            PressSequenceAggregationConfig(
                panel_confirmation_frames=self.config.press_panel_confirmation_frames,
                panel_geometry_tolerance=self.config.press_panel_geometry_tolerance,
                sequence_window_frames=self.config.press_sequence_window_frames,
                sequence_consensus_frames=self.config.press_sequence_consensus_frames,
                per_key_min_aggregated_confidence=self.config.press_per_key_min_aggregated_confidence,
                sequence_min_aggregated_confidence=self.config.press_sequence_min_aggregated_confidence,
            )
        )

    def qualify(
        self,
        raw: ObservationBundle,
        activation: DetectorActivationSnapshot,
    ) -> QualifiedObservationBundle:
        hook_observation, hook = self._hook(raw.hook, activation.hook)
        press_observation, press = self._press(raw.press, activation.press)
        get_observation, get = self._simple(
            "get", raw.get, activation.get, self.config.get_strong_confidence
        )
        return QualifiedObservationBundle(
            ObservationBundle(
                raw.frame_index,
                raw.timestamp,
                raw.prompt,
                hook_observation,
                press_observation,
                get_observation,
            ),
            hook,
            press,
            get,
        )

    def qualify_hook(
        self,
        observation: HookObservation | None,
        mode: DetectorActivationMode,
    ) -> tuple[HookObservation | None, EvidenceQualification]:
        return self._hook(observation, mode)

    def _press(
        self,
        observation: PressObservation | None,
        mode: DetectorActivationMode,
    ) -> tuple[PressObservation | None, EvidenceQualification]:
        if observation is None:
            self.press_aggregator.reset()
            return None, EvidenceQualification(
                "press", mode, False, False, "raw_not_detected", False,
                mode == DetectorActivationMode.OFF,
                press_evidence_kind=PressEvidenceKind.REJECTED,
                sequence_qualification_reason="panel_not_present",
            )
        versioned = observation.evidence.get("press_evidence_version") == 2
        if not versioned:
            sanitized, qualification = self._simple(
                "press", observation, mode, self.config.press_strong_confidence
            )
            if isinstance(sanitized, PressObservation) and sanitized.sequence:
                sanitized = replace(
                    sanitized,
                    panel_candidate=sanitized.detected,
                    panel_present=sanitized.detected,
                    key_box_count=len(sanitized.sequence),
                    stable_key_box_count=len(sanitized.sequence),
                    sequence_candidate=sanitized.sequence,
                    sequence_ready=sanitized.detected,
                    sequence_confidence=sanitized.confidence,
                    sequence_qualification_reason="legacy_ready_sequence",
                )
            return sanitized, replace(
                qualification,
                press_evidence_kind=(
                    PressEvidenceKind.PRESS_SEQUENCE_READY
                    if sanitized and sanitized.detected and sanitized.sequence
                    else PressEvidenceKind.PRESS_PANEL_PRESENT
                    if sanitized and sanitized.detected
                    else PressEvidenceKind.REJECTED
                ),
                sequence_ready=bool(sanitized and sanitized.detected and sanitized.sequence),
                sequence_qualification_reason=(
                    "legacy_ready_sequence" if sanitized and sanitized.detected and sanitized.sequence
                    else "legacy_sequence_not_ready"
                ),
            )

        raw_detected = bool(observation.panel_present)
        diagnostic_only = mode == DetectorActivationMode.OFF
        if diagnostic_only:
            self.press_aggregator.reset()
            kind = (
                PressEvidenceKind.PRESS_PANEL_CANDIDATE
                if observation.panel_candidate else PressEvidenceKind.REJECTED
            )
            sanitized = replace(
                observation,
                detected=False,
                panel_present=False,
                stable_key_box_count=0,
                sequence=(),
                sequence_ready=False,
                sequence_qualification_reason="activation_off_diagnostic_only",
            )
            return sanitized, EvidenceQualification(
                "press", mode, raw_detected, False,
                "activation_off_diagnostic_only", False, True,
                press_evidence_kind=kind,
                sequence_ready=False,
                sequence_qualification_reason="activation_off_diagnostic_only",
            )

        aggregation = self.press_aggregator.update(observation)
        strong_panel = observation.confidence >= self.config.press_strong_confidence
        qualified = bool(aggregation.panel_confirmed and observation.panel_present and strong_panel)
        sequence_ready = bool(qualified and aggregation.sequence_ready)
        if sequence_ready:
            kind = PressEvidenceKind.PRESS_SEQUENCE_READY
            reason = "press_panel_and_sequence_qualified"
        elif qualified:
            kind = PressEvidenceKind.PRESS_PANEL_PRESENT
            reason = "press_panel_structurally_qualified"
        elif observation.panel_candidate:
            kind = PressEvidenceKind.PRESS_PANEL_CANDIDATE
            reason = (
                "press_panel_below_strong_confidence"
                if aggregation.panel_confirmed and not strong_panel
                else aggregation.qualification_reason
            )
        else:
            kind = PressEvidenceKind.REJECTED
            reason = observation.panel_qualification_reason or "raw_not_detected"
        sanitized = replace(
            observation,
            detected=qualified,
            panel_present=qualified,
            stable_key_box_count=aggregation.stable_key_box_count,
            sequence_candidate=aggregation.sequence_candidate,
            sequence=aggregation.sequence_candidate if sequence_ready else (),
            sequence_ready=sequence_ready,
            sequence_confidence=aggregation.sequence_confidence,
            sequence_qualification_reason=aggregation.qualification_reason,
            evidence={
                **observation.evidence,
                "stable_panel_frames": aggregation.stable_panel_frames,
                "per_key_aggregated_confidence": list(aggregation.per_key_confidence),
                "selected_clean_frame": aggregation.selected_clean_frame,
            },
        )
        return sanitized, EvidenceQualification(
            "press", mode, raw_detected, qualified, reason,
            qualified, False,
            press_evidence_kind=kind,
            sequence_ready=sequence_ready,
            sequence_qualification_reason=aggregation.qualification_reason,
        )

    def _hook(
        self,
        observation: HookObservation | None,
        mode: DetectorActivationMode,
    ) -> tuple[HookObservation | None, EvidenceQualification]:
        raw_detected = bool(observation and observation.detected)
        features = set(observation.evidence.get("matched_features", ())) if observation else set()
        fill = observation.fill_ratio if observation else None
        diagnostic_only = mode == DetectorActivationMode.OFF
        kind = HookEvidenceKind.REJECTED
        reason = "raw_not_detected"
        qualified = False
        if raw_detected and ("bar_fill" not in features or fill is None or fill <= 0):
            kind = HookEvidenceKind.RECTANGLE_CANDIDATE
            reason = "rectangle_only_or_nonpositive_fill"
        elif raw_detected:
            kind = HookEvidenceKind.ACTIVE_HOOK_BAR
            if diagnostic_only:
                reason = "activation_off_diagnostic_only"
            elif observation.confidence < self.config.hook_strong_confidence:
                reason = "active_bar_below_strong_confidence"
            else:
                reason = "active_bar_fill_qualified"
                qualified = True
        if diagnostic_only and not qualified:
            reason = "activation_off_diagnostic_only:" + reason
        sanitized = replace(observation, detected=qualified) if observation is not None else None
        return sanitized, EvidenceQualification(
            "hook", mode, raw_detected, qualified, reason,
            qualified, diagnostic_only, hook_evidence_kind=kind,
        )

    @staticmethod
    def _simple(
        detector: str,
        observation: PressObservation | GetObservation | None,
        mode: DetectorActivationMode,
        strong_confidence: float,
    ) -> tuple[PressObservation | GetObservation | None, EvidenceQualification]:
        raw_detected = bool(observation and observation.detected)
        diagnostic_only = mode == DetectorActivationMode.OFF
        qualified = bool(
            raw_detected
            and not diagnostic_only
            and observation is not None
            and observation.confidence >= strong_confidence
        )
        if diagnostic_only:
            reason = "activation_off_diagnostic_only"
        elif not raw_detected:
            reason = "raw_not_detected"
        elif observation is not None and observation.confidence < strong_confidence:
            reason = "below_strong_confidence"
        else:
            reason = "strong_specialized_evidence_qualified"
        sanitized = replace(observation, detected=qualified) if observation is not None else None
        return sanitized, EvidenceQualification(
            detector, mode, raw_detected, qualified, reason,
            qualified, diagnostic_only,
        )
