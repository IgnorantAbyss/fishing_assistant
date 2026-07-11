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


class HookEvidenceKind(str, Enum):
    REJECTED = "REJECTED"
    RECTANGLE_CANDIDATE = "RECTANGLE_CANDIDATE"
    ACTIVE_HOOK_BAR = "ACTIVE_HOOK_BAR"


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


class DetectorEvidenceQualifier:
    def __init__(self, config: EvidenceQualificationConfig | None = None) -> None:
        self.config = config or EvidenceQualificationConfig()

    def qualify(
        self,
        raw: ObservationBundle,
        activation: DetectorActivationSnapshot,
    ) -> QualifiedObservationBundle:
        hook_observation, hook = self._hook(raw.hook, activation.hook)
        press_observation, press = self._simple(
            "press", raw.press, activation.press, self.config.press_strong_confidence
        )
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
            qualified, diagnostic_only, kind,
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
