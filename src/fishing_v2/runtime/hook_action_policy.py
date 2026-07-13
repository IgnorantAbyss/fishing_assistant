"""Hook action timing derived from qualified bar geometry, never global state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.fishing_v2.domain.observations import HookObservation


@dataclass(frozen=True)
class HookActionPolicyConfig:
    divider_safety_margin_px: int = 10
    fallback_trigger_threshold: float = 0.70

    def __post_init__(self) -> None:
        if self.divider_safety_margin_px < 0:
            raise ValueError("Hook divider safety margin must be non-negative pixels")
        if not 0.0 < self.fallback_trigger_threshold <= 1.0:
            raise ValueError("Hook fallback trigger threshold must be within (0, 1]")


@dataclass(frozen=True)
class HookActionDecision:
    divider_line_detected: bool
    divider_line_x: float | None
    divider_confidence: float
    fill_endpoint_x: float | None
    fill_ratio: float | None
    divider_crossed: bool
    divider_margin_passed: bool
    fallback_used: bool
    fallback_reason: str | None
    action_ready: bool
    one_shot_guard_result: str
    reason: str

    def payload(self) -> dict[str, Any]:
        return asdict(self)


class HookActionPolicy:
    def __init__(self, config: HookActionPolicyConfig | None = None) -> None:
        self.config = config or HookActionPolicyConfig()

    def evaluate(
        self,
        observation: HookObservation | None,
        *,
        action_already_proposed: bool,
    ) -> HookActionDecision:
        evidence = observation.evidence if observation else {}
        features = set(evidence.get("matched_features", ()))
        fill_ratio = observation.fill_ratio if observation is not None else None
        raw_valid_fill = bool(
            observation
            and observation.detected
            and "bar_fill" in features
            and fill_ratio is not None
            and fill_ratio > 0.0
        )
        explicit_geometry = evidence.get("crossing_geometry_version") == 1
        explicit_divider_valid = bool(
            explicit_geometry
            and evidence.get("divider_line_detected")
            and evidence.get("divider_line_x") is not None
            and float(evidence.get("divider_confidence", 0.0)) > 0.0
        )
        explicit_fill_x = evidence.get("fill_endpoint_x") if explicit_geometry else None
        valid_fill = bool(
            observation
            and observation.detected
            and (
                explicit_fill_x is not None
                or (not explicit_divider_valid and raw_valid_fill)
            )
        )
        bbox = evidence.get("bar_bbox")
        valid_bbox = bool(
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and float(bbox[2]) > float(bbox[0])
        )
        divider_ratio = observation.divider_ratio if observation else None
        divider_valid = bool(
            not explicit_geometry
            and raw_valid_fill
            and valid_bbox
            and "divider_line" in features
            and divider_ratio is not None
            and 0.0 <= divider_ratio <= 1.0
        )
        fill_x = float(explicit_fill_x) if explicit_fill_x is not None else None
        divider_x = float(evidence["divider_line_x"]) if explicit_divider_valid else None
        divider_confidence = float(evidence.get("divider_confidence", 0.0)) if explicit_divider_valid else 0.0
        divider_valid = explicit_divider_valid or divider_valid
        divider_crossed = margin_passed = False
        if not explicit_divider_valid and raw_valid_fill and valid_bbox:
            left, _, right, _ = (float(value) for value in bbox)
            width = right - left
            fill_x = left + float(fill_ratio) * width
            if divider_valid:
                divider_x = left + float(divider_ratio) * width
                divider_confidence = float(evidence.get("divider_confidence", 1.0))
        if valid_fill and divider_valid and fill_x is not None and divider_x is not None:
            divider_crossed = fill_x >= divider_x
            margin_passed = fill_x >= divider_x + self.config.divider_safety_margin_px

        fallback_used = bool(valid_fill and not divider_valid)
        fallback_reason = "divider_line_not_reliably_available" if fallback_used else None
        crossing_ready = bool(
            margin_passed if divider_valid
            else valid_fill and fill_ratio is not None
            and float(fill_ratio) >= self.config.fallback_trigger_threshold
        )
        action_ready = crossing_ready and not action_already_proposed
        if not observation or not observation.detected:
            reason = "qualified_hook_bar_not_detected"
        elif not valid_fill:
            reason = "valid_positive_bar_fill_required"
        elif action_already_proposed:
            reason = "hook_action_already_proposed_in_episode"
        elif divider_valid and not divider_crossed:
            reason = "fill_has_not_crossed_divider"
        elif divider_valid and not margin_passed:
            reason = "fill_crossed_divider_but_margin_pending"
        elif divider_valid:
            reason = "divider_margin_passed"
        elif crossing_ready:
            reason = "fallback_threshold_passed"
        else:
            reason = "fallback_threshold_pending"
        return HookActionDecision(
            divider_line_detected=divider_valid,
            divider_line_x=divider_x,
            divider_confidence=divider_confidence,
            fill_endpoint_x=fill_x,
            fill_ratio=fill_ratio,
            divider_crossed=divider_crossed,
            divider_margin_passed=margin_passed,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            action_ready=action_ready,
            one_shot_guard_result=(
                "blocked_already_proposed" if action_already_proposed else "not_previously_proposed"
            ),
            reason=reason,
        )
