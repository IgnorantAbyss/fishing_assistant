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
    hook_episode_active: bool
    qualified_active_hook_bar: bool
    current_hook_geometry_is_usable: bool
    divider_line_detected: bool
    divider_line_x: float | None
    divider_confidence: float
    fill_endpoint_x: float | None
    fill_ratio: float | None
    divider_crossed: bool
    divider_margin_passed: bool
    threshold_currently_exceeded: bool
    fallback_used: bool
    fallback_eligible: bool
    fallback_reason: str | None
    fallback_rejection_reason: str | None
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
        hook_episode_active: bool,
    ) -> HookActionDecision:
        evidence = observation.evidence if observation else {}
        features = set(evidence.get("matched_features", ()))
        fill_ratio = observation.fill_ratio if observation is not None else None
        qualified_active = bool(observation and observation.detected)
        raw_valid_fill = bool(
            qualified_active
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
        explicit_geometry_usable = bool(
            hook_episode_active
            and explicit_divider_valid
            and explicit_fill_x is not None
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
        legacy_geometry_usable = bool(
            hook_episode_active
            and not explicit_geometry
            and divider_valid
            and raw_valid_fill
        )
        current_geometry_usable = explicit_geometry_usable or legacy_geometry_usable
        divider_crossed = margin_passed = False
        if not explicit_divider_valid and raw_valid_fill and valid_bbox:
            left, _, right, _ = (float(value) for value in bbox)
            width = right - left
            fill_x = left + float(fill_ratio) * width
            if divider_valid:
                divider_x = left + float(divider_ratio) * width
                divider_confidence = float(evidence.get("divider_confidence", 1.0))
        if current_geometry_usable and fill_x is not None and divider_x is not None:
            divider_crossed = fill_x >= divider_x
            margin_passed = fill_x >= divider_x + self.config.divider_safety_margin_px

        ratio_trustworthy = bool(evidence.get("fallback_ratio_trustworthy", not explicit_geometry))
        fallback_eligible = bool(
            hook_episode_active
            and qualified_active
            and not divider_valid
            and raw_valid_fill
            and ratio_trustworthy
        )
        fallback_used = fallback_eligible
        fallback_reason = "divider_line_not_reliably_available" if fallback_used else None
        if divider_valid:
            fallback_rejection = None
        elif not hook_episode_active:
            fallback_rejection = "hook_episode_not_active"
        elif not qualified_active:
            fallback_rejection = "fallback_requires_qualified_active_hook_bar"
        elif not raw_valid_fill:
            fallback_rejection = "fallback_requires_positive_fill_ratio"
        elif not ratio_trustworthy:
            fallback_rejection = "fallback_ratio_not_trustworthy"
        else:
            fallback_rejection = None
        crossing_ready = bool(
            margin_passed if current_geometry_usable
            else fallback_eligible and fill_ratio is not None
            and float(fill_ratio) >= self.config.fallback_trigger_threshold
        )
        action_ready = hook_episode_active and crossing_ready and not action_already_proposed
        if not hook_episode_active:
            reason = "hook_episode_not_active"
        elif action_already_proposed:
            reason = "hook_action_already_proposed_in_episode"
        elif current_geometry_usable and not divider_crossed:
            reason = "fill_has_not_crossed_divider"
        elif current_geometry_usable and not margin_passed:
            reason = "fill_crossed_divider_but_margin_pending"
        elif current_geometry_usable:
            reason = "divider_margin_passed"
        elif crossing_ready:
            reason = "fallback_threshold_passed"
        elif fallback_rejection is not None:
            reason = fallback_rejection
        elif explicit_divider_valid:
            reason = "current_fill_geometry_unavailable"
        else:
            reason = "fallback_threshold_pending"
        return HookActionDecision(
            hook_episode_active=hook_episode_active,
            qualified_active_hook_bar=qualified_active,
            current_hook_geometry_is_usable=current_geometry_usable,
            divider_line_detected=divider_valid,
            divider_line_x=divider_x,
            divider_confidence=divider_confidence,
            fill_endpoint_x=fill_x,
            fill_ratio=fill_ratio,
            divider_crossed=divider_crossed,
            divider_margin_passed=margin_passed,
            threshold_currently_exceeded=crossing_ready,
            fallback_used=fallback_used,
            fallback_eligible=fallback_eligible,
            fallback_reason=fallback_reason,
            fallback_rejection_reason=fallback_rejection,
            action_ready=action_ready,
            one_shot_guard_result=(
                "blocked_already_proposed" if action_already_proposed else "not_previously_proposed"
            ),
            reason=reason,
        )
