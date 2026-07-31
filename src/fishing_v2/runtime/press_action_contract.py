"""Strict frozen PRESS sequence contract shared by Safety and Live input."""

from __future__ import annotations

from typing import Any, Mapping


PRESS_KEYS = frozenset({"W", "A", "S", "D"})
MAX_PRESS_SLOT_CAPACITY = 14


def validate_frozen_press_payload(
    payload: Mapping[str, Any],
    *,
    require_eligibility: bool = True,
) -> tuple[tuple[str, ...], int]:
    """Return an unchanged sequence after validating the complete payload."""
    if require_eligibility:
        required = (
            "active_press_episode",
            "panel_confirmed",
            "frozen_by_consensus",
        )
        if not all(payload.get(name) is True for name in required):
            raise ValueError("press_sequence_eligibility_not_confirmed")

    raw_sequence = payload.get("sequence")
    if not isinstance(raw_sequence, (list, tuple)) or not raw_sequence:
        raise ValueError("invalid_press_sequence")
    sequence = tuple(raw_sequence)
    if any(
        not isinstance(symbol, str) or symbol not in PRESS_KEYS
        for symbol in sequence
    ):
        raise ValueError("invalid_press_sequence")

    raw_capacity = payload.get("slot_capacity")
    if (
        isinstance(raw_capacity, bool)
        or not isinstance(raw_capacity, int)
        or not 1 <= raw_capacity <= MAX_PRESS_SLOT_CAPACITY
    ):
        raise ValueError("invalid_press_slot_capacity")
    if len(sequence) > raw_capacity:
        raise ValueError("press_sequence_exceeds_slot_capacity")
    return sequence, raw_capacity


def validate_press_timing_plan(
    payload: Mapping[str, Any],
    *,
    sequence_length: int,
    required: bool = False,
    initial_delay_range_ms: tuple[int, int] | None = None,
    inter_key_gap_range_ms: tuple[int, int] | None = None,
    required_key_hold_ms: int | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], int | None, int | None]:
    """Validate a pre-sampled PRESS pacing plan without transforming it."""
    plan = payload.get("press_timing_plan")
    if plan is None:
        if required:
            raise ValueError("press_timing_plan_required")
        return (), (), None, None
    if not isinstance(plan, Mapping):
        raise ValueError("invalid_press_timing_plan")
    holds = plan.get("key_hold_ms")
    gaps = plan.get("inter_key_gap_ms")
    if not isinstance(holds, (list, tuple)) or not isinstance(
        gaps, (list, tuple)
    ):
        raise ValueError("invalid_press_timing_plan")
    if len(holds) != sequence_length or len(gaps) != max(
        0, sequence_length - 1
    ):
        raise ValueError("invalid_press_timing_plan")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        for value in holds
    ):
        raise ValueError("invalid_press_key_hold_plan")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in gaps
    ):
        raise ValueError("invalid_press_inter_key_gap_plan")
    initial = plan.get("sampled_initial_delay_ms")
    total = plan.get("planned_total_duration_ms")
    if (
        isinstance(initial, bool)
        or not isinstance(initial, int)
        or initial < 0
        or isinstance(total, bool)
        or not isinstance(total, int)
        or total != initial + sum(holds) + sum(gaps)
    ):
        raise ValueError("invalid_press_timing_plan")
    if (
        initial_delay_range_ms is not None
        and not initial_delay_range_ms[0]
        <= initial
        <= initial_delay_range_ms[1]
    ):
        raise ValueError("press_initial_delay_out_of_range")
    if inter_key_gap_range_ms is not None and any(
        not inter_key_gap_range_ms[0]
        <= value
        <= inter_key_gap_range_ms[1]
        for value in gaps
    ):
        raise ValueError("press_inter_key_gap_out_of_range")
    if required_key_hold_ms is not None and any(
        value != required_key_hold_ms for value in holds
    ):
        raise ValueError("press_key_hold_out_of_range")
    return tuple(holds), tuple(gaps), initial, total
