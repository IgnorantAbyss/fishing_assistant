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
