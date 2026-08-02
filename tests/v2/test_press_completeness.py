from __future__ import annotations

from dataclasses import replace

import pytest

from src.fishing_v2.domain.observations import PressObservation
from src.fishing_v2.runtime.press_sequence_aggregator import (
    PressSequenceAggregationConfig,
    PressSequenceTemporalAggregator,
)


def _slot(
    index: int,
    *,
    occupancy: str,
    key: str | None = None,
    confidence: float = 0.99,
    possible: bool = False,
) -> dict[str, object]:
    pixels = 100 if occupancy == "OCCUPIED" else 7 if possible else 0
    return {
        "index": index,
        "bbox": [100 + index * 40, 300, 138 + index * 40, 360],
        "occupancy": occupancy,
        "occupancy_confidence": 0.99 if occupancy != "UNCERTAIN" else 0.5,
        "mapped_key": key,
        "arrow_confidence": confidence if key else 0.0,
        "letter_pixel_count": pixels,
        "arrow_pixel_count": 20 if occupancy == "OCCUPIED" else 0,
        "coloured_pixel_count": pixels,
        "possible_occupied": possible,
    }


def _press(
    frame: int,
    sequence: str,
    *,
    capacity: int = 10,
    occupied_count: int | None = None,
    decoded_count: int | None = None,
    ambiguous_index: int | None = None,
    possible_index: int | None = None,
    clean: bool = True,
) -> PressObservation:
    occupied_count = len(sequence) if occupied_count is None else occupied_count
    decoded_count = len(sequence) if decoded_count is None else decoded_count
    slots = []
    for index in range(capacity):
        if index < occupied_count:
            key = sequence[index] if index < min(decoded_count, len(sequence)) else None
            occupancy = "UNCERTAIN" if index == ambiguous_index else "OCCUPIED"
            slots.append(_slot(index, occupancy=occupancy, key=key))
        else:
            slots.append(_slot(
                index,
                occupancy="EMPTY",
                possible=index == possible_index,
            ))
    decoded = sequence[:decoded_count]
    return PressObservation(
        detected=True,
        confidence=0.96,
        frame_index=frame,
        timestamp=frame * 0.2,
        sequence_candidate=tuple(decoded),
        panel_candidate=True,
        panel_present=True,
        panel_qualification_reason="structural_panel_present",
        key_box_count=len(decoded),
        sequence_confidence=0.9931,
        evidence={
            "press_evidence_version": 2,
            "panel_bbox": [80, 280, 500, 360],
            "panel_phase": "PANEL_CLEAN" if clean else "PANEL_APPEARING",
            "clean_frame_eligible": clean,
            "input_effect_detected": False,
            "arrow_sequence_ready": clean and decoded_count == occupied_count,
            "total_slot_count": capacity,
            "occupied_slot_count": occupied_count,
            "layout_conflict": False,
            "slots": slots,
        },
    )


def _aggregate(*observations: PressObservation):
    aggregator = PressSequenceTemporalAggregator(PressSequenceAggregationConfig(
        panel_confirmation_frames=1,
    ))
    result = None
    for observation in observations:
        result = aggregator.update(observation)
    assert result is not None
    return result


@pytest.mark.parametrize("sequence", ["D", "DS"])
def test_complete_short_sequence_can_freeze_when_trailing_slots_are_empty(
    sequence: str,
) -> None:
    result = _aggregate(*(_press(frame, sequence) for frame in range(1, 4)))
    assert result.sequence_ready is True
    assert result.sequence_candidate == tuple(sequence)
    assert result.completeness is not None
    assert result.completeness.complete is True


def test_high_confidence_partial_prefix_cannot_freeze() -> None:
    result = _aggregate(*(
        _press(frame, "D", occupied_count=7, decoded_count=1)
        for frame in range(1, 5)
    ))
    assert result.sequence_ready is False
    assert result.completeness is not None
    assert result.completeness.occupied_count == 7
    assert result.completeness.decoded_count == 1
    assert "occupied_slots_not_fully_decoded" in result.completeness.rejection_reasons


def test_one_missing_decode_prevents_freeze() -> None:
    result = _aggregate(*(
        _press(frame, "DSAWDD", occupied_count=7, decoded_count=6)
        for frame in range(1, 5)
    ))
    assert result.sequence_ready is False


def test_ambiguous_or_trailing_possible_slot_prevents_freeze() -> None:
    ambiguous = _aggregate(*(
        _press(frame, "DSAWDDS", occupied_count=7, ambiguous_index=3)
        for frame in range(1, 5)
    ))
    assert ambiguous.completeness is not None
    assert ambiguous.completeness.complete is False
    assert any(ambiguous.completeness.ambiguous_mask)

    possible = _aggregate(*(
        _press(frame, "DS", possible_index=5)
        for frame in range(1, 5)
    ))
    assert possible.completeness is not None
    assert possible.completeness.complete is False
    assert "possible_occupied_slot_after_decoded_sequence" in possible.completeness.rejection_reasons


def test_repeated_partial_consensus_never_freezes_then_complete_sequence_does() -> None:
    aggregator = PressSequenceTemporalAggregator(PressSequenceAggregationConfig(
        panel_confirmation_frames=1,
    ))
    for frame in range(1, 5):
        partial = aggregator.update(
            _press(frame, "D", occupied_count=7, decoded_count=1)
        )
        assert partial.sequence_ready is False
    final = None
    for frame in range(5, 8):
        final = aggregator.update(_press(frame, "DSAWDDS"))
    assert final is not None
    assert final.sequence_ready is True
    assert final.sequence_candidate == tuple("DSAWDDS")


def test_fuzzy_frame_does_not_erase_complete_candidate_history() -> None:
    aggregator = PressSequenceTemporalAggregator(PressSequenceAggregationConfig(
        panel_confirmation_frames=1,
    ))
    aggregator.update(_press(1, "WASD"))
    aggregator.update(_press(2, "WASD"))
    fuzzy = _press(3, "W", occupied_count=4, decoded_count=1, clean=False)
    aggregator.update(fuzzy)
    result = aggregator.update(_press(4, "WASD"))
    assert result.sequence_ready is True
    assert result.sequence_candidate == tuple("WASD")


def test_panel_disappearance_before_complete_abstains() -> None:
    aggregator = PressSequenceTemporalAggregator(PressSequenceAggregationConfig(
        panel_confirmation_frames=1,
        panel_disappearance_frames=1,
    ))
    aggregator.update(_press(1, "D", occupied_count=7, decoded_count=1))
    absent = replace(
        _press(2, "", occupied_count=0, decoded_count=0),
        detected=False,
        panel_present=False,
        panel_candidate=False,
    )
    result = aggregator.update(absent)
    assert result.panel_disappeared is True
    assert result.sequence_ready is False
    assert result.completeness is not None
    assert result.completeness.complete is False
