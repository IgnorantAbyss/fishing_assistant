from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.fishing_v2.domain.observations import PressObservation
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.live.live_detect_only import (
    LivePreflightError,
    validate_emit_actions,
)
from src.fishing_v2.live.press_shadow_verification import PressShadowVerifier
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode


def _press(
    frame: int,
    sequence: str,
    *,
    ready: bool,
    x_positions: tuple[int, ...] | None = None,
    slot_capacity: int | None = None,
) -> PressObservation:
    positions = x_positions or tuple(range(len(sequence)))
    boxes = [
        {
            "key": key,
            "bbox": [x * 20, 10, x * 20 + 15, 30],
            "confidence": 0.9,
        }
        for key, x in zip(sequence, positions, strict=True)
    ]
    return PressObservation(
        detected=True,
        confidence=0.95,
        frame_index=frame,
        timestamp=frame / 20.0,
        sequence=tuple(sequence) if ready else (),
        panel_candidate=True,
        panel_present=True,
        key_box_count=len(sequence),
        stable_key_box_count=len(sequence) if ready else 0,
        sequence_candidate=tuple(sequence),
        sequence_ready=ready,
        sequence_confidence=0.9,
        sequence_qualification_reason=(
            "earliest_clean_arrow_sequence_frozen"
            if ready else "sequence_consensus_pending"
        ),
        evidence={
            "press_evidence_version": 2,
            "key_boxes": boxes,
            "total_slot_count": (
                slot_capacity
                if slot_capacity is not None else len(sequence)
            ),
            "panel_disappeared": False,
        },
    )


def _observe(
    verifier: PressShadowVerifier,
    observation: PressObservation,
    *,
    state: RuntimeState = RuntimeState.PRESS,
    mode: DetectorActivationMode = DetectorActivationMode.ACTIVE,
):
    return verifier.observe(
        timestamp=observation.timestamp,
        frame_index=observation.frame_index,
        fsm_state=state,
        activation_mode=mode,
        raw=observation,
        qualified=observation,
        qualification_reason=observation.sequence_qualification_reason,
        safety_reason="no_action_requested",
        roi_pixels=np.zeros((20, 80, 3), dtype=np.uint8),
    )


def test_shadow_sorts_left_to_right_preserves_repeats_and_proposes_once(
    tmp_path: Path,
) -> None:
    verifier = PressShadowVerifier()
    raw = _press(
        1,
        "WASD",
        ready=True,
        x_positions=(3, 1, 0, 2),
        slot_capacity=8,
    )
    qualified = _press(1, "SSWAD", ready=True)
    proposal = verifier.observe(
        timestamp=0.05,
        frame_index=1,
        fsm_state=RuntimeState.PRESS,
        activation_mode=DetectorActivationMode.ACTIVE,
        raw=raw,
        qualified=qualified,
        qualification_reason="earliest_clean_arrow_sequence_frozen",
        safety_reason="no_action_requested",
        roi_pixels=np.zeros((20, 80, 3), dtype=np.uint8),
    )
    assert proposal is not None
    assert proposal.sequence == tuple("SSWAD")
    assert _observe(verifier, _press(2, "SSWAD", ready=True)) is None
    verifier.observe(
        timestamp=0.15,
        frame_index=3,
        fsm_state=RuntimeState.RESULT_PENDING,
        activation_mode=DetectorActivationMode.ARMED,
        raw=None,
        qualified=None,
        qualification_reason="panel_disappeared",
        safety_reason="no_action_requested",
    )

    summary = verifier.write_artifacts(tmp_path)
    trace = [
        json.loads(line)
        for line in Path(summary["press_decision_trace_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert trace[0]["left_to_right_sequence"] == ["S", "A", "D", "W"]
    assert trace[0]["frozen_sequence"] == list("SSWAD")
    assert trace[0]["proposal_created"] is True
    assert trace[1]["one_shot_blocked"] is True
    assert all(row["action_sink_called"] is False for row in trace)
    assert summary["press_roi_clip_path"] is not None
    assert Path(summary["press_roi_clip_path"]).exists()


def test_shadow_requires_press_state_and_active_activation() -> None:
    verifier = PressShadowVerifier()
    ready = _press(1, "WWAASSDD", ready=True)
    assert _observe(verifier, ready, state=RuntimeState.WAITING) is None
    assert _observe(
        verifier,
        ready,
        mode=DetectorActivationMode.BURST,
    ) is None
    assert verifier.proposal_count == 0
    later = _press(2, "WWAASSDD", ready=True)
    assert _observe(verifier, later) is not None
    assert verifier.proposal_count == 1


def test_next_press_episode_can_propose_a_different_length_sequence() -> None:
    verifier = PressShadowVerifier()
    first = _press(1, "WWAA", ready=True)
    assert _observe(verifier, first) is not None
    verifier.observe(
        timestamp=0.1,
        frame_index=2,
        fsm_state=RuntimeState.RESULT_PENDING,
        activation_mode=DetectorActivationMode.ARMED,
        raw=None,
        qualified=None,
        qualification_reason="panel_disappeared",
        safety_reason="no_action_requested",
    )
    second = _press(3, "SSWADDDD", ready=True)
    proposal = _observe(verifier, second)
    assert proposal is not None
    assert proposal.sequence == tuple("SSWADDDD")
    assert proposal.episode_index == 2
    assert verifier.proposal_count == 2


def test_trace_and_video_are_deferred_until_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verifier = PressShadowVerifier()
    writes: list[str] = []

    original_open = Path.open

    def tracked_open(path, *args, **kwargs):
        writes.append(str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    _observe(verifier, _press(1, "WASD", ready=True))
    assert writes == []

    summary = verifier.write_artifacts(tmp_path)
    assert writes
    assert Path(summary["press_decision_trace_path"]).exists()
    with Path(summary["press_roi_frames_path"]).open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["capture_frame_index"] == "1"


def test_press_sequence_remains_rejected_by_live_allowlist() -> None:
    with pytest.raises(
        LivePreflightError,
        match="--enable-live-press-sequence",
    ):
        validate_emit_actions(
            True,
            "sendinput",
            "CAST,START_HOOK,HOOK_ACTION,PRESS_SEQUENCE,COLLECT",
        )
