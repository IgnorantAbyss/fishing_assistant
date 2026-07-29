from pathlib import Path
from dataclasses import replace

from src.detectors.press_detector import detect_press_sequence
from src.fishing_v2.domain.observations import (
    GetObservation,
    HookObservation,
    PressObservation,
)
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.detector_activation import (
    DetectorActivationMode,
    DetectorActivationSnapshot,
)
from src.fishing_v2.runtime.detector_evidence import (
    DetectorEvidenceQualifier,
    EvidenceQualificationConfig,
    PressEvidenceKind,
)


ROOT = Path(__file__).resolve().parents[2]
PILOT_FRAMES = ROOT / "assets" / "replay" / "sessions" / "session_20260710_130308" / "frames"


def _activation(mode: DetectorActivationMode = DetectorActivationMode.BURST) -> DetectorActivationSnapshot:
    return DetectorActivationSnapshot(
        hook=DetectorActivationMode.OFF,
        press=mode,
        get=DetectorActivationMode.OFF,
        hook_fps=0.0,
        press_fps=20.0 if mode != DetectorActivationMode.OFF else 0.0,
        get_fps=0.0,
    )


def _press(frame: int, *, panel: bool, sequence: str = "WWDDWWSS") -> PressObservation:
    timestamp = frame * 0.2
    boxes = [
        {
            "key": key,
            "bbox": [100 + index * 40, 300, 138 + index * 40, 340],
            "confidence": 0.60,
            "top_candidates": [
                {"key": key, "confidence": 0.60},
                {"key": "W" if key != "W" else "D", "confidence": 0.40},
            ],
        }
        for index, key in enumerate(sequence)
    ] if panel else []
    return PressObservation(
        detected=panel,
        confidence=0.95 if panel else 0.0,
        frame_index=frame,
        timestamp=timestamp,
        sequence_candidate=tuple(sequence) if panel else (),
        panel_candidate=panel,
        panel_present=panel,
        panel_qualification_reason="structural_panel_present" if panel else "panel_not_found",
        key_box_count=len(boxes),
        sequence_confidence=0.60 if panel else 0.0,
        sequence_qualification_reason="temporal_consensus_required" if panel else "panel_not_present",
        evidence={
            "press_evidence_version": 2,
            "key_boxes": boxes,
            "panel_bbox": [80, 280, 500, 350] if panel else None,
        },
    )


def _bundle(press: PressObservation) -> ObservationBundle:
    return ObservationBundle(
        press.frame_index,
        press.timestamp,
        None,
        HookObservation(False, 0.0, press.frame_index, press.timestamp),
        press,
        GetObservation(False, 0.0, press.frame_index, press.timestamp),
    )


def _with_arrow_phase(
    observation: PressObservation,
    *,
    clean: bool,
    input_effect: bool,
) -> PressObservation:
    slots = [
        {"occupancy": "OCCUPIED", "arrow_confidence": 0.9}
        for _ in observation.sequence_candidate
    ]
    return replace(observation, evidence={
        **observation.evidence,
        "clean_frame_eligible": clean,
        "input_effect_detected": input_effect,
        "arrow_sequence_ready": clean,
        "slots": slots,
    })


def test_real_pilot_panel_presence_does_not_depend_on_glyph_threshold() -> None:
    result = detect_press_sequence(PILOT_FRAMES / "000415.jpg", save_debug=False)
    assert result["panel_candidate"] is True
    assert result["panel_present"] is True
    assert result["detected"] is True
    assert result["sequence_ready"] is False
    assert result["clean_frame_eligible"] is True
    assert "".join(result["sequence_candidate"]) == "WWDDWWSS"


def test_non_press_frame_has_no_structural_panel() -> None:
    result = detect_press_sequence(PILOT_FRAMES / "000100.jpg", save_debug=False)
    assert result["panel_present"] is False
    assert result["detected"] is False


def test_panel_confirmation_precedes_temporal_sequence_consensus() -> None:
    qualifier = DetectorEvidenceQualifier(EvidenceQualificationConfig(
        press_panel_confirmation_frames=2,
        press_sequence_window_frames=5,
        press_sequence_consensus_frames=3,
        press_per_key_min_aggregated_confidence=0.68,
        press_sequence_min_aggregated_confidence=0.68,
    ))
    first = qualifier.qualify(_bundle(_press(1, panel=True)), _activation())
    assert first.press.press_evidence_kind == PressEvidenceKind.PRESS_PANEL_CANDIDATE
    assert first.press.qualified_detected is False

    second = qualifier.qualify(_bundle(_press(2, panel=True)), _activation())
    assert second.press.press_evidence_kind == PressEvidenceKind.PRESS_PANEL_PRESENT
    assert second.press.qualified_detected is True
    assert second.bundle.press is not None
    assert second.bundle.press.sequence_ready is False
    assert second.bundle.press.sequence == ()

    third = qualifier.qualify(_bundle(_press(3, panel=True)), _activation())
    assert third.press.press_evidence_kind == PressEvidenceKind.PRESS_SEQUENCE_READY
    assert third.bundle.press is not None
    assert third.bundle.press.sequence_ready is True
    assert third.bundle.press.sequence == tuple("WWDDWWSS")
    assert third.bundle.press.stable_key_box_count == 8


def test_panel_candidate_never_enters_fusion_or_creates_a_sequence() -> None:
    raw = _press(1, panel=False)
    raw = PressObservation(
        **{**raw.__dict__, "panel_candidate": True, "panel_qualification_reason": "geometry_incomplete"}
    )
    qualified = DetectorEvidenceQualifier().qualify(_bundle(raw), _activation())
    assert qualified.press.press_evidence_kind == PressEvidenceKind.PRESS_PANEL_CANDIDATE
    assert qualified.press.used_by_fusion is False
    assert qualified.bundle.press is not None
    assert qualified.bundle.press.sequence_ready is False
    assert qualified.bundle.press.sequence == ()


def test_off_mode_keeps_press_panel_diagnostic_only() -> None:
    qualifier = DetectorEvidenceQualifier(EvidenceQualificationConfig(press_panel_confirmation_frames=1))
    result = qualifier.qualify(_bundle(_press(1, panel=True)), _activation(DetectorActivationMode.OFF))
    assert result.press.diagnostic_only is True
    assert result.press.used_by_fusion is False
    assert result.press.qualified_detected is False


def test_temporal_sequence_supports_variable_length_without_padding() -> None:
    qualifier = DetectorEvidenceQualifier(EvidenceQualificationConfig(
        press_panel_confirmation_frames=1,
        press_sequence_window_frames=4,
        press_sequence_consensus_frames=3,
    ))
    results = [
        qualifier.qualify(_bundle(_press(frame, panel=True, sequence="WASDW")), _activation())
        for frame in range(1, 4)
    ]
    final = results[-1].bundle.press
    assert final is not None
    assert final.sequence_ready is True
    assert final.sequence == tuple("WASDW")
    assert final.stable_key_box_count == 5


def test_single_clean_arrow_frame_is_not_frozen_before_input_effect() -> None:
    qualifier = DetectorEvidenceQualifier(EvidenceQualificationConfig(
        press_panel_confirmation_frames=2,
        press_sequence_consensus_frames=3,
    ))
    first = _with_arrow_phase(_press(1, panel=True, sequence="WASD"), clean=True, input_effect=False)
    second = _with_arrow_phase(_press(2, panel=True, sequence="DDDD"), clean=False, input_effect=True)
    qualifier.qualify(_bundle(first), _activation())
    result = qualifier.qualify(_bundle(second), _activation())
    assert result.bundle.press is not None
    assert result.bundle.press.sequence_ready is False
    assert result.bundle.press.sequence == ()
    assert result.bundle.press.evidence["selected_clean_frame"] is None


def test_two_matching_clean_frames_freeze_across_one_missing_glyph_frame() -> None:
    qualifier = DetectorEvidenceQualifier(EvidenceQualificationConfig(
        press_panel_confirmation_frames=2,
        press_sequence_consensus_frames=3,
    ))
    first = _with_arrow_phase(
        _press(1, panel=True, sequence="WASD"),
        clean=True,
        input_effect=False,
    )
    missing = _with_arrow_phase(
        _press(2, panel=True, sequence="WAS"),
        clean=False,
        input_effect=False,
    )
    recovered = _with_arrow_phase(
        _press(3, panel=True, sequence="WASD"),
        clean=True,
        input_effect=False,
    )
    qualifier.qualify(_bundle(first), _activation())
    qualifier.qualify(_bundle(missing), _activation())
    result = qualifier.qualify(_bundle(recovered), _activation())
    assert result.bundle.press is not None
    assert result.bundle.press.sequence_ready is True
    assert result.bundle.press.sequence == tuple("WASD")
    assert result.bundle.press.evidence["selected_clean_frame"] == 1


def test_clean_looking_frame_after_input_effect_is_never_frozen() -> None:
    qualifier = DetectorEvidenceQualifier(EvidenceQualificationConfig(
        press_panel_confirmation_frames=1,
        press_sequence_consensus_frames=3,
    ))
    first = _with_arrow_phase(_press(1, panel=True, sequence="WASD"), clean=False, input_effect=True)
    later = _with_arrow_phase(_press(2, panel=True, sequence="WASD"), clean=True, input_effect=False)
    qualifier.qualify(_bundle(first), _activation())
    result = qualifier.qualify(_bundle(later), _activation())
    assert result.bundle.press is not None
    assert result.bundle.press.sequence_ready is False
    assert result.bundle.press.evidence["selected_clean_frame"] is None
