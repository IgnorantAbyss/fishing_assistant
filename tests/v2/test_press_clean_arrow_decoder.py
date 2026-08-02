from pathlib import Path
import csv
import json

import cv2
import pytest

from src.fishing_v2.domain.observations import PressObservation
from src.fishing_v2.runtime.press_sequence_aggregator import (
    PressSequenceAggregationConfig,
    PressSequenceTemporalAggregator,
)
from src.fishing_v2.data.press_sequence_ground_truth import load_press_sequence_ground_truth
from src.detectors import press_detector
from src.detectors.press_detector import (
    ARROW_TO_KEY,
    _decode_arrow_slots,
    _find_panel_geometry,
    detect_press_sequence,
)


ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH = ROOT / "data" / "annotations" / "press_sequence_ground_truth.yaml"
SESSIONS = ROOT / "assets" / "replay" / "sessions"
DIAGNOSTICS = ROOT / "reports" / "fishing_v2" / "press_detector_diagnostics_summary.json"
REVIEW_CSV = ROOT / "reports" / "fishing_v2" / "press_sequence_review" / "review_items.csv"
SAS_FIXTURES = ROOT / "tests" / "fixtures" / "press_sas"
STRUCTURAL_FIXTURES = ROOT / "tests" / "fixtures" / "press_structural_occupancy"


def test_all_user_confirmed_press_sequences_are_preserved() -> None:
    episodes = load_press_sequence_ground_truth(GROUND_TRUTH)
    assert [(item.session_id, item.press_start, item.press_end, "".join(item.sequence)) for item in episodes] == [
        ("session_20260709_192315", 472, 483, "ASDWWDWS"),
        ("session_20260710_061220", 507, 514, "AWSA"),
        ("session_20260710_123210", 574, 583, "DW"),
        ("session_20260710_124419", 334, 355, "DSWSS"),
        ("session_20260710_125441", 396, 419, "WASASDD"),
        ("session_20260710_130308", 415, 438, "WWDDWWSS"),
        ("session_20260710_131254", 225, 249, "WDASADS"),
        ("session_20260710_131254", 554, 576, "WAAASA"),
    ]


def test_arrow_direction_mapping_is_explicit() -> None:
    assert ARROW_TO_KEY == {"LEFT": "A", "DOWN": "S", "RIGHT": "D", "UP": "W"}


def test_192315_earliest_clean_frame_decodes_exact_sequence() -> None:
    result = detect_press_sequence(
        SESSIONS / "session_20260709_192315" / "frames" / "000472.jpg",
        save_debug=False,
    )
    assert result["panel_phase"] == "PANEL_CLEAN"
    assert result["clean_frame_eligible"] is True
    assert result["occupied_slot_count"] == 8
    assert result["empty_slot_count"] == 2
    assert result["uncertain_slot_count"] == 0
    assert "".join(result["sequence_candidate"]) == "ASDWWDWS"
    assert result["debug"]["arrow_sequence_ready"] is True


def test_123210_earliest_clean_frame_decodes_exact_sequence() -> None:
    result = detect_press_sequence(
        SESSIONS / "session_20260710_123210" / "frames" / "000574.jpg",
        save_debug=False,
    )
    assert result["panel_phase"] == "PANEL_CLEAN"
    assert result["clean_frame_eligible"] is True
    assert result["occupied_slot_count"] == 2
    assert result["empty_slot_count"] == 8
    assert result["uncertain_slot_count"] == 0
    assert "".join(result["sequence_candidate"]) == "DW"
    assert result["debug"]["arrow_sequence_ready"] is True


@pytest.mark.parametrize(
    ("session_id", "frame", "expected"),
    [
        ("session_20260709_192315", 472, "ASDWWDWS"),
        ("session_20260710_061220", 507, "AWSA"),
        ("session_20260710_123210", 574, "DW"),
        ("session_20260710_124419", 334, "DSWSS"),
        ("session_20260710_125441", 396, "WASASDD"),
        ("session_20260710_130308", 415, "WWDDWWSS"),
        ("session_20260710_131254", 225, "WDASADS"),
        ("session_20260710_131254", 554, "WAAASA"),
    ],
)
def test_all_confirmed_earliest_frames_decode_exact_geometry_sequence(
    session_id: str,
    frame: int,
    expected: str,
) -> None:
    result = detect_press_sequence(
        SESSIONS / session_id / "frames" / f"{frame:06}.jpg",
        save_debug=False,
    )
    assert result["debug"]["arrow_sequence_ready"] is True
    assert "".join(result["sequence_candidate"]) == expected
    assert len(result["sequence_candidate"]) == len(expected)


def _decode_sas_fixture(frame: int) -> dict:
    image = cv2.imread(str(SAS_FIXTURES / f"frame_{frame:06d}.jpg"), cv2.IMREAD_COLOR)
    assert image is not None
    geometry = _find_panel_geometry(image)
    assert geometry["panel_present"] is True
    return _decode_arrow_slots(image, geometry)


def _decode_structural_fixture(name: str) -> dict:
    image = cv2.imread(str(STRUCTURAL_FIXTURES / name), cv2.IMREAD_COLOR)
    assert image is not None
    geometry = _find_panel_geometry(image)
    assert geometry["panel_present"] is True
    return _decode_arrow_slots(image, geometry)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("episode_1_shifted_wwaw.png", "WWAW"),
        ("episode_2_transparent_wsaaswa.png", "WWSAASWA"),
    ],
)
def test_live_structural_occupancy_fixtures_decode_complete_prefix(
    name: str,
    expected: str,
) -> None:
    result = _decode_structural_fixture(name)
    assert result["occupied_slot_count"] == len(expected)
    assert "".join(result["arrow_sequence"]) == expected
    assert all(
        slot["occupancy"] == "EMPTY"
        for slot in result["slots"][len(expected):]
    )


def test_saturated_transparent_background_is_diagnostic_not_occupancy() -> None:
    result = _decode_structural_fixture("episode_2_transparent_wsaaswa.png")
    for slot in result["slots"][8:]:
        assert slot["raw_coloured_pixel_count"] >= 300
        assert slot["selected_letter_component"] is None
        assert slot["arrow_pixel_count"] == 0
        assert slot["occupancy"] == "EMPTY"


def test_vertically_shifted_arrows_are_found_without_fixed_40_percent_crop() -> None:
    result = _decode_structural_fixture("episode_1_shifted_wwaw.png")
    for slot in result["slots"][:4]:
        assert slot["mapped_key"] in "WAW"
        assert slot["arrow_bbox"] is not None
        # The native arrows lie above the old round(60 * 0.40)=24/28px band.
        assert slot["arrow_bbox"][1] < slot["bbox"][1] + 28
        assert slot["component_assignment_mode"] == "joint_full_slot_components"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("episode_1_shifted_wwaw.png", "WWAW"),
        ("episode_2_transparent_wsaaswa.png", "WWSAASWA"),
    ],
)
def test_live_structural_fixture_builds_complete_certificate_and_freezes(
    name: str,
    expected: str,
) -> None:
    decoded = _decode_structural_fixture(name)
    aggregator = PressSequenceTemporalAggregator(PressSequenceAggregationConfig(
        panel_confirmation_frames=1,
    ))
    result = None
    for frame in range(1, 4):
        result = aggregator.update(PressObservation(
            detected=True,
            confidence=1.0,
            frame_index=frame,
            timestamp=frame * 0.2,
            panel_candidate=True,
            panel_present=True,
            sequence_candidate=tuple(decoded["arrow_sequence"]),
            sequence_confidence=1.0,
            key_box_count=len(expected),
            evidence={
                "panel_phase": "PANEL_CLEAN",
                "clean_frame_eligible": True,
                "input_effect_detected": False,
                "total_slot_count": decoded["total_slot_count"],
                "occupied_slot_count": decoded["occupied_slot_count"],
                "layout_conflict": decoded["layout_conflict"],
                "slots": decoded["slots"],
            },
        ))
    assert result is not None
    assert result.completeness is not None
    assert result.completeness.occupied_count == len(expected)
    assert result.completeness.decoded_count == len(expected)
    assert result.completeness.complete is True
    assert result.sequence_ready is True
    assert "".join(result.sequence_candidate) == expected


def test_sas_live_regression_uses_down_left_down_arrow_geometry() -> None:
    for frame in (159, 160, 161):
        result = _decode_sas_fixture(frame)
        occupied = result["slots"][:result["occupied_slot_count"]]
        assert [item["arrow_direction"] for item in occupied] == [
            "DOWN", "LEFT", "DOWN",
        ]
        assert "".join(result["arrow_sequence"]) == "SAS"
        assert result["arrow_sequence_ready"] is True


def test_arrow_region_excludes_selected_upper_letter_component() -> None:
    result = detect_press_sequence(
        SESSIONS / "session_20260710_130308" / "frames" / "000415.jpg",
        save_debug=False,
    )
    for slot in result["slots"][:result["occupied_slot_count"]]:
        letter = slot["selected_letter_component"]
        assert letter is not None
        joint_arrow = slot["selected_joint_arrow_component"]
        assert joint_arrow is not None
        assert joint_arrow["bbox"][1] >= letter["bbox"][3] - 1
        assert not (
            letter["component_source"] == "arrow_colour"
            and letter["label"] == joint_arrow["label"]
        )
        selected = next(
            item for item in slot["arrow_component_candidates"]
            if item["selected"]
        )
        assert selected["slot_centroid"][1] > letter["centroid"][1]


def test_occupancy_and_input_effect_do_not_depend_on_classifier_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = press_detector._classify_arrow

    def abstaining_classifier(mask):
        result = original(mask)
        return {
            **result,
            "arrow_direction": None,
            "arrow_confidence": 0.0,
            "mapped_key": None,
        }

    monkeypatch.setattr(press_detector, "_classify_arrow", abstaining_classifier)
    result = _decode_sas_fixture(159)
    assert result["occupied_slot_count"] == 3
    assert [item["occupancy"] for item in result["slots"][:3]] == [
        "OCCUPIED", "OCCUPIED", "OCCUPIED",
    ]
    assert result["arrow_sequence"] == ()
    assert result["arrow_sequence_ready"] is False
    assert result["input_effect_detected"] is False
    assert result["input_effect_uses_classification"] is False


def test_later_input_effect_frames_are_not_selected_for_sequence() -> None:
    result = detect_press_sequence(
        SESSIONS / "session_20260709_192315" / "frames" / "000474.jpg",
        save_debug=False,
    )
    assert result["input_effect_detected"] is True
    assert result["clean_frame_eligible"] is False
    assert result["selected_for_sequence"] is False


def test_diagnostics_reports_all_nine_regressions_in_accuracy_denominator() -> None:
    data = json.loads(DIAGNOSTICS.read_text(encoding="utf-8"))
    episodes = {
        (item["session_id"], item["press_start"]): item for item in data["episodes"]
    }
    first = episodes[("session_20260709_192315", 472)]
    assert first["selected_review_frame"] == 472
    assert first["predicted_sequence"] == "ASDWWDWS"
    assert first["exact_match"] is True
    second = episodes[("session_20260710_123210", 574)]
    assert second["selected_review_frame"] == 574
    assert second["predicted_sequence"] == "DW"
    assert second["exact_match"] is True
    recovered = episodes[("session_20260710_124419", 334)]
    assert recovered["sequence_evaluable"] is True
    assert recovered["predicted_sequence"] == "DSWSS"
    assert recovered["exact_match"] is True
    sas = episodes[("session_20260731_194416_local_fixture", 159)]
    assert sas["predicted_sequence"] == "SAS"
    assert sas["exact_match"] is True
    assert data["episode_count"] == 9
    assert data["sequence_ready_coverage"] == 1.0
    assert data["all_episode_exact_sequence_accuracy"] == 1.0
    assert data["abstained_episode_count"] == 0
    assert data["wrong_sequence_episode_count"] == 0


def test_review_csv_preserves_all_manual_confirmations() -> None:
    with REVIEW_CSV.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["manually_confirmed_sequence"] for row in rows] == [
        "ASDWWDWS", "AWSA", "DW", "DSWSS",
        "WASASDD", "WWDDWWSS", "WDASADS", "WAAASA",
    ]
    assert all(row["review_status"] == "human_confirmed_adjusted" for row in rows)
    assert all(row["source_yaml"].endswith("press_sequence_ground_truth.yaml") for row in rows)
