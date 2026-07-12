from pathlib import Path
import csv
import json

from src.fishing_v2.data.press_sequence_ground_truth import load_press_sequence_ground_truth
from src.detectors.press_detector import ARROW_TO_KEY, detect_press_sequence


ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH = ROOT / "data" / "annotations" / "press_sequence_ground_truth.yaml"
SESSIONS = ROOT / "assets" / "replay" / "sessions"
DIAGNOSTICS = ROOT / "reports" / "fishing_v2" / "press_detector_diagnostics_summary.json"
REVIEW_CSV = ROOT / "reports" / "fishing_v2" / "press_sequence_review" / "review_items.csv"


def test_all_user_confirmed_press_sequences_are_preserved() -> None:
    episodes = load_press_sequence_ground_truth(GROUND_TRUTH)
    assert [(item.session_id, item.press_start, item.press_end, "".join(item.sequence)) for item in episodes] == [
        ("session_20260709_192315", 472, 483, "ASDWWDWS"),
        ("session_20260710_061220", 507, 517, "AWSA"),
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


def test_later_input_effect_frames_are_not_selected_for_sequence() -> None:
    for session_id, frame in (
        ("session_20260709_192315", 474),
        ("session_20260710_123210", 577),
    ):
        result = detect_press_sequence(
            SESSIONS / session_id / "frames" / f"{frame:06}.jpg",
            save_debug=False,
        )
        assert result["input_effect_detected"] is True
        assert result["clean_frame_eligible"] is False
        assert result["selected_for_sequence"] is False


def test_diagnostics_selects_requested_earliest_frames_and_exact_matches() -> None:
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
    no_clean = episodes[("session_20260710_124419", 334)]
    assert no_clean["sequence_evaluable"] is False
    assert no_clean["exact_match"] is None


def test_review_csv_preserves_all_manual_confirmations() -> None:
    with REVIEW_CSV.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [row["manually_confirmed_sequence"] for row in rows] == [
        "ASDWWDWS", "AWSA", "DW", "DSWSS",
        "WASASDD", "WWDDWWSS", "WDASADS", "WAAASA",
    ]
    assert all(row["review_status"] == "confirmed" for row in rows)
