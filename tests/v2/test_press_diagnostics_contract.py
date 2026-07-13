import csv
from pathlib import Path

from tools.press_sequence_diagnostics import REVIEW_FIELDS, _mode_consistency, _write_review


def test_sequence_consistency_does_not_create_manual_ground_truth() -> None:
    candidate, consistency = _mode_consistency([
        tuple("WASD"), tuple("WASD"), tuple("WASD"), tuple("WSAD"),
    ])
    assert candidate == "WASD"
    assert consistency == 0.75


def test_review_bundle_keeps_manual_sequence_blank(tmp_path: Path) -> None:
    episode = {
        "session_id": "session_test",
        "press_start": 10,
        "press_end": 20,
        "review_frame": 12,
        "panel_present_first_frame": 10,
        "panel_confirmed_first_frame": 11,
        "sequence_ready_first_frame": None,
        "modal_sequence_candidate": "WASD",
        "predicted_sequence": "",
        "sequence_status": "sequence_not_recoverable_from_replay",
        "review_top_candidates": [],
        "review_image_name": "session_test_10_20.jpg",
        "panel_candidate_first_frame": 10,
        "earliest_clean_frame": None,
        "first_input_effect_frame": None,
        "selected_review_frame": 12,
        "selected_review_reason": "fallback_visual_review_no_clean_frame",
        "selected_panel_phase": "PANEL_APPEARING",
        "selected_slot_observations": [],
        "selected_occupied_slot_count": 0,
        "selected_empty_slot_count": 0,
        "selected_uncertain_slot_count": 0,
        "selected_layout_conflict": False,
        "expected_sequence": None,
        "exact_match": None,
    }
    review = {
        "session_id": "session_test",
        "press_start": 10,
        "press_end": 20,
        "review_frame": 12,
        "predicted_sequence": "",
        "manually_confirmed_sequence": "",
        "review_status": "pending_manual_review",
        "source_yaml": "data/annotations/press_sequence_ground_truth.yaml",
        "notes": "sequence_not_recoverable_from_replay",
    }
    _write_review(tmp_path, [episode], [review])
    with (tmp_path / "review_items.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert tuple(rows[0]) == REVIEW_FIELDS
    assert rows[0]["manually_confirmed_sequence"] == ""
    assert rows[0]["review_status"] == "pending_manual_review"
    assert "never loaded as ground truth" in (tmp_path / "index.html").read_text(encoding="utf-8")
