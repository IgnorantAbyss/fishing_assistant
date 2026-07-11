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
    }
    review = {
        "session_id": "session_test",
        "press_start": 10,
        "press_end": 20,
        "review_frame": 12,
        "predicted_sequence": "",
        "manually_confirmed_sequence": "",
        "review_status": "pending_manual_review",
        "notes": "sequence_not_recoverable_from_replay",
    }
    _write_review(tmp_path, [episode], [review])
    with (tmp_path / "review_items.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert tuple(rows[0]) == REVIEW_FIELDS
    assert rows[0]["manually_confirmed_sequence"] == ""
    assert rows[0]["review_status"] == "pending_manual_review"
    assert "not ground truth" in (tmp_path / "index.html").read_text(encoding="utf-8")
