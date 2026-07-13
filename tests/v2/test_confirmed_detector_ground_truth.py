import csv
from pathlib import Path

from src.fishing_v2.data.hook_bar_ground_truth import (
    HookBarEpisodeGroundTruth,
    load_hook_bar_ground_truth,
)
from src.fishing_v2.data.press_sequence_ground_truth import load_press_sequence_ground_truth
from tools.hook_bar_visible_validation import outside_visible_counts, summarize_episode
from tools.sync_ground_truth_reviews import sync_review_csvs


ROOT = Path(__file__).resolve().parents[2]
SESSION_ROOT = ROOT / "assets" / "replay" / "sessions"
HOOK_YAML = ROOT / "data" / "annotations" / "hook_bar_ground_truth.yaml"
PRESS_YAML = ROOT / "data" / "annotations" / "press_sequence_ground_truth.yaml"


def _episode() -> HookBarEpisodeGroundTruth:
    return HookBarEpisodeGroundTruth(
        "session_test", 1, 8, 15, 10, 14, 11, 13, "high", (), ""
    )


def _row(frame: int, *, raw=False, qualified=False, fill=None, kind="REJECTED"):
    return {
        "frame": frame,
        "raw_detected": raw,
        "qualified": qualified,
        "used_by_fusion": qualified,
        "valid_fill": fill is not None and fill > 0,
        "fill_ratio": fill,
        "hook_evidence_kind": kind,
    }


def test_confirmed_yaml_loaders_do_not_require_review_csv() -> None:
    assert len(load_hook_bar_ground_truth(HOOK_YAML, session_root=SESSION_ROOT)) == 9
    assert len(load_press_sequence_ground_truth(PRESS_YAML, session_root=SESSION_ROOT)) == 8


def test_review_csv_is_overwritten_from_yaml_not_loaded_as_truth(tmp_path: Path) -> None:
    hook_csv = tmp_path / "hook.csv"
    press_csv = tmp_path / "press.csv"
    hook_csv.write_text("human_confirmed_start\n999\n", encoding="utf-8")
    press_csv.write_text("manually_confirmed_sequence\nXXXX\n", encoding="utf-8")

    sync_review_csvs(
        HOOK_YAML, PRESS_YAML, hook_csv, press_csv, SESSION_ROOT,
        tmp_path / "summary.md",
    )

    with hook_csv.open(encoding="utf-8-sig", newline="") as file:
        hook_rows = list(csv.DictReader(file))
    with press_csv.open(encoding="utf-8-sig", newline="") as file:
        press_rows = list(csv.DictReader(file))
    assert hook_rows[0]["human_confirmed_start"] == "459"
    assert press_rows[0]["manually_confirmed_sequence"] == "ASDWWDWS"
    assert all(row["source_yaml"].endswith("hook_bar_ground_truth.yaml") for row in hook_rows)
    assert all(row["review_status"] == "human_confirmed_adjusted" for row in press_rows)


def test_visible_and_clear_denominators_are_independent_of_global_hook() -> None:
    rows = [_row(frame) for frame in range(8, 16)]
    rows[2] = _row(10, raw=True, qualified=True, fill=0.4, kind="ACTIVE_HOOK_BAR")
    rows[3] = _row(11, raw=True, qualified=True, fill=0.7, kind="ACTIVE_HOOK_BAR")
    summary = summarize_episode(_episode(), rows, safe_zone_start=0.65, safe_zone_end=0.85)

    assert summary["visible_support_frames"] == 5
    assert summary["qualified_active_hook_bar_frames"] == 2
    assert summary["visible_qualified_recall"] == 0.4
    assert summary["clear_support_frames"] == 3
    assert summary["valid_fill_frames"] == 1
    assert summary["valid_fill_recall"] == 0.3333
    assert summary["safe_zone_frames"] == [11]


def test_qualified_detection_outside_visible_range_is_false_positive() -> None:
    rows = [
        _row(9, raw=True, qualified=True, fill=0.7, kind="ACTIVE_HOOK_BAR"),
        _row(10, raw=True, qualified=True, fill=0.7, kind="ACTIVE_HOOK_BAR"),
        _row(15, raw=True, qualified=False, fill=0.0, kind="RECTANGLE_CANDIDATE"),
    ]
    outside = outside_visible_counts(rows, [_episode()])
    assert outside["qualified_frames"] == [9]
    assert outside["rectangle_only_frames"] == [15]


def test_press_human_range_excludes_pure_ignore_tail() -> None:
    episodes = load_press_sequence_ground_truth(PRESS_YAML, session_root=SESSION_ROOT)
    item = next(value for value in episodes if value.session_id == "session_20260710_061220")
    assert item.press_end == 514
    assert not (item.press_start <= 515 <= item.press_end)
