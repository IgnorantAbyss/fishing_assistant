from pathlib import Path

from src.fishing_v2.data.hook_bar_ground_truth import (
    FORMAL_SESSION_IDS,
    load_hook_bar_ground_truth,
)


ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH = ROOT / "data" / "annotations" / "hook_bar_ground_truth.yaml"
SESSION_ROOT = ROOT / "assets" / "replay" / "sessions"


def test_human_confirmed_hook_bar_ground_truth_schema() -> None:
    episodes = load_hook_bar_ground_truth(GROUND_TRUTH, session_root=SESSION_ROOT)

    assert len(episodes) == 9
    assert {item.session_id for item in episodes} == set(FORMAL_SESSION_IDS)
    assert all(item.visible_start <= item.first_clear_frame <= item.last_clear_frame <= item.visible_end for item in episodes)


def test_prompt_phase_can_precede_human_visible_bar_range() -> None:
    episodes = load_hook_bar_ground_truth(GROUND_TRUTH, session_root=SESSION_ROOT)

    assert all(item.global_hook_start <= item.visible_start for item in episodes)
    assert any(item.global_hook_start < item.visible_start for item in episodes)


def test_hook_ground_truth_records_visual_review_without_detector_output() -> None:
    import yaml

    data = yaml.safe_load(GROUND_TRUTH.read_text(encoding="utf-8"))
    assert data["status"] == "human_confirmed"
    assert "human_review" in data["source"]
    assert data["review_method"]["image_source"] == "raw_replay_frames"
    assert data["review_method"]["detector_output_used"] is False
