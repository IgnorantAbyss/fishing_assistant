from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_PATH = ROOT / "data" / "annotations" / "hook_bar_ground_truth.yaml"
FORMAL_SESSIONS = {
    "session_20260709_192315",
    "session_20260710_061220",
    "session_20260710_123210",
    "session_20260710_124419",
    "session_20260710_125441",
    "session_20260710_130308",
    "session_20260710_131254",
}
TRIAL_SESSION = "session_20260709_192231"


def _load_candidate():
    return yaml.safe_load(CANDIDATE_PATH.read_text(encoding="utf-8"))


def test_hook_bar_candidate_has_formal_sessions_and_excludes_trial():
    data = _load_candidate()

    assert data["status"] == "auto_candidate"
    assert set(data["sessions"]) == FORMAL_SESSIONS
    assert TRIAL_SESSION not in data["sessions"]
    assert TRIAL_SESSION in data["excluded_sessions"]


def test_hook_bar_candidate_ranges_are_valid():
    data = _load_candidate()

    for session in data["sessions"].values():
        for episode in session["episodes"]:
            start = episode["visible_start"]
            end = episode["visible_end"]
            first_clear = episode["first_clear_frame"]
            last_clear = episode["last_clear_frame"]

            assert start <= first_clear <= last_clear <= end
            assert episode["confidence"] in {"high", "medium", "low"}
            assert episode["global_hook_start"] - 5 <= start
            assert end <= episode["global_hook_end"] + 5
            assert all(start <= frame <= end for frame in episode["uncertain_frames"])


def test_hook_bar_candidate_is_visual_only():
    data = _load_candidate()

    assert data["source"] == "codex_visual_review"
    assert data["review_method"]["image_source"] == "raw_replay_frames"
    assert data["review_method"]["detector_output_used"] is False
