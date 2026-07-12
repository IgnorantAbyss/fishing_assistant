from pathlib import Path

from tools.validate_scripted_replays import (
    FORMAL_SESSION_IDS,
    TRIAL_SESSION_ID,
    build_cross_session_summary,
    summarize_replay,
    validate_annotation,
    write_session_summary,
)


ROOT = Path(__file__).resolve().parents[2]
SESSIONS = ROOT / "assets" / "replay" / "sessions"


def _summary(session: str, *, actions: int = 0) -> dict:
    return {
        "session": session,
        "annotation": {"valid": True},
        "replay_completed": True,
        "actions_applied": actions,
        "sync_required_frames": 0,
        "unexplained_runtime_transitions": [],
        "false_action_intents": {
            "CAST": [], "START_HOOK": [], "HOOK_ACTION": [], "PRESS_SEQUENCE": [],
        },
        "failure_reasons": [],
        "result": "PASS",
    }


def test_formal_sessions_are_fixed_and_trial_is_excluded() -> None:
    assert len(FORMAL_SESSION_IDS) == 7
    assert TRIAL_SESSION_ID not in FORMAL_SESSION_IDS
    assert not (SESSIONS / TRIAL_SESSION_ID / "prompt_ground_truth.yaml").exists()


def test_all_formal_annotations_are_complete_and_valid() -> None:
    for session_id in FORMAL_SESSION_IDS:
        result = validate_annotation(SESSIONS / session_id)
        assert result["valid"] is True
        assert result["coverage_frames"] == result["total_frames"]
        assert result["coverage_percentage"] == 100.0


def test_per_session_reports_do_not_overwrite_each_other(tmp_path: Path) -> None:
    first = _summary(FORMAL_SESSION_IDS[0]) | {
        "frame_count": 1, "final_frame": 1, "final_state": "IDLE",
        "proposed_intents": {}, "transitions": [],
        "hook": {"qualified_detected": 0, "support_frames": 0},
        "press": {
            "panel_present_frames": 0, "support_frames": 0,
            "sequence_ready_frames": 0, "clean_pre_input_frame": None,
        },
        "get": {"qualified_detected": 0, "support_frames": 0},
        "detector_warnings": {"hook": [], "press": [], "get": []},
    }
    second = first | {"session": FORMAL_SESSION_IDS[1]}
    first_paths = write_session_summary(first, tmp_path)
    second_paths = write_session_summary(second, tmp_path)
    assert all(path.is_file() for path in (*first_paths, *second_paths))
    assert first_paths != second_paths


def test_cross_session_gate_requires_zero_applied_actions() -> None:
    safe = build_cross_session_summary([_summary(session) for session in FORMAL_SESSION_IDS])
    assert safe["actions_applied_total"] == 0
    assert safe["ready_for_prompt_observer"] is True
    unsafe_items = [_summary(session) for session in FORMAL_SESSION_IDS]
    unsafe_items[0] = _summary(FORMAL_SESSION_IDS[0], actions=1)
    unsafe = build_cross_session_summary(unsafe_items)
    assert unsafe["ready_for_prompt_observer"] is False


def test_terminal_runtime_mismatch_is_a_replay_failure() -> None:
    row = {
        "frame_index": 1,
        "global_ground_truth": "IDLE",
        "prompt_observation": "UNKNOWN",
        "previous_runtime_state": "PRESS",
        "next_runtime_state": "PRESS",
        "transition_reason": "held",
        "proposed_intent": "NONE",
        "action_applied": False,
        "raw_detected": {"hook": False, "press": False, "get": False},
        "qualified_detected": {"hook": False, "press": False, "get": False},
        "used_by_fusion": {"hook": False, "press": False, "get": False},
        "press_panel_candidate": False,
        "press_panel_present_raw": False,
        "press_sequence_ready": False,
        "qualified_press_observation": None,
        "press_sequence_candidate": [],
    }
    result = summarize_replay(
        "session_test", [row],
        {"valid": True, "total_frames": 1},
    )
    assert result["result"] == "FAIL"
    assert "unexplained_runtime_transition" in result["failure_reasons"]
    assert result["unexplained_runtime_transitions"][0]["issue"] == "terminal_runtime_state_mismatch"
