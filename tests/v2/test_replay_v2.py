import json
from pathlib import Path

import cv2
import numpy as np

from src.fishing_v2.data.prompt_annotation import write_prompt_ground_truth
from src.fishing_v2.domain.observations import GetObservation, HookObservation, PressObservation
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.replay.v2_replay_runner import V2ReplayRunner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "config" / "fishing_v2.yaml"


def _session(root: Path) -> Path:
    session = root / "session_synthetic"
    frames = session / "frames"
    frames.mkdir(parents=True)
    for frame in range(1, 4):
        cv2.imwrite(str(frames / f"{frame:06d}.jpg"), np.full((60, 100, 3), frame * 20, dtype=np.uint8))
    (session / "manifest.json").write_text(json.dumps({
        "session_id": session.name, "frame_count": 3, "image_format": "jpg",
        "screen_size": [100, 60], "interval_sec": 0.2,
    }), encoding="utf-8")
    (session / "ground_truth.yaml").write_text(
        "segments:\n  - start: 1\n    end: 3\n    state: HOOK\n", encoding="utf-8"
    )
    write_prompt_ground_truth(session / "prompt_ground_truth.yaml", [
        {"start": 1, "end": 1, "observation": "WAITING_IN_PROGRESS"},
        {"start": 2, "end": 2, "observation": "IGNORE"},
        {"start": 3, "end": 3, "observation": "READY_BITE"},
    ], 3)
    return session


class NullHook:
    def observe(self, frame, context):
        return HookObservation(False, 0.0, context.frame_index, context.timestamp)


class NullPress:
    def observe(self, frame, context):
        return PressObservation(False, 0.0, context.frame_index, context.timestamp)


class NullGet:
    def observe(self, frame, context):
        return GetObservation(False, 0.0, context.frame_index, context.timestamp)


def _runner() -> V2ReplayRunner:
    return V2ReplayRunner(CONFIG, hook_detector=NullHook(), press_detector=NullPress(), get_detector=NullGet())


def test_scripted_prompt_reads_prompt_ground_truth_only(tmp_path: Path) -> None:
    run = _runner().run(_session(tmp_path), mode="scripted_prompt", report_dir=tmp_path / "reports", start_state=RuntimeState.WAITING)
    assert [row["prompt_observation"] for row in run.rows] == ["WAITING_IN_PROGRESS", "UNKNOWN", "READY_BITE"]
    assert all(row["global_ground_truth"] == "HOOK" for row in run.rows)


def test_global_ground_truth_is_reporting_only(tmp_path: Path) -> None:
    run = _runner().run(_session(tmp_path), mode="no_prompt", report_dir=tmp_path / "reports", start_state=RuntimeState.WAITING)
    assert run.rows[0]["global_ground_truth"] == "HOOK"
    assert run.rows[0]["next_runtime_state"] == "WAITING"


def test_no_prompt_mode_executes_without_observer(tmp_path: Path) -> None:
    run = _runner().run(_session(tmp_path), mode="no_prompt", report_dir=tmp_path / "reports", start_state=RuntimeState.WAITING)
    assert len(run.rows) == 3
    assert all(row["prompt_observation"] is None for row in run.rows)


def test_replay_report_contains_transition_reason(tmp_path: Path) -> None:
    run = _runner().run(_session(tmp_path), mode="scripted_prompt", report_dir=tmp_path / "reports", start_state=RuntimeState.WAITING)
    assert run.csv_path.is_file()
    assert run.report_path.is_file()
    assert all(row["transition_reason"] for row in run.rows)
    assert "Architecture validation only" in run.report_path.read_text(encoding="utf-8")


def test_v2_runner_does_not_reference_v1_prompt_or_state_detector() -> None:
    source = (PROJECT_ROOT / "src" / "fishing_v2" / "replay" / "v2_replay_runner.py").read_text(encoding="utf-8")
    assert "PromptClassifier" not in source
    assert "state_detector" not in source
    assert "global_ground_truth" in source  # output field only


def test_v2_config_action_emission_and_final_test_are_safe() -> None:
    import yaml

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["safety"]["emit_actions"] is False
    assert config["prompt"]["observer"] == "unimplemented"
    assert config["prompt"]["roi_status"] == "unapproved"
    assert config["data"]["final_test"] == {"status": "not_collected", "sessions": []}
