from pathlib import Path

from src.fishing_v2.domain.observations import PromptObservation, PromptObservationKind
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.replay import v2_replay_runner as runner_module
from src.fishing_v2.runtime.runtime_controller import ActionExecutionMode
from tests.v2.test_replay_v2 import _runner, _session


ROOT = Path(__file__).resolve().parents[2]


class FixedObserver:
    def __init__(self) -> None:
        self.frames = []

    def observe(self, frame, context):
        self.frames.append(context.frame_index)
        return PromptObservation(
            PromptObservationKind.WAITING_IN_PROGRESS,
            0.99,
            {PromptObservationKind.WAITING_IN_PROGRESS.value: 0.99},
            "prototype_v1",
            context.frame_index,
            context.timestamp,
            {
                "predicted_label": "WAITING_IN_PROGRESS",
                "observer_version": "prototype_v1",
            },
        )


def test_predicted_replay_does_not_load_prompt_ground_truth_for_runtime(monkeypatch, tmp_path: Path) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("Predicted Runtime must not load Prompt Ground Truth")

    monkeypatch.setattr(runner_module, "load_prompt_ground_truth", forbidden)
    observer = FixedObserver()
    run = _runner().run(
        _session(tmp_path),
        mode="predicted_prompt",
        report_dir=tmp_path / "reports",
        start_state=RuntimeState.WAITING,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        prompt_observer=observer,
    )
    assert observer.frames == [1, 2, 3]
    assert all(row["prompt_observation"] == "WAITING_IN_PROGRESS" for row in run.rows)
    assert all(row["prompt_observer_raw"]["observer_version"] == "prototype_v1" for row in run.rows)
    assert all(row["action_applied"] is False for row in run.rows)


def test_predicted_replay_has_no_keyboard_sink_or_specialized_config_change() -> None:
    source = (ROOT / "src" / "fishing_v2" / "replay" / "v2_replay_runner.py").read_text(encoding="utf-8")
    assert "action_sink=None" in source
    assert "pydirectinput" not in source
    assert "pyautogui" not in source
    assert "keyboard" not in source.lower()
    assert "LegacyHookDetectorAdapter" in source
    assert "LegacyPressDetectorAdapter" in source
    assert "LegacyGetDetectorAdapter" in source
