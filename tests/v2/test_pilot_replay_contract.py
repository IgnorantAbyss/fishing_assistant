from pathlib import Path

import yaml

from src.fishing_v2.data.prompt_roi import load_approved_prompt_roi
from src.fishing_v2.domain.observations import (
    GetObservation,
    HookObservation,
    PressObservation,
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import ObservationFusion
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM
from src.fishing_v2.runtime.runtime_controller import (
    ActionExecutionMode,
    RuntimeController,
)
from src.fishing_v2.runtime.safety_policy import SafetyConfig, SafetyPolicy


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "fishing_v2.yaml"
PILOT = ROOT / "assets" / "replay" / "sessions" / "session_20260710_130308"
EXPECTED_SEGMENTS = [
    {"start": 1, "end": 20, "observation": "READY_BITE"},
    {"start": 21, "end": 46, "observation": "HOOK_INSTRUCTION"},
    {"start": 47, "end": 62, "observation": "IGNORE"},
    {"start": 63, "end": 99, "observation": "IDLE_CAST"},
    {"start": 100, "end": 320, "observation": "WAITING_IN_PROGRESS"},
    {"start": 321, "end": 379, "observation": "READY_BITE"},
    {"start": 380, "end": 414, "observation": "HOOK_INSTRUCTION"},
    {"start": 415, "end": 438, "observation": "PRESS_INSTRUCTION"},
    {"start": 439, "end": 453, "observation": "IGNORE"},
    {"start": 454, "end": 512, "observation": "IDLE_CAST"},
    {"start": 513, "end": 600, "observation": "WAITING_IN_PROGRESS"},
]


def _bundle(
    frame: int,
    prompt: PromptObservationKind,
    *,
    hook=False,
    fill=None,
    get=False,
) -> ObservationBundle:
    timestamp = frame * 0.2
    return ObservationBundle(
        frame,
        timestamp,
        PromptObservation(prompt, 1.0, {prompt.value: 1.0}, "synthetic", frame, timestamp),
        HookObservation(
            hook, 0.98 if hook else 0.0, frame, timestamp,
            fill_ratio=fill,
            evidence={"matched_features": ["hook_bar_rect", "bar_fill"] if hook and fill else ["hook_bar_rect"] if hook else []},
        ),
        PressObservation(False, 0.0, frame, timestamp),
        GetObservation(get, 0.98 if get else 0.0, frame, timestamp),
    )


def _controller(state: RuntimeState) -> RuntimeController:
    return RuntimeController(
        ObservationFusion(),
        FishingFSM(FSMConfig(stable_frames=1), initial_state=state),
        SafetyPolicy(SafetyConfig(emit_actions=False)),
    )


def test_pilot_annotation_is_exactly_the_user_authored_content() -> None:
    data = yaml.safe_load((PILOT / "prompt_ground_truth.yaml").read_text(encoding="utf-8"))
    assert data == {"version": 1, "segments": EXPECTED_SEGMENTS}


def test_prompt_roi_is_approved_with_fixed_pixel_source() -> None:
    roi = load_approved_prompt_roi(CONFIG)
    assert roi is not None
    assert roi.candidate_id == "prompt_final_candidate"
    assert roi.pixel == (940, 36, 1620, 100)


def test_recorded_replay_does_not_commit_ready_proposal() -> None:
    controller = _controller(RuntimeState.READY)
    result = controller.process(
        _bundle(1, PromptObservationKind.READY_BITE),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
    )
    assert result.fsm.action_request.intent.value == "START_HOOK"
    assert result.action_applied is False
    assert result.fsm.next_state == RuntimeState.READY


def test_recorded_hook_prompt_then_qualified_bar_advances_visually() -> None:
    controller = _controller(RuntimeState.READY)
    prompt = controller.process(
        _bundle(1, PromptObservationKind.HOOK_INSTRUCTION, hook=True, fill=0.0),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
    )
    assert prompt.fsm.next_state == RuntimeState.HOOK_PENDING
    assert prompt.qualified.hook.qualified_detected is False
    active = controller.process(
        _bundle(2, PromptObservationKind.HOOK_INSTRUCTION, hook=True, fill=0.4),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
    )
    assert active.qualified.hook.qualified_detected is True
    assert active.fsm.next_state == RuntimeState.HOOK


def test_get_panel_outranks_idle_cast_in_recorded_replay() -> None:
    result = _controller(RuntimeState.RESULT_PENDING).process(
        _bundle(454, PromptObservationKind.IDLE_CAST, get=True),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
    )
    assert result.qualified.get.qualified_detected is True
    assert result.fsm.next_state == RuntimeState.GET


def test_get_panel_outranks_recorded_press_prompt() -> None:
    result = _controller(RuntimeState.HOOK).process(
        _bundle(440, PromptObservationKind.PRESS_INSTRUCTION, get=True),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
    )
    assert result.qualified.get.qualified_detected is True
    assert result.fsm.next_state == RuntimeState.GET
    assert result.fsm.visual_acknowledgement == "qualified_get_panel_present"


def test_no_prompt_dataset_is_materialized() -> None:
    assert not (ROOT / "datasets" / "prompt_observation_v1").exists()
