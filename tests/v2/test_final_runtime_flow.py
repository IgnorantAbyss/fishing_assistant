from pathlib import Path

import pytest
import yaml

from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.observations import (
    GetObservation,
    HookObservation,
    PressObservation,
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import ObservationFusion, StateEvidence
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM
from src.fishing_v2.runtime.runtime_controller import RuntimeController
from src.fishing_v2.runtime.safety_policy import SafetyConfig, SafetyDecision, SafetyPolicy
from src.fishing_v2.runtime.scheduling import PromptPollingConfig, RuntimeSchedulePolicy


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "fishing_v2.yaml"


def _bundle(
    timestamp: float,
    *,
    prompt: PromptObservationKind = PromptObservationKind.UNKNOWN,
    hook: bool = False,
    fill_ratio: float | None = None,
    press: bool = False,
    sequence: tuple[str, ...] = (),
    get: bool = False,
) -> ObservationBundle:
    frame = round(timestamp * 10) + 1
    return ObservationBundle(
        frame,
        timestamp,
        PromptObservation(prompt, 0.95, {prompt.value: 0.95}, "synthetic", frame, timestamp),
        HookObservation(
            hook, 0.98 if hook else 0.0, frame, timestamp,
            fill_ratio=fill_ratio,
            evidence={"matched_features": ["hook_bar_rect", "bar_fill"] if hook and fill_ratio else ["hook_bar_rect"] if hook else []},
        ),
        PressObservation(press, 0.98 if press else 0.0, frame, timestamp, sequence=sequence),
        GetObservation(get, 0.98 if get else 0.0, frame, timestamp),
    )


def _evidence(state: RuntimeState | None, timestamp: float = 0.0, confidence: float = 0.95) -> StateEvidence:
    return StateEvidence(
        {} if state is None else {state: confidence},
        ("synthetic",), (), state, confidence, "synthetic", 1, timestamp,
    )


def _controller(state: RuntimeState, *, emit_actions: bool = False, sink=None) -> RuntimeController:
    return RuntimeController(
        ObservationFusion(),
        FishingFSM(FSMConfig(stable_frames=1), initial_state=state),
        SafetyPolicy(SafetyConfig(emit_actions=emit_actions)),
        action_sink=sink,
    )


def test_ready_action_arms_hook_detector() -> None:
    class Sink:
        def emit(self, request): pass

    result = _controller(RuntimeState.READY, emit_actions=True, sink=Sink()).process(
        _bundle(0.1, prompt=PromptObservationKind.READY_BITE),
        foreground=True,
        runtime_environment_supported=True,
    )
    assert result.fsm.next_state == RuntimeState.HOOK_PENDING
    assert result.fsm.action_request.intent == ActionIntent.START_HOOK
    assert result.action_applied is True
    assert result.next_activation.hook == DetectorActivationMode.ARMED


def test_emit_actions_false_keeps_ready_after_proposal() -> None:
    result = _controller(RuntimeState.READY).process(
        _bundle(0.1, prompt=PromptObservationKind.READY_BITE),
        foreground=True,
        runtime_environment_supported=True,
    )
    assert result.fsm.action_request.intent == ActionIntent.START_HOOK
    assert result.action_applied is False
    assert result.fsm.next_state == RuntimeState.READY


def test_hook_instruction_uses_burst_without_confirming_hook() -> None:
    result = _controller(RuntimeState.HOOK_PENDING).process(
        _bundle(0.1, prompt=PromptObservationKind.HOOK_INSTRUCTION),
        foreground=True,
        runtime_environment_supported=True,
    )
    assert result.fsm.next_state == RuntimeState.HOOK_PENDING
    assert result.activation.hook == DetectorActivationMode.BURST


def test_hook_bar_is_required_for_hook_active() -> None:
    result = _controller(RuntimeState.HOOK_PENDING).process(
        _bundle(0.1, hook=True, fill_ratio=0.4),
        foreground=True,
        runtime_environment_supported=True,
    )
    assert result.fsm.next_state == RuntimeState.HOOK
    assert result.next_activation.hook == DetectorActivationMode.ACTIVE


def test_hook_action_arms_press_and_get_detectors() -> None:
    class Sink:
        def emit(self, request): pass

    result = _controller(RuntimeState.HOOK, emit_actions=True, sink=Sink()).process(
        _bundle(0.1, hook=True, fill_ratio=0.70),
        foreground=True,
        runtime_environment_supported=True,
    )
    assert result.fsm.action_request.intent == ActionIntent.HOOK_ACTION
    assert result.fsm.next_state == RuntimeState.RESULT_PENDING
    assert result.action_applied is True
    assert result.next_activation.press == DetectorActivationMode.ARMED
    assert result.next_activation.get == DetectorActivationMode.ARMED


def test_press_instruction_uses_burst_without_confirming_press() -> None:
    result = _controller(RuntimeState.RESULT_PENDING).process(
        _bundle(0.1, prompt=PromptObservationKind.PRESS_INSTRUCTION),
        foreground=True,
        runtime_environment_supported=True,
    )
    assert result.fsm.next_state == RuntimeState.RESULT_PENDING
    assert result.activation.press == DetectorActivationMode.BURST


def test_press_panel_confirms_press_active() -> None:
    result = _controller(RuntimeState.RESULT_PENDING).process(
        _bundle(0.1, press=True, sequence=("W", "A", "S", "D")),
        foreground=True,
        runtime_environment_supported=True,
    )
    assert result.fsm.next_state == RuntimeState.PRESS
    assert result.next_activation.press == DetectorActivationMode.ACTIVE
    assert result.fsm.action_request.intent == ActionIntent.PRESS_SEQUENCE


def test_get_retries_at_configured_point_four_seconds() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.GET)
    first = fsm.advance(_evidence(RuntimeState.GET, 0.0), 0.0, _bundle(0.0, get=True))
    assert fsm.commit_action(first.action_request, 0.0).action_applied
    early = fsm.advance(_evidence(RuntimeState.GET, 0.2), 0.2, _bundle(0.2, get=True))
    retry = fsm.advance(_evidence(RuntimeState.GET, 0.4), 0.4, _bundle(0.4, get=True))
    assert first.action_request.payload["attempt"] == 1
    assert early.action_request.intent == ActionIntent.NONE
    assert retry.action_request.payload["attempt"] == 2


def test_get_panel_disappearance_stops_collect_immediately() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.GET)
    fsm.advance(_evidence(RuntimeState.GET), 0.0, _bundle(0.0, get=True))
    result = fsm.advance(_evidence(None, 0.1), 0.1, _bundle(0.1, get=False))
    assert result.next_state == RuntimeState.COLLECT_PENDING
    assert result.action_request.intent == ActionIntent.NONE


def test_get_attempt_limit_enters_sync_required() -> None:
    config = FSMConfig(stable_frames=1, get_retry_interval_seconds=0.4, get_max_attempts=2, get_max_duration_seconds=5.0)
    fsm = FishingFSM(config, initial_state=RuntimeState.GET)
    first = fsm.advance(_evidence(RuntimeState.GET), 0.0, _bundle(0.0, get=True))
    fsm.commit_action(first.action_request, 0.0)
    second = fsm.advance(_evidence(RuntimeState.GET), 0.4, _bundle(0.4, get=True))
    fsm.commit_action(second.action_request, 0.4)
    result = fsm.advance(_evidence(RuntimeState.GET), 0.8, _bundle(0.8, get=True))
    assert result.next_state == RuntimeState.SYNC_REQUIRED
    assert result.transition_reason == "get_retry_attempts_exceeded"


def test_get_duration_limit_enters_sync_required() -> None:
    config = FSMConfig(stable_frames=1, get_retry_interval_seconds=0.4, get_max_attempts=12, get_max_duration_seconds=0.7)
    fsm = FishingFSM(config, initial_state=RuntimeState.GET)
    fsm.advance(_evidence(RuntimeState.GET), 0.0, _bundle(0.0, get=True))
    fsm.advance(_evidence(RuntimeState.GET), 0.4, _bundle(0.4, get=True))
    result = fsm.advance(_evidence(RuntimeState.GET), 0.8, _bundle(0.8, get=True))
    assert result.next_state == RuntimeState.SYNC_REQUIRED
    assert result.transition_reason == "get_retry_duration_exceeded"


def test_waiting_polling_and_detector_frequencies_are_configured() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    polling = config["prompt_polling"]
    assert polling["waiting_min_seconds"] == 3.0
    assert polling["waiting_interval_seconds"] == 4.0
    assert polling["waiting_max_seconds"] == 5.0
    assert config["hook_detector"]["burst_fps"] == 25
    assert config["press_detector"]["burst_fps"] == 20
    policy = RuntimeSchedulePolicy(PromptPollingConfig(**polling))
    assert policy.prompt_interval_seconds(RuntimeState.WAITING) == 4.0
    assert policy.prompt_interval_seconds(RuntimeState.READY) == pytest.approx(0.2)


def test_safety_rejects_foreground_and_resolution_failures() -> None:
    policy = SafetyPolicy(SafetyConfig(emit_actions=True))
    request = ActionRequest(ActionIntent.CAST, 0.95, "cast")
    evidence = _evidence(RuntimeState.IDLE)
    foreground = policy.evaluate(
        request, RuntimeState.IDLE, evidence,
        foreground=False, already_sent=False, elapsed_since_action=None,
        runtime_environment_supported=True, get_panel_present=False,
    )
    resolution = policy.evaluate(
        request, RuntimeState.IDLE, evidence,
        foreground=True, already_sent=False, elapsed_since_action=None,
        runtime_environment_supported=False, get_panel_present=False,
    )
    assert foreground.reason == "foreground_window_not_confirmed"
    assert resolution.reason == "unsupported_runtime_resolution"


def test_safety_cast_guard_and_collect_limits() -> None:
    policy = SafetyPolicy(SafetyConfig(emit_actions=True))
    evidence = _evidence(RuntimeState.GET)
    cast = policy.evaluate(
        ActionRequest(ActionIntent.CAST, 0.95, "cast"), RuntimeState.IDLE, evidence,
        foreground=True, already_sent=False, elapsed_since_action=None,
        runtime_environment_supported=True, get_panel_present=True,
    )
    collect = policy.evaluate(
        ActionRequest(ActionIntent.COLLECT, 0.95, "collect", {
            "attempt": 2, "max_attempts": 12,
            "elapsed_seconds": 0.4, "max_duration_seconds": 5.0,
        }), RuntimeState.GET, evidence,
        foreground=True, already_sent=False, elapsed_since_action=0.4,
        runtime_environment_supported=True, get_panel_present=True,
    )
    assert cast.reason == "get_panel_guard_cancelled_cast"
    assert collect.decision == SafetyDecision.ALLOW


def test_emit_actions_remains_false() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["safety"]["emit_actions"] is False
