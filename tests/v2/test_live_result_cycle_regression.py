from __future__ import annotations

from pathlib import Path

from src.config_loader import ROIConfig
from src.detectors.get_detector import detect_get_window
from src.fishing_v2.domain.action_intent import ActionIntent
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import (
    GetObservation,
    PromptObservation,
    PromptObservationKind,
    ResultBannerObservation,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import ObservationFusion
from src.fishing_v2.legacy_adapters.get_detector_adapter import LegacyGetDetectorAdapter
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.replay.v2_replay_runner import _config_objects
from src.fishing_v2.runtime.detector_activation import DetectorActivationPolicy
from src.fishing_v2.runtime.detector_evidence import DetectorEvidenceQualifier
from src.fishing_v2.runtime.fishing_fsm import FishingFSM
from src.fishing_v2.runtime.runtime_controller import (
    ActionExecutionMode,
    RuntimeController,
)
from src.fishing_v2.runtime.safety_policy import SafetyPolicy


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "fishing_v2.yaml"
GET = ROOT / "assets" / "reference" / "get" / "live_get_regression"
ROI_AS_FRAME = ROIConfig(
    (486, 418),
    {"get_window": (0.0, 0.0, 1.0, 1.0), "get_search": (0.0, 0.0, 1.0, 1.0)},
    None,
)


def _controller() -> RuntimeController:
    fusion, fsm_config, _, safety, activation, qualification = _config_objects(CONFIG)
    fsm = FishingFSM(fsm_config, initial_state=RuntimeState.RESULT_PENDING)
    return RuntimeController(
        ObservationFusion(fusion), fsm, SafetyPolicy(safety),
        action_sink=None,
        activation_policy=DetectorActivationPolicy(activation),
        evidence_qualifier=DetectorEvidenceQualifier(qualification),
    )


def _prompt(frame: int, timestamp: float, kind: PromptObservationKind) -> PromptObservation:
    return PromptObservation(kind, 0.99, {kind.value: 0.99}, "reviewed_live", frame, timestamp)


def _raw_get(path: Path, frame: int, timestamp: float) -> GetObservation:
    return LegacyGetDetectorAdapter(
        lambda _image: detect_get_window(path, roi_config=ROI_AS_FRAME)
    ).observe(object(), FrameContext(frame, timestamp))


def _get_cycle(files: tuple[str, str, str]) -> tuple[list, RuntimeController]:
    controller = _controller()
    results = []
    for index, name in enumerate(files, start=1):
        timestamp = index * 0.1
        get = _raw_get(GET / name, index, timestamp)
        results.append(controller.process(
            ObservationBundle(
                index, timestamp,
                _prompt(index, timestamp, PromptObservationKind.IDLE_CAST),
                None, None, get, None,
            ),
            foreground=True,
            runtime_environment_supported=True,
            action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        ))
    return results, controller


def test_cycle1_saved_live_evidence_enters_get_and_would_collect_once() -> None:
    results, _ = _get_cycle((
        "cycle1_002428_clear.jpg",
        "cycle1_002433_clear_confirm.jpg",
        "cycle1_002538_last.jpg",
    ))
    assert [result.qualified.get.qualified_detected for result in results] == [False, True, True]
    assert results[1].fsm.next_state == RuntimeState.GET
    assert results[2].fsm.action_request.intent == ActionIntent.COLLECT
    assert all(result.action_applied is False for result in results)


def test_cycle3_saved_live_evidence_enters_get_despite_idle_prompt_overlap() -> None:
    results, _ = _get_cycle((
        "cycle3_006528_clear.jpg",
        "cycle3_006529_clear_confirm.jpg",
        "cycle3_006573_idle_prompt_overlap.jpg",
    ))
    assert results[1].fsm.next_state == RuntimeState.GET
    assert results[2].fsm.next_state == RuntimeState.GET
    assert results[2].fsm.action_request.intent == ActionIntent.COLLECT
    assert all(result.action_applied is False for result in results)


def test_cycle2_banner_holds_then_stable_idle_exits_without_collect() -> None:
    controller = _controller()
    banner = ResultBannerObservation(
        True, 0.99, 1, 2.0,
        evidence={"collect_eligible": False, "runtime_effect": "hold_result_pending_only"},
    )
    held = controller.process(
        ObservationBundle(
            1, 2.0,
            _prompt(1, 2.0, PromptObservationKind.IDLE_CAST),
            None, None, GetObservation(False, 0.0, 1, 2.0), banner,
        ),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
    )
    first_idle = controller.process(
        ObservationBundle(
            2, 2.1,
            _prompt(2, 2.1, PromptObservationKind.IDLE_CAST),
            None, None, GetObservation(False, 0.0, 2, 2.1),
            ResultBannerObservation(False, 0.0, 2, 2.1),
        ),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
    )
    stable_idle = controller.process(
        ObservationBundle(
            3, 2.2,
            _prompt(3, 2.2, PromptObservationKind.IDLE_CAST),
            None, None, GetObservation(False, 0.0, 3, 2.2),
            ResultBannerObservation(False, 0.0, 3, 2.2),
        ),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
    )
    assert held.fsm.next_state == RuntimeState.RESULT_PENDING
    assert first_idle.fsm.next_state == RuntimeState.RESULT_PENDING
    assert stable_idle.fsm.next_state == RuntimeState.IDLE
    assert all(
        result.fsm.action_request.intent == ActionIntent.NONE
        for result in (held, first_idle, stable_idle)
    )
    assert all(result.action_applied is False for result in (held, first_idle, stable_idle))
