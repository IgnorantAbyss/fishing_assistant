from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import pytest
import yaml

from src.fishing_v2.domain.action_intent import ActionIntent
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import (
    GetObservation,
    PressObservation,
    PromptObservation,
    PromptObservationKind,
    ResultBannerObservation,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import ObservationFusion, StateEvidence
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.perception.result_banner_observer import (
    ResultBannerConfig,
    ResultBannerObserver,
)
from src.fishing_v2.replay.v2_replay_runner import _config_objects
from src.fishing_v2.runtime.detector_activation import (
    DetectorActivationMode,
    DetectorActivationPolicy,
)
from src.fishing_v2.runtime.detector_evidence import DetectorEvidenceQualifier
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM
from src.fishing_v2.runtime.runtime_controller import (
    ActionExecutionMode,
    RuntimeController,
)
from src.fishing_v2.runtime.safety_policy import SafetyConfig, SafetyPolicy


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "fishing_v2.yaml"
BANNER_ROOT = ROOT / "assets" / "reference" / "result_banner"


def _prompt(frame: int, timestamp: float, kind: PromptObservationKind) -> PromptObservation:
    return PromptObservation(
        kind, 0.99, {kind.value: 0.99}, "synthetic", frame, timestamp
    )


def _bundle(
    frame: int,
    timestamp: float,
    *,
    prompt: PromptObservationKind = PromptObservationKind.UNKNOWN,
    get: bool = False,
    banner: bool = False,
    press: bool = False,
) -> ObservationBundle:
    return ObservationBundle(
        frame,
        timestamp,
        _prompt(frame, timestamp, prompt),
        None,
        PressObservation(press, 0.99 if press else 0.0, frame, timestamp),
        GetObservation(get, 0.99 if get else 0.0, frame, timestamp),
        ResultBannerObservation(
            banner, 0.99 if banner else 0.0, frame, timestamp,
            evidence={"collect_eligible": False},
        ),
    )


def _evidence(frame: int, timestamp: float, state: RuntimeState | None) -> StateEvidence:
    return StateEvidence(
        {} if state is None else {state: 0.99},
        (), (), state, 0.99 if state else 0.0, "synthetic", frame, timestamp,
    )


def test_result_pending_minimum_grace_blocks_idle_prompt() -> None:
    fsm = FishingFSM(
        FSMConfig(stable_frames=1, result_minimum_pending_sec=1.5),
        initial_state=RuntimeState.RESULT_PENDING,
    )
    result = fsm.advance(
        _evidence(1, 0.8, RuntimeState.IDLE), 0.8,
        _bundle(1, 0.8, prompt=PromptObservationKind.IDLE_CAST),
        recorded_observation=True,
    )
    assert result.next_state == RuntimeState.RESULT_PENDING
    assert result.transition_reason == "result_pending_minimum_grace"
    assert result.action_request.intent == ActionIntent.NONE


def test_idle_requires_grace_then_stable_idle_without_get_or_banner() -> None:
    fsm = FishingFSM(
        FSMConfig(stable_frames=2, result_minimum_pending_sec=1.5),
        initial_state=RuntimeState.RESULT_PENDING,
    )
    first = fsm.advance(
        _evidence(8, 1.6, RuntimeState.IDLE), 1.6,
        _bundle(8, 1.6, prompt=PromptObservationKind.IDLE_CAST),
        recorded_observation=True,
    )
    second = fsm.advance(
        _evidence(9, 1.7, RuntimeState.IDLE), 1.7,
        _bundle(9, 1.7, prompt=PromptObservationKind.IDLE_CAST),
        recorded_observation=True,
    )
    assert first.next_state == RuntimeState.RESULT_PENDING
    assert second.next_state == RuntimeState.IDLE
    assert second.action_request.intent == ActionIntent.NONE


def test_qualified_get_has_priority_during_grace_and_over_idle() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.RESULT_PENDING)
    result = fsm.advance(
        _evidence(1, 0.1, RuntimeState.IDLE), 0.1,
        _bundle(1, 0.1, prompt=PromptObservationKind.IDLE_CAST, get=True),
        recorded_observation=True,
    )
    assert result.next_state == RuntimeState.GET
    assert result.transition_reason == "get_panel_priority"
    assert result.action_request.intent == ActionIntent.NONE


def test_result_banner_holds_pending_without_collect_or_cast() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.RESULT_PENDING)
    result = fsm.advance(
        _evidence(20, 4.0, RuntimeState.IDLE), 4.0,
        _bundle(
            20, 4.0,
            prompt=PromptObservationKind.IDLE_CAST,
            banner=True,
        ),
        recorded_observation=True,
    )
    assert result.next_state == RuntimeState.RESULT_PENDING
    assert result.visual_acknowledgement == "RESULT_BANNER_PRESENT"
    assert result.action_request.intent == ActionIntent.NONE


def test_result_pending_maximum_timeout_safely_requests_resynchronization() -> None:
    fsm = FishingFSM(
        FSMConfig(
            stable_frames=1,
            result_minimum_pending_sec=1.5,
            result_maximum_pending_sec=10.0,
        ),
        initial_state=RuntimeState.RESULT_PENDING,
    )
    result = fsm.advance(
        _evidence(51, 10.1, None), 10.1,
        _bundle(51, 10.1, banner=True),
        recorded_observation=True,
    )
    assert result.next_state == RuntimeState.SYNC_REQUIRED
    assert result.transition_reason == "result_pending_maximum_timeout"
    assert result.action_request.intent == ActionIntent.NONE


def test_result_pending_uses_get_burst_at_twenty_fps() -> None:
    _, _, _, _, activation_config, _ = _config_objects(CONFIG)
    snapshot = DetectorActivationPolicy(activation_config).evaluate(
        RuntimeState.RESULT_PENDING,
        _bundle(1, 0.1),
        recorded_observation=True,
    )
    assert snapshot.get == DetectorActivationMode.BURST
    assert snapshot.get_fps == 20.0


def _banner_observer() -> ResultBannerObserver:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["result"]["banner_observer"]
    config = dict(config)
    config["roi"] = {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}
    return ResultBannerObserver(ResultBannerConfig.from_mapping(config))


def test_result_banner_manifest_hashes_and_strict_observer() -> None:
    manifest = yaml.safe_load(
        (BANNER_ROOT / "live_result_banner.yaml").read_text(encoding="utf-8")
    )
    observer = _banner_observer()
    for item in manifest["items"]:
        path = BANNER_ROOT / item["image"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["source_image_sha256"]
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        result = observer.observe(frame, FrameContext(item["source_frame"], 0.0))
        assert result.detected is item["expected_result_banner_present"]
        assert result.evidence["collect_eligible"] is False
        assert result.evidence["runtime_effect"] == "hold_result_pending_only"


def test_result_banner_never_enters_get_or_proposes_collect() -> None:
    fusion_config, fsm_config, _, safety_config, activation_config, qualification_config = (
        _config_objects(CONFIG)
    )
    fsm = FishingFSM(fsm_config, initial_state=RuntimeState.RESULT_PENDING)
    controller = RuntimeController(
        ObservationFusion(fusion_config),
        fsm,
        SafetyPolicy(safety_config),
        action_sink=None,
        activation_policy=DetectorActivationPolicy(activation_config),
        evidence_qualifier=DetectorEvidenceQualifier(qualification_config),
    )
    result = controller.process(
        _bundle(20, 4.0, prompt=PromptObservationKind.IDLE_CAST, banner=True),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
    )
    assert result.fsm.next_state == RuntimeState.RESULT_PENDING
    assert result.fsm.action_request.intent == ActionIntent.NONE
    assert result.action_applied is False
