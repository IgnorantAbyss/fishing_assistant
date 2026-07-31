from __future__ import annotations

from src.fishing_v2.domain.action_intent import ActionIntent
from src.fishing_v2.domain.observations import (
    HookObservation,
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import ObservationFusion
from src.fishing_v2.live.live_detect_only import WouldFireDeduplicator
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM
from src.fishing_v2.runtime.runtime_controller import (
    ActionExecutionMode,
    RuntimeController,
)
from src.fishing_v2.runtime.safety_policy import SafetyConfig, SafetyPolicy


def _bundle(
    frame: int,
    timestamp: float,
    *,
    confidence: float = 1.0,
    fill_ratio: float | None = 0.70,
    hook_detected: bool = True,
) -> ObservationBundle:
    prompt = PromptObservation(
        PromptObservationKind.HOOK_INSTRUCTION,
        0.99,
        {PromptObservationKind.HOOK_INSTRUCTION.value: 0.99},
        "session_20260725_033103",
        frame,
        timestamp,
    )
    hook = (
        HookObservation(
            hook_detected,
            confidence,
            frame,
            timestamp,
            fill_ratio=fill_ratio,
            evidence={
                "matched_features": [
                    "hook_bar_rect",
                    "bar_fill",
                ],
                "fallback_ratio_trustworthy": True,
            },
        )
        if hook_detected else None
    )
    return ObservationBundle(frame, timestamp, prompt, hook)


def _controller() -> RuntimeController:
    return RuntimeController(
        ObservationFusion(),
        FishingFSM(
            FSMConfig(stable_frames=2),
            initial_state=RuntimeState.HOOK_PENDING,
            initial_timestamp=46.2,
        ),
        SafetyPolicy(SafetyConfig(emit_actions=False)),
    )


def _explicit_geometry_bundle(
    frame: int = 535,
    timestamp: float = 26.75,
    *,
    endpoint: float | None = 1397.0,
    divider: float | None = 1330.0,
    divider_confidence: float = 1.0,
    prompt_kind: PromptObservationKind = (
        PromptObservationKind.HOOK_INSTRUCTION
    ),
) -> ObservationBundle:
    return ObservationBundle(
        frame,
        timestamp,
        PromptObservation(
            prompt_kind,
            0.99,
            {prompt_kind.value: 0.99},
            "session_live_frame_535",
            frame,
            timestamp,
        ),
        HookObservation(
            False,
            0.4301,
            frame,
            timestamp,
            evidence={
                "matched_features": [],
                "crossing_geometry_version": 1,
                "divider_line_detected": divider is not None,
                "divider_line_x": divider,
                "divider_confidence": divider_confidence,
                "fill_endpoint_x": endpoint,
                "fallback_ratio_trustworthy": False,
            },
        ),
    )


def _hook_state_controller() -> RuntimeController:
    return RuntimeController(
        ObservationFusion(),
        FishingFSM(
            FSMConfig(stable_frames=1),
            initial_state=RuntimeState.HOOK,
            initial_timestamp=26.0,
        ),
        SafetyPolicy(SafetyConfig(emit_actions=False)),
    )


def test_session_132933_episode5_triggers_on_first_explicit_crossing() -> None:
    controller = _hook_state_controller()
    controller.fsm.force_state(
        RuntimeState.HOOK,
        274.0845991,
        "session_132933_episode5_hook",
    )
    endpoints = (
        (6210, 274.5883367, 1476.0),
        (6212, 274.6426050, 1446.0),
        (6214, 274.6914746, 1424.0),
        (6216, 274.7400555, 1392.0),
        (6218, 274.7946130, 1346.0),
        (6290, 276.5609259, 1364.0),
    )
    results: list[tuple[float, object]] = []

    for frame, timestamp, endpoint in endpoints:
        result = controller.process(
            _explicit_geometry_bundle(
                frame,
                timestamp,
                endpoint=endpoint,
                prompt_kind=PromptObservationKind.READY_BITE,
            ),
            foreground=True,
            runtime_environment_supported=True,
            action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
            preserve_proposal=True,
        )
        results.append((timestamp, result))

    emitted = [
        timestamp
        for timestamp, result in results
        if result.fsm.action_request.intent == ActionIntent.HOOK_ACTION
    ]
    assert emitted == [274.5883367]
    first = results[0][1]
    assert first.qualified.bundle.hook is not None
    assert first.qualified.bundle.hook.detected is False
    assert first.evidence.recommended_state == RuntimeState.READY
    assert first.fsm.next_state == RuntimeState.HOOK
    assert first.fsm.action_request.payload["reason"] == (
        "divider_margin_passed"
    )
    assert first.safety.reason == "action_emission_disabled"


def test_frame_535_explicit_geometry_satisfies_hook_safety_confidence() -> None:
    result = _hook_state_controller().process(
        _explicit_geometry_bundle(),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )

    assert result.qualified.bundle.hook is not None
    assert result.qualified.bundle.hook.detected is False
    assert result.qualified.bundle.hook.confidence == 0.4301
    assert result.evidence.confidence < 0.8
    assert result.fsm.action_request.intent == ActionIntent.HOOK_ACTION
    assert result.fsm.action_request.payload["action_ready"] is True
    assert (
        result.fsm.action_request.payload[
            "current_hook_geometry_is_usable"
        ]
        is True
    )
    assert result.safety.reason == "action_emission_disabled"
    assert result.action_applied is False


def test_hook_safety_block_before_emission_can_retry_same_explicit_window() -> None:
    controller = _hook_state_controller()
    blocked = controller.process(
        _explicit_geometry_bundle(frame=535, timestamp=26.75),
        foreground=False,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )
    assert blocked.fsm.action_request.intent == ActionIntent.HOOK_ACTION
    assert blocked.safety.reason == "foreground_window_not_confirmed"
    assert controller.fsm.hook_opportunity_lifecycle == "reserved"
    assert controller.release_external_hook_proposal_for_retry(
        blocked.fsm.action_request
    ) is True

    retried = controller.process(
        _explicit_geometry_bundle(frame=536, timestamp=26.78),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )
    assert retried.fsm.action_request.intent == ActionIntent.HOOK_ACTION
    assert retried.safety.reason == "action_emission_disabled"
    assert controller.mark_external_hook_emission_started(
        retried.fsm.action_request
    ) is True
    commit = controller.commit_external_action(
        retried.fsm.action_request,
        26.78,
    )
    assert commit.action_applied is True
    assert commit.next_state == RuntimeState.RESULT_PENDING


def test_incomplete_or_unsafe_explicit_geometry_never_becomes_sendable() -> None:
    unsafe_cases = (
        _explicit_geometry_bundle(endpoint=None),
        _explicit_geometry_bundle(divider=None),
        _explicit_geometry_bundle(divider_confidence=0.79),
        _explicit_geometry_bundle(endpoint=1339.0),
    )

    for bundle in unsafe_cases:
        result = _hook_state_controller().process(
            bundle,
            foreground=True,
            runtime_environment_supported=True,
            action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
            preserve_proposal=True,
        )
        if result.fsm.action_request.intent == ActionIntent.HOOK_ACTION:
            assert result.safety.reason == "evidence_confidence_too_low"
        else:
            assert result.fsm.action_request.intent == ActionIntent.NONE
        assert result.action_applied is False


def test_session_033103_dispatches_hook_on_committed_transition_frame() -> None:
    controller = _controller()
    first = controller.process(
        _bundle(1035, 46.30, fill_ratio=0.60),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
    )
    assert first.fsm.next_state == RuntimeState.HOOK_PENDING
    assert first.fsm.action_request.intent == ActionIntent.NONE

    committed = controller.process(
        _bundle(1036, 46.362, fill_ratio=0.70),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        preserve_proposal=True,
    )
    assert committed.fsm.previous_state == RuntimeState.HOOK_PENDING
    assert committed.fsm.next_state == RuntimeState.HOOK
    assert committed.fsm.action_request.intent == ActionIntent.HOOK_ACTION
    assert committed.safety.reason == "action_emission_disabled"
    assert committed.action_applied is False

    would_fire = WouldFireDeduplicator()
    event = would_fire.observe(
        committed.fsm.action_request,
        safety_reason=committed.safety.reason,
        frame_index=1036,
        timestamp=46.362,
        runtime_state=committed.fsm.next_state.value,
        prompt_evidence={},
        specialized_evidence={},
    )
    assert event is not None
    assert event["event_type"] == "WOULD_HOOK_ACTION"
    assert would_fire.raw_proposals == {"HOOK_ACTION": 1}
    assert would_fire.unique_events == {"WOULD_HOOK_ACTION": 1}

    repeated = controller.process(
        _bundle(1037, 46.415, fill_ratio=0.90),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
    )
    assert repeated.fsm.action_request.intent == ActionIntent.NONE
    assert repeated.action_applied is False


def test_hook_pending_without_strong_detector_trigger_never_proposes() -> None:
    for bundle in (
        _bundle(
            1035,
            46.30,
            confidence=0.84,
            fill_ratio=0.70,
        ),
        _bundle(
            1035,
            46.30,
            hook_detected=False,
            fill_ratio=None,
        ),
    ):
        controller = _controller()
        for frame in (1035, 1036):
            result = controller.process(
                _bundle(
                    frame,
                    bundle.timestamp + (frame - 1035) * 0.05,
                    confidence=(
                        bundle.hook.confidence
                        if bundle.hook is not None else 0.0
                    ),
                    fill_ratio=(
                        bundle.hook.fill_ratio
                        if bundle.hook is not None else None
                    ),
                    hook_detected=bundle.hook is not None,
                ),
                foreground=True,
                runtime_environment_supported=True,
                action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
            )
            assert result.fsm.action_request.intent == ActionIntent.NONE
