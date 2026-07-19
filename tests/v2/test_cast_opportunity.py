from src.fishing_v2.domain.observations import (
    GetObservation,
    PromptObservationKind,
    ResultBannerObservation,
)
from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.live.cast_opportunity import (
    CastOpportunityConfig,
    CastOpportunityController,
    PostCycleClearanceTracker,
    PresenceState,
)
from src.fishing_v2.ports.action_sink import ActionExecutionResult
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode
from src.fishing_v2.live.live_detect_only import WouldFireDeduplicator


def _execution(action_id: str, timestamp: float, *, applied: bool = True):
    return ActionExecutionResult(
        action_id=action_id,
        intent_type="CAST",
        requested_at=timestamp,
        started_at=timestamp,
        completed_at=timestamp,
        success=applied,
        applied=applied,
        emitted_event_count=2 if applied else 0,
        expected_event_count=2,
        target_hwnd=10,
        foreground_hwnd=10,
        rejection_reason=None if applied else "foreground_window_mismatch",
        os_input_emitted=applied,
    )


def _schedule(controller: CastOpportunityController, timestamp: float = 0.0):
    return controller.schedule(
        timestamp=timestamp,
        clearance_id="no_get_clearance:1",
        runtime_state=RuntimeState.IDLE,
        prompt_kind=PromptObservationKind.IDLE_CAST,
        physical_get_episode_open=False,
    )


def test_cast_requires_explicit_clearance() -> None:
    controller = CastOpportunityController()
    attempt, events = controller.schedule(
        timestamp=0.0,
        clearance_id=None,
        runtime_state=RuntimeState.IDLE,
        prompt_kind=PromptObservationKind.IDLE_CAST,
        physical_get_episode_open=False,
    )
    assert attempt is None
    assert events == ()


def test_cast_rejects_active_or_terminal_but_visible_get_episode() -> None:
    for runtime_state, prompt_kind, episode_open in (
        (RuntimeState.GET, PromptObservationKind.IDLE_CAST, False),
        (RuntimeState.IDLE, PromptObservationKind.UNKNOWN, False),
        (RuntimeState.IDLE, PromptObservationKind.IDLE_CAST, True),
    ):
        controller = CastOpportunityController()
        attempt, events = controller.schedule(
            timestamp=0.0,
            clearance_id="no_get_clearance:1",
            runtime_state=runtime_state,
            prompt_kind=prompt_kind,
            physical_get_episode_open=episode_open,
        )
        assert attempt is None
        assert events == ()


def _get(frame: int, timestamp: float, detected: bool) -> GetObservation:
    return GetObservation(detected, 0.99, frame, timestamp)


def _banner(frame: int, timestamp: float, detected: bool) -> ResultBannerObservation:
    return ResultBannerObservation(detected, 0.99, frame, timestamp)


def _observe_clearance(
    tracker: PostCycleClearanceTracker,
    *,
    frame: int,
    timestamp: float,
    previous: RuntimeState,
    current: RuntimeState,
    get: GetObservation | None,
    banner: ResultBannerObservation | None,
    mode: DetectorActivationMode = DetectorActivationMode.BURST,
    prompt: PromptObservationKind = PromptObservationKind.IDLE_CAST,
    episode_open: bool = False,
):
    return tracker.observe(
        timestamp=timestamp,
        previous_state=previous,
        current_state=current,
        prompt_kind=prompt,
        prompt_frame_index=frame,
        get_observation=get,
        get_activation_mode=mode,
        result_banner=banner,
        physical_get_episode_open=episode_open,
    )


def test_session_132835_no_get_timeline_creates_one_fresh_clearance() -> None:
    tracker = PostCycleClearanceTracker(CastOpportunityConfig(
        clearance_freshness_seconds=4.0,
        stable_idle_frames_required=2,
    ))
    _observe_clearance(
        tracker,
        frame=1856,
        timestamp=82.7906535,
        previous=RuntimeState.HOOK,
        current=RuntimeState.RESULT_PENDING,
        get=_get(1856, 82.7906535, False),
        banner=None,
        mode=DetectorActivationMode.ARMED,
    )
    first_status = tracker.status(82.7906535)
    assert first_status.get_presence_state == PresenceState.UNKNOWN
    assert first_status.result_banner_presence_state == PresenceState.UNKNOWN

    _observe_clearance(
        tracker,
        frame=1860,
        timestamp=83.3,
        previous=RuntimeState.RESULT_PENDING,
        current=RuntimeState.RESULT_PENDING,
        get=_get(1860, 83.3, False),
        banner=_banner(1860, 83.3, False),
    )
    events = _observe_clearance(
        tracker,
        frame=1868,
        timestamp=84.4862955,
        previous=RuntimeState.RESULT_PENDING,
        current=RuntimeState.IDLE,
        get=_get(1868, 84.4862955, False),
        banner=_banner(1868, 84.4862955, False),
    )
    assert [event.event_type for event in events] == ["no_get_clearance_created"]
    clearance = tracker.current(84.5)
    assert clearance is not None
    assert clearance.clearance_id == "no_get_clearance:1"
    assert tracker.summary()["no_get_clearance_count"] == 1
    certified = tracker.certified_get_absence(timestamp=84.5, frame_index=1869)
    assert certified is not None
    assert certified.source == "post_cycle_no_get_clearance"
    assert certified.evidence["certified_absence"] is True


def test_transition_edge_can_be_missed_and_next_frame_still_creates_clearance() -> None:
    tracker = PostCycleClearanceTracker(CastOpportunityConfig(
        stable_idle_frames_required=2
    ))
    _observe_clearance(
        tracker, frame=1, timestamp=1.0,
        previous=RuntimeState.HOOK, current=RuntimeState.RESULT_PENDING,
        get=_get(1, 1.0, False), banner=_banner(1, 1.0, False),
        prompt=PromptObservationKind.UNKNOWN,
    )
    transition_events = _observe_clearance(
        tracker, frame=2, timestamp=2.0,
        previous=RuntimeState.RESULT_PENDING, current=RuntimeState.IDLE,
        get=_get(2, 2.0, False), banner=_banner(2, 2.0, False),
    )
    assert transition_events == ()
    next_events = _observe_clearance(
        tracker, frame=3, timestamp=2.1,
        previous=RuntimeState.IDLE, current=RuntimeState.IDLE,
        get=None, banner=None, mode=DetectorActivationMode.OFF,
    )
    assert [event.event_type for event in next_events] == [
        "no_get_clearance_created"
    ]


def test_detector_off_or_none_is_unknown_not_get_absence() -> None:
    tracker = PostCycleClearanceTracker()
    _observe_clearance(
        tracker, frame=1, timestamp=1.0,
        previous=RuntimeState.HOOK, current=RuntimeState.RESULT_PENDING,
        get=_get(1, 1.0, False), banner=None,
        mode=DetectorActivationMode.OFF,
    )
    _observe_clearance(
        tracker, frame=2, timestamp=2.0,
        previous=RuntimeState.RESULT_PENDING, current=RuntimeState.IDLE,
        get=None, banner=None, mode=DetectorActivationMode.OFF,
    )
    status = tracker.status(2.0)
    assert status.get_presence_state == PresenceState.UNKNOWN
    assert status.result_banner_presence_state == PresenceState.UNKNOWN
    assert tracker.current(2.0) is None


def test_stale_clearance_cannot_cast() -> None:
    tracker = PostCycleClearanceTracker(CastOpportunityConfig(
        clearance_freshness_seconds=1.0,
        stable_idle_frames_required=1,
    ))
    _observe_clearance(
        tracker, frame=1, timestamp=1.0,
        previous=RuntimeState.RESULT_PENDING, current=RuntimeState.IDLE,
        get=_get(1, 1.0, False), banner=_banner(1, 1.0, False),
    )
    assert tracker.current(1.9) is not None
    assert tracker.current(2.01) is None
    assert tracker.status(2.01).clearance_expired is True


def test_clearance_expiry_event_is_throttled() -> None:
    tracker = PostCycleClearanceTracker(CastOpportunityConfig(
        clearance_freshness_seconds=1.0,
        stable_idle_frames_required=1,
    ))
    _observe_clearance(
        tracker, frame=1, timestamp=1.0,
        previous=RuntimeState.RESULT_PENDING, current=RuntimeState.IDLE,
        get=_get(1, 1.0, False), banner=_banner(1, 1.0, False),
    )
    first = _observe_clearance(
        tracker, frame=2, timestamp=2.1,
        previous=RuntimeState.IDLE, current=RuntimeState.IDLE,
        get=None, banner=None, mode=DetectorActivationMode.OFF,
    )
    second = _observe_clearance(
        tracker, frame=3, timestamp=2.2,
        previous=RuntimeState.IDLE, current=RuntimeState.IDLE,
        get=None, banner=None, mode=DetectorActivationMode.OFF,
    )
    assert [event.event_type for event in first] == ["no_get_clearance_expired"]
    assert second == ()


def test_qualified_get_or_open_episode_prevents_no_get_clearance() -> None:
    for episode_open in (False, True):
        tracker = PostCycleClearanceTracker(CastOpportunityConfig(
            stable_idle_frames_required=1
        ))
        _observe_clearance(
            tracker, frame=1, timestamp=1.0,
            previous=RuntimeState.HOOK, current=RuntimeState.RESULT_PENDING,
            get=_get(1, 1.0, episode_open is False),
            banner=_banner(1, 1.0, False), episode_open=episode_open,
        )
        _observe_clearance(
            tracker, frame=2, timestamp=2.0,
            previous=RuntimeState.RESULT_PENDING, current=RuntimeState.IDLE,
            get=_get(2, 2.0, False), banner=_banner(2, 2.0, False),
            episode_open=episode_open,
        )
        assert tracker.current(2.0) is None


def test_result_banner_must_be_observed_absent_after_present_hold() -> None:
    tracker = PostCycleClearanceTracker(CastOpportunityConfig(
        stable_idle_frames_required=1
    ))
    _observe_clearance(
        tracker, frame=1, timestamp=1.0,
        previous=RuntimeState.HOOK, current=RuntimeState.RESULT_PENDING,
        get=_get(1, 1.0, False), banner=_banner(1, 1.0, True),
    )
    _observe_clearance(
        tracker, frame=2, timestamp=2.0,
        previous=RuntimeState.RESULT_PENDING, current=RuntimeState.IDLE,
        get=_get(2, 2.0, False), banner=_banner(2, 2.0, True),
    )
    assert tracker.current(2.0) is None
    events = _observe_clearance(
        tracker, frame=3, timestamp=2.1,
        previous=RuntimeState.IDLE, current=RuntimeState.IDLE,
        get=None, banner=_banner(3, 2.1, False),
        mode=DetectorActivationMode.OFF,
    )
    assert [event.event_type for event in events] == ["no_get_clearance_created"]


def test_one_clearance_is_consumed_once() -> None:
    tracker = PostCycleClearanceTracker(CastOpportunityConfig(
        stable_idle_frames_required=1
    ))
    _observe_clearance(
        tracker, frame=1, timestamp=1.0,
        previous=RuntimeState.RESULT_PENDING, current=RuntimeState.IDLE,
        get=_get(1, 1.0, False), banner=_banner(1, 1.0, False),
    )
    clearance = tracker.current(1.0)
    assert clearance is not None
    assert tracker.consume(clearance.clearance_id) is True
    assert tracker.consume(clearance.clearance_id) is False
    assert tracker.current(1.0) is None


def test_session_132835_deterministic_timeline_yields_one_would_cast() -> None:
    clearance_tracker = PostCycleClearanceTracker(CastOpportunityConfig(
        clearance_freshness_seconds=4.0,
        stable_idle_frames_required=2,
    ))
    _observe_clearance(
        clearance_tracker, frame=1856, timestamp=82.7906535,
        previous=RuntimeState.HOOK, current=RuntimeState.RESULT_PENDING,
        get=None, banner=None, mode=DetectorActivationMode.ARMED,
    )
    _observe_clearance(
        clearance_tracker, frame=1860, timestamp=83.3,
        previous=RuntimeState.RESULT_PENDING,
        current=RuntimeState.RESULT_PENDING,
        get=_get(1860, 83.3, False), banner=_banner(1860, 83.3, False),
    )
    _observe_clearance(
        clearance_tracker, frame=1868, timestamp=84.4862955,
        previous=RuntimeState.RESULT_PENDING, current=RuntimeState.IDLE,
        get=_get(1868, 84.4862955, False),
        banner=_banner(1868, 84.4862955, False),
    )
    clearance = clearance_tracker.current(84.5)
    assert clearance is not None

    controller = CastOpportunityController()
    attempt, _ = controller.schedule(
        timestamp=84.5,
        clearance_id=clearance.clearance_id,
        runtime_state=RuntimeState.IDLE,
        prompt_kind=PromptObservationKind.IDLE_CAST,
        physical_get_episode_open=False,
    )
    assert attempt is not None
    assert clearance_tracker.consume(attempt.clearance_id) is True

    deduplicator = WouldFireDeduplicator()
    request = ActionRequest(ActionIntent.CAST, 0.99, "certified no-GET idle")
    values = dict(
        safety_reason="action_emission_disabled",
        frame_index=1869,
        timestamp=84.5,
        runtime_state=RuntimeState.IDLE.value,
        prompt_evidence={"predicted_label": "IDLE_CAST"},
        specialized_evidence={"get": {"certified_absence": True}},
        identity_suffix=attempt.opportunity_id,
    )
    first = deduplicator.observe(request, **values)
    second = deduplicator.observe(request, **values)
    assert first is not None
    assert first["event_type"] == "WOULD_CAST"
    assert second is None
    assert controller.summary()["cast_opportunity_count"] == 1
    assert clearance_tracker.summary()["no_get_clearance_count"] == 1


def test_cast_is_one_shot_until_waiting_visual_acknowledgement() -> None:
    controller = CastOpportunityController()
    attempt, events = _schedule(controller)
    assert attempt is not None
    assert attempt.action_id == "cast_opportunity:1:CAST"
    assert [item.event_type for item in events] == ["cast_opportunity_started"]
    controller.record_execution(attempt, _execution(attempt.action_id, 0.0), timestamp=0.0)
    assert _schedule(controller, 0.1)[0] is None
    assert controller.observe(
        timestamp=0.5,
        runtime_state=RuntimeState.CAST_PENDING,
        prompt_kind=PromptObservationKind.UNKNOWN,
    ) == ()
    acknowledged = controller.observe(
        timestamp=0.6,
        runtime_state=RuntimeState.WAITING,
        prompt_kind=PromptObservationKind.WAITING_IN_PROGRESS,
    )
    assert [item.event_type for item in acknowledged] == ["cast_visual_acknowledged"]
    next_attempt, _ = _schedule(controller, 2.0)
    assert next_attempt is not None
    assert next_attempt.action_id == "cast_opportunity:2:CAST"


def test_cast_timeout_is_terminal_and_never_retries_across_sync_cycles() -> None:
    controller = CastOpportunityController(CastOpportunityConfig(
        visual_ack_timeout_seconds=4.0
    ))
    attempt, _ = _schedule(controller)
    assert attempt is not None
    controller.record_execution(attempt, _execution(attempt.action_id, 0.0), timestamp=0.0)
    timeout = controller.observe(
        timestamp=4.0,
        runtime_state=RuntimeState.SYNC_REQUIRED,
        prompt_kind=PromptObservationKind.IDLE_CAST,
    )
    assert [item.event_type for item in timeout] == ["cast_visual_timeout"]
    assert timeout[0].payload["retry_scheduled"] is False
    for timestamp in (4.1, 8.0, 20.0):
        assert _schedule(controller, timestamp)[0] is None
    assert controller.summary() == {
        "cast_opportunity_count": 1,
        "cast_attempt_count": 1,
        "cast_visual_acknowledged_count": 0,
        "cast_timeout_count": 1,
    }


def test_rejected_or_partial_cast_is_never_retried() -> None:
    controller = CastOpportunityController()
    attempt, _ = _schedule(controller)
    assert attempt is not None
    controller.record_execution(
        attempt, _execution(attempt.action_id, 0.0, applied=False), timestamp=0.0
    )
    assert _schedule(controller, 1.0)[0] is None
    assert controller.summary()["cast_attempt_count"] == 1
