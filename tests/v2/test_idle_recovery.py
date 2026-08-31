from dataclasses import replace

import pytest

from src.fishing_v2.domain.observations import (
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.live.idle_recovery import (
    CAST_SOURCE_POST_CYCLE,
    CAST_SOURCE_RECOVERY,
    CAST_SOURCE_STARTUP,
    CastArmingLifecycle,
    CastArmingPreparationStatus,
    IdleRecoveryConfig,
    IdleRecoveryTracker,
)


def _prompt(frame: int, timestamp: float, kind=PromptObservationKind.IDLE_CAST):
    return PromptObservation(
        kind,
        0.99,
        {kind.value: 0.99},
        "test",
        frame,
        timestamp,
    )


def _observe(
    tracker: IdleRecoveryTracker,
    prompt: PromptObservation | None,
    timestamp: float,
    **overrides,
):
    values = {
        "conflicting_evidence": False,
        "action_emission_in_progress": False,
        "key_currently_down": False,
        "panic_triggered": False,
    }
    values.update(overrides)
    return tracker.observe(prompt, timestamp=timestamp, **values)


def test_idle_certificate_requires_four_of_five_over_half_second() -> None:
    tracker = IdleRecoveryTracker()
    kinds = (
        PromptObservationKind.IDLE_CAST,
        PromptObservationKind.UNKNOWN,
        PromptObservationKind.IDLE_CAST,
        PromptObservationKind.IDLE_CAST,
        PromptObservationKind.IDLE_CAST,
    )
    certificate = None
    for frame, (timestamp, kind) in enumerate(
        zip((0.0, 0.13, 0.26, 0.39, 0.52), kinds), start=1
    ):
        certificate, _ = _observe(
            tracker, _prompt(frame, timestamp, kind), timestamp
        )
    assert certificate is not None
    assert certificate.observation_count == 5
    assert certificate.idle_count == 4
    assert certificate.window_end - certificate.window_start >= 0.5


def test_idle_certificate_refreshes_window_and_rejects_pre_sync_snapshot() -> None:
    tracker = IdleRecoveryTracker()
    certificate = None
    for frame, timestamp in enumerate(
        (0.0, 0.13, 0.26, 0.39, 0.52), start=1
    ):
        certificate, _ = _observe(
            tracker, _prompt(frame, timestamp), timestamp
        )
    assert certificate is not None
    stale = replace(certificate, latest_observation_age_ms=0.0)
    assert not stale.is_fresh_after(10.0)

    refreshed = None
    for frame, timestamp in enumerate(
        (10.1, 10.23, 10.36, 10.49, 10.62), start=6
    ):
        refreshed, _ = _observe(
            tracker, _prompt(frame, timestamp), timestamp
        )
    assert refreshed is not None
    assert refreshed.certificate_id != certificate.certificate_id
    assert refreshed.window_start == pytest.approx(10.1)
    assert refreshed.window_end == pytest.approx(10.62)
    assert refreshed.is_fresh_after(10.0)


@pytest.mark.parametrize(
    "blocker",
    (
        "conflicting_evidence",
        "action_emission_in_progress",
        "key_currently_down",
        "panic_triggered",
    ),
)
def test_idle_certificate_fails_closed_for_safety_blockers(blocker: str) -> None:
    tracker = IdleRecoveryTracker()
    for frame, timestamp in enumerate((0.0, 0.13, 0.26, 0.39), start=1):
        _observe(tracker, _prompt(frame, timestamp), timestamp)
    certificate, events = _observe(
        tracker,
        _prompt(5, 0.52),
        0.52,
        **{blocker: True},
    )
    assert certificate is None
    assert events[-1].event_type == "idle_recovery_candidate_cancelled"


def test_single_idle_frame_never_certifies() -> None:
    certificate, _ = _observe(
        IdleRecoveryTracker(), _prompt(1, 0.0), 0.0
    )
    assert certificate is None


def test_cast_arming_unifies_sources_and_deduplicates_physical_idle() -> None:
    lifecycle = CastArmingLifecycle(retry_min_interval_seconds=3.0)
    first, events = lifecycle.request_cast_opportunity(
        source_type=CAST_SOURCE_STARTUP,
        source_id="startup:1",
        physical_idle_id="idle:1",
        cycle_id=0,
        timestamp=1.0,
        cooldown_seconds=0.5,
    )
    assert first is not None and events == ()
    assert not lifecycle.ready(
        timestamp=1.49, runtime_idle=True, idle_certificate_valid=True
    )
    assert lifecycle.ready(
        timestamp=1.5, runtime_idle=True, idle_certificate_valid=True
    )
    duplicate, events = lifecycle.request_cast_opportunity(
        source_type=CAST_SOURCE_POST_CYCLE,
        source_id="clearance:1",
        physical_idle_id="idle:1",
        cycle_id=0,
        timestamp=1.5,
        cooldown_seconds=0.0,
    )
    assert duplicate is None
    assert events[0].event_type == "cast_opportunity_deduplicated"
    assert events[0].payload["dedupe_reason"] == "physical_idle_source_merged"


def test_same_cast_source_id_is_exactly_once() -> None:
    lifecycle = CastArmingLifecycle()
    record, _ = lifecycle.request_cast_opportunity(
        source_type=CAST_SOURCE_RECOVERY,
        source_id="recovery:1",
        physical_idle_id="idle:1",
        cycle_id=4,
        timestamp=0.0,
        cooldown_seconds=0.5,
    )
    assert record is not None
    lifecycle.cancel("test")
    duplicate, events = lifecycle.request_cast_opportunity(
        source_type=CAST_SOURCE_RECOVERY,
        source_id="recovery:1",
        physical_idle_id="idle:1",
        cycle_id=4,
        timestamp=1.0,
        cooldown_seconds=0.5,
    )
    assert duplicate is None
    assert events[0].payload["dedupe_reason"] == "source_id_already_seen"


def test_cast_retry_is_delayed_and_bounded_to_one_retry() -> None:
    lifecycle = CastArmingLifecycle(retry_min_interval_seconds=3.0)

    def arm(source_id: str, timestamp: float):
        return lifecycle.request_cast_opportunity(
            source_type=CAST_SOURCE_RECOVERY,
            source_id=source_id,
            physical_idle_id="idle:1",
            cycle_id=7,
            timestamp=timestamp,
            cooldown_seconds=0.5,
        )[0]

    first = arm("recovery:1", 0.0)
    assert first is not None
    lifecycle.mark_cast_started("cast:1")
    lifecycle.record_execution(
        timestamp=0.5,
        emission_started=True,
        applied=True,
        terminal_outcome="applied",
    )
    assert lifecycle.authorize_retry_after_visual_timeout() == "idle:1"
    retry = arm("recovery:2", 1.0)
    assert retry is not None
    assert retry.recovery_generation == 1
    assert retry.recovery_budget_consumed == 1
    assert retry.recovery_budget_remaining == 1
    assert retry.payload(1.0)["dedupe_key"] == (
        "idle:1:generation:1"
    )
    assert retry.eligible_at == pytest.approx(3.5)
    lifecycle.mark_cast_started("cast:2")
    lifecycle.record_execution(
        timestamp=3.5,
        emission_started=True,
        applied=True,
        terminal_outcome="applied",
    )
    blocked, events = lifecycle.request_cast_opportunity(
        source_type=CAST_SOURCE_RECOVERY,
        source_id="recovery:3",
        physical_idle_id="idle:1",
        cycle_id=8,
        timestamp=8.0,
        cooldown_seconds=0.5,
    )
    assert blocked is None
    assert events[0].payload["dedupe_reason"] == (
        "physical_idle_retry_limit_reached"
    )
    assert events[0].payload["recovery_budget_remaining"] == 0


def test_cast_recovery_prepare_is_transactional_and_reports_retry_limit() -> None:
    lifecycle = CastArmingLifecycle(retry_min_interval_seconds=3.0)

    for generation in range(2):
        if generation:
            assert lifecycle.authorize_retry_after_visual_timeout() == "idle:1"
        source_id = f"recovery:{generation}"
        prepared = lifecycle.prepare_cast_opportunity(
            source_type=CAST_SOURCE_RECOVERY,
            source_id=source_id,
            physical_idle_id="idle:1",
            cycle_id=7,
        )
        assert prepared.status == CastArmingPreparationStatus.SERVICEABLE
        record, _ = lifecycle.request_cast_opportunity(
            source_type=CAST_SOURCE_RECOVERY,
            source_id=source_id,
            physical_idle_id="idle:1",
            cycle_id=7,
            timestamp=float(generation * 4),
            cooldown_seconds=0.0,
        )
        assert record is not None
        lifecycle.mark_cast_started(f"cast:{generation}")
        lifecycle.record_execution(
            timestamp=float(generation * 4),
            emission_started=True,
            applied=True,
            terminal_outcome="applied",
        )

    exhausted = lifecycle.prepare_cast_opportunity(
        source_type=CAST_SOURCE_RECOVERY,
        source_id="recovery:2",
        physical_idle_id="idle:1",
        cycle_id=7,
    )
    assert exhausted.status == (
        CastArmingPreparationStatus.RETRY_LIMIT_REACHED
    )
    assert not exhausted.serviceable
    assert exhausted.blocker == "physical_idle_retry_limit_reached"
    assert exhausted.recovery_budget_remaining == 0
    assert lifecycle.active is None


def test_consumed_physical_idle_deduplicates_other_sources_without_timeout() -> None:
    lifecycle = CastArmingLifecycle()
    first, _ = lifecycle.request_cast_opportunity(
        source_type=CAST_SOURCE_STARTUP,
        source_id="startup:1",
        physical_idle_id="idle:1",
        cycle_id=0,
        timestamp=0.0,
        cooldown_seconds=0.0,
    )
    assert first is not None
    lifecycle.mark_cast_started("cast:1")
    lifecycle.record_execution(
        timestamp=0.1,
        emission_started=True,
        applied=True,
        terminal_outcome="applied",
    )
    duplicate, events = lifecycle.request_cast_opportunity(
        source_type=CAST_SOURCE_POST_CYCLE,
        source_id="clearance:1",
        physical_idle_id="idle:1",
        cycle_id=0,
        timestamp=5.0,
        cooldown_seconds=0.0,
    )
    assert duplicate is None
    assert events[0].payload["dedupe_reason"] == (
        "physical_idle_already_consumed"
    )


def test_cast_proposal_does_not_consume_before_emission() -> None:
    lifecycle = CastArmingLifecycle()
    record, _ = lifecycle.request_cast_opportunity(
        source_type=CAST_SOURCE_POST_CYCLE,
        source_id="clearance:1",
        physical_idle_id="idle:1",
        cycle_id=1,
        timestamp=0.0,
        cooldown_seconds=0.0,
    )
    assert record is not None
    lifecycle.mark_cast_started("cast:1")
    assert lifecycle.active is not None
    assert lifecycle.active.cast_started is True
    assert lifecycle.active.consumed is False


def test_cast_sources_for_same_physical_idle_are_merged_not_dropped() -> None:
    lifecycle = CastArmingLifecycle()
    recovery, _ = lifecycle.request_cast_opportunity(
        source_type=CAST_SOURCE_RECOVERY,
        source_id="recovery:1",
        physical_idle_id="idle:1",
        cycle_id=1,
        timestamp=0.0,
        cooldown_seconds=0.5,
    )
    assert recovery is not None

    duplicate, events = lifecycle.request_cast_opportunity(
        source_type=CAST_SOURCE_POST_CYCLE,
        source_id="clearance:1",
        physical_idle_id="idle:1",
        cycle_id=1,
        timestamp=0.2,
        cooldown_seconds=0.0,
    )

    assert duplicate is None
    assert lifecycle.active is recovery
    assert events[0].payload["dedupe_reason"] == "physical_idle_source_merged"
    assert "clearance:1" in lifecycle.active.merged_source_ids


def test_unstarted_cast_arm_can_expire_instead_of_hanging_forever() -> None:
    lifecycle = CastArmingLifecycle()
    record, _ = lifecycle.request_cast_opportunity(
        source_type=CAST_SOURCE_RECOVERY,
        source_id="recovery:1",
        physical_idle_id="idle:1",
        cycle_id=1,
        timestamp=0.0,
        cooldown_seconds=0.5,
    )
    assert record is not None

    assert lifecycle.expire_if_overdue(
        timestamp=3.49,
        service_timeout_seconds=3.0,
    ) is None
    expired = lifecycle.expire_if_overdue(
        timestamp=3.5,
        service_timeout_seconds=3.0,
    )

    assert expired is record
    assert expired.terminal_outcome == "expired:cast_arm_service_timeout"
    assert lifecycle.active is None
