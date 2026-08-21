from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.fishing_v2.domain.observations import (
    HookObservation,
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.live.hook_pending_timeout_evidence import (
    HookPendingTimeoutEvidenceConfig,
    HookPendingTimeoutEvidenceRecorder,
)
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode
from src.fishing_v2.runtime.detector_evidence import (
    EvidenceQualification,
    HookEvidenceKind,
)


def _begin(
    recorder: HookPendingTimeoutEvidenceRecorder,
    *,
    applied: bool = True,
    started: float | None = 10.0,
    completed: float | None = 10.04,
) -> bool:
    return recorder.begin(
        runtime_state=RuntimeState.HOOK_PENDING,
        cycle_id=7,
        ready_identity="ready:4",
        start_hook_opportunity_id="ready:4:START_HOOK",
        start_hook_action_id="action:start-hook:4",
        start_hook_emission_started_at=started,
        start_hook_applied_at=completed,
        hook_pending_started_at=10.04,
        start_hook_action_applied=applied,
    )


def _hook(
    frame: int,
    *,
    detected: bool = True,
    divider: bool = True,
    fill: bool = True,
    confidence: float = 0.9,
) -> HookObservation:
    features = ["hook_bar_rect"] if detected else []
    if fill:
        features.append("bar_fill")
    if divider:
        features.append("divider_line")
    return HookObservation(
        detected=detected,
        confidence=confidence,
        frame_index=frame,
        timestamp=10.04 + frame / 40.0,
        fill_ratio=0.72 if fill else None,
        divider_ratio=0.58 if divider else None,
        evidence={
            "crossing_geometry_version": 1,
            "matched_features": features,
            "bar_bbox": [973, 468, 1587, 490] if detected else None,
            "divider_line_detected": divider,
            "divider_line_x": 1330.0 if divider else None,
            "divider_confidence": 1.0 if divider else 0.0,
            "fill_endpoint_x": 1397.0 if fill else None,
            "legacy_debug": {
                "roi_name": "hook_bar_precise",
                "raw_values": {
                    "red_pixels": 812 if detected else 0,
                    "cyan_pixels": 604 if fill else 0,
                },
            },
        },
    )


def _qualification(
    *,
    raw: bool = True,
    qualified: bool = True,
) -> EvidenceQualification:
    return EvidenceQualification(
        "hook",
        DetectorActivationMode.BURST,
        raw,
        qualified,
        (
            "active_bar_fill_qualified"
            if qualified else "active_bar_below_strong_confidence"
        ),
        qualified,
        False,
        hook_evidence_kind=(
            HookEvidenceKind.ACTIVE_HOOK_BAR
            if raw else HookEvidenceKind.REJECTED
        ),
    )


def _prompt(frame: int) -> PromptObservation:
    return PromptObservation(
        PromptObservationKind.HOOK_INSTRUCTION,
        0.97,
        {PromptObservationKind.HOOK_INSTRUCTION.value: 0.97},
        "test",
        frame,
        10.04 + frame / 40.0,
    )


def _record(
    recorder: HookPendingTimeoutEvidenceRecorder,
    *,
    frame: int,
    executed: bool = True,
    detected: bool = True,
    divider: bool = True,
    fill: bool = True,
    qualified: bool = True,
    roi_shape: tuple[int, int, int] = (12, 20, 3),
    transition_reason: str = "candidate_not_stable",
) -> bool:
    raw = (
        _hook(
            frame,
            detected=detected,
            divider=divider,
            fill=fill,
        )
        if executed else None
    )
    qualification = (
        _qualification(raw=detected, qualified=qualified)
        if executed else None
    )
    return recorder.record(
        frame_index=frame,
        timestamp=10.04 + frame / 40.0,
        runtime_state=RuntimeState.HOOK_PENDING,
        roi_pixels=np.full(roi_shape, frame % 255, dtype=np.uint8),
        hook_detector_executed=executed,
        detector_mode=DetectorActivationMode.BURST,
        cadence={
            "hook_critical_mode": True,
            "detector_due": True,
            "target_fps": 40.0,
        },
        prompt=_prompt(frame),
        raw_observation=raw,
        qualified_observation=raw,
        qualification=qualification,
        fsm_diagnostics={
            "previous_state": RuntimeState.HOOK_PENDING.value,
            "committed_state": (
                RuntimeState.HOOK.value
                if transition_reason == "stable_strong_hook_evidence"
                else RuntimeState.HOOK_PENDING.value
            ),
            "transition_reason": transition_reason,
            "recommended_state": (
                RuntimeState.HOOK.value if qualified else None
            ),
            "fusion_reason": (
                "strong_hook_evidence" if qualified
                else "insufficient_observation_evidence_no_fallback"
            ),
        },
    )


def _timeout(recorder: HookPendingTimeoutEvidenceRecorder) -> bool:
    return recorder.finish(
        next_state=RuntimeState.SYNC_REQUIRED,
        timestamp=13.04,
        reason="hook_pending_timeout",
    )


def test_only_applied_start_hook_in_hook_pending_starts_recorder(
    tmp_path: Path,
) -> None:
    recorder = HookPendingTimeoutEvidenceRecorder(
        tmp_path,
        HookPendingTimeoutEvidenceConfig(enabled=True),
    )
    assert not recorder.begin(
        runtime_state=RuntimeState.READY,
        cycle_id=1,
        ready_identity="ready:1",
        start_hook_opportunity_id="ready:1:START_HOOK",
        start_hook_action_id=None,
        start_hook_emission_started_at=None,
        start_hook_applied_at=None,
        hook_pending_started_at=1.0,
        start_hook_action_applied=False,
    )
    assert not _begin(recorder, applied=False)
    assert not _begin(recorder, started=None)
    assert _begin(recorder)
    assert recorder.active
    recorder.close()


@pytest.mark.parametrize(
    ("next_state", "reason"),
    [
        (RuntimeState.HOOK, "stable_strong_hook_evidence"),
        (RuntimeState.IDLE, "authoritative_idle_prompt_recovery"),
        (RuntimeState.SYNC_REQUIRED, "persistent_conflicting_or_illegal_evidence"),
    ],
)
def test_non_timeout_exit_discards_ram_and_writes_nothing(
    tmp_path: Path,
    next_state: RuntimeState,
    reason: str,
) -> None:
    recorder = HookPendingTimeoutEvidenceRecorder(
        tmp_path,
        HookPendingTimeoutEvidenceConfig(enabled=True),
    )
    assert _begin(recorder)
    _record(
        recorder,
        frame=1,
        transition_reason=(
            "stable_strong_hook_evidence"
            if next_state == RuntimeState.HOOK else "state_held"
        ),
    )
    assert not recorder.finish(
        next_state=next_state,
        timestamp=10.5,
        reason=reason,
    )
    summary = recorder.close()
    assert summary["hook_pending_timeout_evidence_saved_count"] == 0
    assert not (tmp_path / "hook_pending_anomalies").exists()


def test_timeout_dump_distinguishes_detector_and_detection_layers(
    tmp_path: Path,
) -> None:
    recorder = HookPendingTimeoutEvidenceRecorder(
        tmp_path,
        HookPendingTimeoutEvidenceConfig(enabled=True),
    )
    assert _begin(recorder)
    _record(recorder, frame=1, executed=False)
    _record(recorder, frame=2, detected=False, divider=False, fill=False, qualified=False)
    _record(recorder, frame=3, divider=False, fill=True, qualified=False)
    _record(recorder, frame=4, divider=True, fill=False, qualified=False)
    _record(recorder, frame=5, divider=True, fill=True, qualified=False)
    assert _timeout(recorder)
    summary = recorder.close()
    manifest_path = next(tmp_path.rglob("manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]

    assert summary["hook_pending_timeout_evidence_saved_count"] == 1
    assert samples[0]["hook_detector_executed"] is False
    assert samples[0]["raw_observation"]["status"] == "not_evaluated"
    assert samples[0]["qualification"]["status"] == "not_evaluated"
    assert samples[1]["raw_observation"]["locator_success"] is False
    assert samples[2]["raw_observation"]["divider_found"] is False
    assert samples[3]["raw_observation"]["fill_found"] is False
    assert samples[4]["raw_observation"]["geometry_valid"] is True
    assert samples[4]["qualification"]["strong_hook_evidence"] is False
    assert all(
        (manifest_path.parent / item["raw_image_filename"]).is_file()
        for item in samples
    )


def test_stable_strong_transition_is_captured_then_discarded(
    tmp_path: Path,
) -> None:
    recorder = HookPendingTimeoutEvidenceRecorder(
        tmp_path,
        HookPendingTimeoutEvidenceConfig(enabled=True),
    )
    _begin(recorder)
    _record(
        recorder,
        frame=8,
        transition_reason="stable_strong_hook_evidence",
    )
    assert not recorder.finish(
        next_state=RuntimeState.HOOK,
        timestamp=10.3,
        reason="stable_strong_hook_evidence",
    )
    assert recorder.close()["hook_pending_timeout_evidence_saved_count"] == 0


def test_sample_and_episode_caps_and_new_episode_identity(
    tmp_path: Path,
) -> None:
    recorder = HookPendingTimeoutEvidenceRecorder(
        tmp_path,
        HookPendingTimeoutEvidenceConfig(
            enabled=True,
            max_samples=3,
            max_episodes=1,
        ),
    )
    _begin(recorder)
    for frame in range(1, 7):
        _record(recorder, frame=frame)
    assert recorder.sample_count == 3
    assert _timeout(recorder)
    assert _begin(recorder)
    _record(recorder, frame=20)
    assert not _timeout(recorder)
    summary = recorder.close()
    manifest = json.loads(next(tmp_path.rglob("manifest.json")).read_text())
    assert [item["frame_index"] for item in manifest["samples"]] == [4, 5, 6]
    assert manifest["episode_id"] == "hook_pending:1"
    assert summary["hook_pending_timeout_evidence_cap_drop_count"] == 1


def test_production_roi_worst_case_ram_is_bounded(tmp_path: Path) -> None:
    recorder = HookPendingTimeoutEvidenceRecorder(
        tmp_path,
        HookPendingTimeoutEvidenceConfig(enabled=True, max_samples=140),
    )
    _begin(recorder)
    _record(recorder, frame=1, roi_shape=(58, 614, 3))
    assert recorder.worst_case_ram_bytes == 614 * 58 * 3 * 140
    recorder.close()


def test_io_failure_is_fail_open_and_does_not_touch_control_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = HookPendingTimeoutEvidenceRecorder(
        tmp_path,
        HookPendingTimeoutEvidenceConfig(enabled=True),
    )
    calls = {"capture": 0, "detector": 0, "sink": 0}

    def fail_write(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(recorder, "_write_episode", fail_write)
    _begin(recorder)
    _record(recorder, frame=1)
    assert _timeout(recorder)
    summary = recorder.close()
    assert summary["hook_pending_timeout_evidence_failure_count"] == 1
    assert calls == {"capture": 0, "detector": 0, "sink": 0}


def test_disabled_recorder_has_no_behavior_or_io(tmp_path: Path) -> None:
    recorder = HookPendingTimeoutEvidenceRecorder(tmp_path)
    assert not _begin(recorder)
    assert not _record(recorder, frame=1)
    assert not _timeout(recorder)
    summary = recorder.close()
    assert summary["hook_pending_timeout_evidence_saved_count"] == 0
    assert list(tmp_path.iterdir()) == []
