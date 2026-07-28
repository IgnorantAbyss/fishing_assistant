from __future__ import annotations

import numpy as np
import pytest

from src.fishing_v2.live.hook_critical_loop import (
    HookCriticalFrameAssembler,
    HookDecisionTraceBuffer,
    HookEpisodeTelemetry,
    HookROIFrame,
    LatestHookFrameSlot,
)


def test_latest_hook_frame_slot_replaces_unconsumed_stale_frame() -> None:
    slot = LatestHookFrameSlot()
    old = HookROIFrame(10, 1.0, np.zeros((2, 3, 3), dtype=np.uint8))
    latest = HookROIFrame(11, 1.02, np.ones((2, 3, 3), dtype=np.uint8))

    slot.publish(old)
    slot.publish(latest)

    assert slot.take_latest() is latest
    assert slot.take_latest() is None
    assert slot.stale_frames_dropped == 1


def test_hook_decision_trace_is_bounded_until_drained() -> None:
    trace = HookDecisionTraceBuffer(max_entries=2)
    trace.start("episode-5")
    trace.record({"capture_frame_index": 6208})
    trace.record({"capture_frame_index": 6210})
    trace.record({"capture_frame_index": 6212})

    rows = trace.drain("episode_ended")

    assert [row["capture_frame_index"] for row in rows] == [6210, 6212]
    assert all(row["hook_episode_id"] == "episode-5" for row in rows)
    assert all(row["trace_dropped_entries"] == 1 for row in rows)
    assert all(row["trace_flush_reason"] == "episode_ended" for row in rows)
    assert trace.active is False


def test_hook_episode_telemetry_measures_real_thirty_fps_timestamps() -> None:
    telemetry = HookEpisodeTelemetry()
    telemetry.start(0.0, 2)
    for index in range(31):
        telemetry.record_detector_frame(index / 30.0)
    telemetry.finish(1.0, 3)

    summary = telemetry.summaries()[0]
    assert summary["hook_detector_frame_count"] == 31
    assert summary["hook_detector_active_duration"] == 1.0
    assert summary["hook_detector_actual_fps"] == 30.0
    assert summary["hook_frame_interval_mean_ms"] == pytest.approx(
        1000.0 / 30.0
    )
    assert summary["hook_frame_interval_p95_ms"] == pytest.approx(
        1000.0 / 30.0
    )
    assert summary["hook_frame_interval_max_ms"] == pytest.approx(
        1000.0 / 30.0
    )
    assert summary["stale_hook_frames_dropped"] == 1
    assert summary["max_gap_diagnostic"] is None


def test_hook_frame_assembler_preserves_cached_prompt_and_latest_roi() -> None:
    assembler = HookCriticalFrameAssembler(
        (8, 6),
        (2, 3, 6, 5),
        (1, 0, 7, 2),
    )
    full = np.zeros((6, 8, 3), dtype=np.uint8)
    full[0:2, 1:7] = 7
    assembler.update_prompt_context(full)

    composed = assembler.compose(
        np.full((2, 4, 3), 13, dtype=np.uint8)
    )

    assert np.all(composed[0:2, 1:7] == 7)
    assert np.all(composed[3:5, 2:6] == 13)
    assert np.all(composed[2, :] == 0)
