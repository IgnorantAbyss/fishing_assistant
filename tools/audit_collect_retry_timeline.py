"""Offline-only COLLECT retry scheduling audit for one saved Live session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fishing_v2.live.collect_retry import (  # noqa: E402
    CollectRetryConfig,
    CollectRetryController,
)
from src.fishing_v2.ports.action_sink import ActionExecutionResult  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_session(session: Path, config_path: Path) -> dict:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    controller = CollectRetryController(
        CollectRetryConfig.from_mapping(raw_config.get("collect"))
    )
    evidence_path = session / "diagnostic_evidence" / "detector_evidence.jsonl"
    records = _read_jsonl(evidence_path)
    events = _read_jsonl(session / "events.jsonl")
    scheduled: list[dict] = []
    lifecycle_events: list[dict] = []
    opportunity = "cycle:1:COLLECT"
    for record in records:
        detector = record.get("detectors", {}).get("get", {})
        if not detector.get("executed"):
            continue
        timestamp = float(record["timestamp"])
        qualified = detector.get("qualified") or {}
        diagnostics = detector.get("diagnostics") or {}
        visible = bool(qualified.get("qualified_detected"))
        observed_events = controller.observe_panel(
            opportunity_id=opportunity,
            timestamp=timestamp,
            panel_observed=True,
            panel_visible=visible,
            get_confidence=float(diagnostics.get("confidence") or 0.0),
            get_confirmation_frames=int(
                diagnostics.get("temporal_confirmation_count") or 0
            ),
        )
        lifecycle_events.extend({
            "event_type": item.event_type,
            "timestamp": timestamp,
            **dict(item.payload),
        } for item in observed_events)
        if not visible:
            continue
        attempt, schedule_events = controller.schedule_attempt(
            timestamp=timestamp,
            get_confidence=float(diagnostics.get("confidence") or 0.0),
            get_confirmation_frames=int(
                diagnostics.get("temporal_confirmation_count") or 0
            ),
        )
        lifecycle_events.extend({
            "event_type": item.event_type,
            "timestamp": timestamp,
            **dict(item.payload),
        } for item in schedule_events)
        if attempt is None:
            continue
        scheduled.append({
            "attempt_id": attempt.attempt_id,
            "attempt_number": attempt.attempt_number,
            "frame_index": int(record["capture_frame_index"]),
            "timestamp": timestamp,
            "elapsed_since_get_appearance": attempt.elapsed_seconds,
        })
        # Scheduling-only simulation: construct a result object directly. No
        # ActionSink or Windows input API is instantiated or called.
        simulated = ActionExecutionResult(
            action_id=attempt.attempt_id,
            intent_type="COLLECT",
            requested_at=timestamp,
            started_at=timestamp,
            completed_at=timestamp,
            success=True,
            applied=True,
            emitted_event_count=2,
            expected_event_count=2,
            target_hwnd=None,
            foreground_hwnd=None,
            os_input_emitted=True,
        )
        execution_events = controller.record_execution(
            attempt, simulated, timestamp=timestamp
        )
        lifecycle_events.extend({
            "event_type": item.event_type,
            "timestamp": timestamp,
            "simulation_only": True,
            **dict(item.payload),
        } for item in execution_events)

    appearance = next(
        (float(item["timestamp"]) for item in events if item["event_type"] == "get_appearance"),
        None,
    )
    old_attempt = next(
        (float(item["timestamp"]) for item in events if item["event_type"] == "WOULD_COLLECT"),
        None,
    )
    return {
        "session": str(session),
        "mode": "offline_scheduling_only_no_os_input",
        "get_appearance": appearance,
        "old_first_attempt": old_attempt,
        "old_first_attempt_delay_ms": (
            (old_attempt - appearance) * 1000.0
            if old_attempt is not None and appearance is not None else None
        ),
        "candidate_attempts": scheduled,
        "candidate_first_attempt_delay_ms": (
            (scheduled[0]["timestamp"] - appearance) * 1000.0
            if scheduled and appearance is not None else None
        ),
        "lifecycle_events": lifecycle_events,
        "summary": controller.summary(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "fishing_v2.yaml"
    )
    args = parser.parse_args()
    result = audit_session(args.session.resolve(), args.config.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
