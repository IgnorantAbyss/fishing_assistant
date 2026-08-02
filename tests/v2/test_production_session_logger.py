import json
from pathlib import Path

from src.fishing_v2.live.session_logger import (
    ProductionSessionLogger,
    cleanup_production_sessions,
)
from tools.run_live_detect_only import parse_args
from tools.run_fishing_production import PRODUCTION_DEFAULTS


def _events(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_repeated_noise_is_summarized_but_action_outcomes_are_not(
    tmp_path: Path,
) -> None:
    logger = ProductionSessionLogger(
        tmp_path,
        bundle_version="test",
        repeat_window_seconds=10.0,
    )
    logger.event("cast_opportunity_blocked", {
        "timestamp": 0.0,
        "reason": "not_ready",
        "runtime_state": "WAITING",
    })
    logger.event("cast_opportunity_blocked", {
        "timestamp": 1.0,
        "reason": "not_ready",
        "runtime_state": "WAITING",
    })
    logger.event("cast_opportunity_blocked", {
        "timestamp": 11.0,
        "reason": "not_ready",
        "runtime_state": "WAITING",
    })
    for timestamp in (2.0, 2.1):
        logger.event("action_failed", {
            "timestamp": timestamp,
            "intent": "CAST",
            "reason": "sendinput_incomplete",
            "runtime_state": "IDLE",
        })
    logger.finalize({
        "result": "completed",
        "duration_seconds": 12.0,
        "final_state": "WAITING",
        "runtime_profile": "production",
    })

    rows = _events(logger.events_path)
    blocked = [row for row in rows if row["event_type"] == "cast_opportunity_blocked"]
    failures = [row for row in rows if row["event_type"] == "action_failed"]
    assert len(blocked) == 2
    assert blocked[-1]["suppressed_count"] == 1
    assert len(failures) == 2


def test_first_io_failure_disables_logger_without_raising(
    tmp_path: Path,
) -> None:
    logger = ProductionSessionLogger(
        tmp_path,
        bundle_version="test",
    )

    def disk_full(_event_type, _row):
        raise OSError("disk full")

    logger._write = disk_full  # type: ignore[method-assign]
    logger.event("action_applied", {
        "timestamp": 1.0,
        "intent": "CAST",
    })
    logger.finalize({
        "result": "completed",
        "duration_seconds": 1.0,
        "final_state": "CAST_PENDING",
        "runtime_profile": "production",
        "actions_applied": 1,
    })

    assert logger.logging_disabled is True
    assert "disk full" in str(logger.logging_failure_reason)


def test_production_screenshot_is_a_zero_copy_noop(tmp_path: Path) -> None:
    logger = ProductionSessionLogger(tmp_path, bundle_version="test")
    sentinel = object()
    assert logger.save_screenshot(sentinel, 1, "transition") == ""
    logger.finalize({
        "result": "completed",
        "duration_seconds": 0.0,
        "final_state": "SYNCING",
        "runtime_profile": "production",
    })
    assert not (logger.path / "screenshots").exists()


def test_cli_and_launcher_define_bounded_production_defaults() -> None:
    args = parse_args([
        "--window-title", "black desert",
        "--duration-seconds", "0",
        "--max-completed-cycles", "0",
    ])
    assert args.runtime_profile == "production"
    assert args.duration_seconds == 0
    assert args.max_completed_cycles == 0
    assert args.log_repeat_window_seconds == 10.0
    assert args.log_max_file_mb == 10.0
    assert args.log_backup_count == 5
    assert args.log_retention_days == 14
    assert args.log_max_total_mb == 100.0
    assert args.hook_action_stall_timeout_seconds == 3.0
    defaults = list(PRODUCTION_DEFAULTS)
    assert defaults[defaults.index("--hook-critical-fps") + 1] == "40"
    assert defaults[
        defaults.index("--hook-action-stall-timeout-seconds") + 1
    ] == "3.0"
    assert defaults[defaults.index("--press-key-hold-ms") + 1] == "40"
    assert defaults[defaults.index("--duration-seconds") + 1] == "0"
    assert defaults[defaults.index("--max-completed-cycles") + 1] == "0"
    assert "--enable-live-press-sequence" in defaults


def test_retention_cleanup_never_removes_diagnostic_sessions(
    tmp_path: Path,
) -> None:
    production = ProductionSessionLogger(tmp_path, bundle_version="test")
    production.finalize({
        "result": "completed",
        "duration_seconds": 0.0,
        "final_state": "IDLE",
        "runtime_profile": "production",
    })
    production_path = production.path
    diagnostic_path = tmp_path / "session_diagnostic_fixture"
    (diagnostic_path / "diagnostic_evidence").mkdir(parents=True)
    (diagnostic_path / "events.jsonl").write_text("", encoding="utf-8")

    warnings = cleanup_production_sessions(
        tmp_path,
        retention_days=14,
        max_total_mb=0,
    )

    assert warnings == []
    assert not production_path.exists()
    assert diagnostic_path.exists()
