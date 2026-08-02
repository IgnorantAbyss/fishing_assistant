"""Observe the game; optionally apply explicitly allowlisted foreground actions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.live.live_detect_only import (  # noqa: E402
    LiveDetectOnlyConfig,
    LiveDetectOnlyRuntime,
    LivePreflightError,
    RUNTIME_PROFILES,
    validate_emit_actions,
)
from src.fishing_v2.live.capture_backends import (  # noqa: E402
    CAPTURE_BACKENDS,
    MSS_REGION_BACKEND,
    WINDOWS_GRAPHICS_CAPTURE_BACKEND,
    create_live_capture_session,
)
from src.fishing_v2.live.diagnostic_evidence import EVIDENCE_MODES  # noqa: E402
from src.fishing_v2.live.windows_action_sink import (  # noqa: E402
    ACTION_SINKS,
    ACTION_SINK_NONE,
)
from src.fishing_v2.live.window_resolver import (  # noqa: E402
    WindowResolutionError,
    format_window_candidates,
    resolve_window_target,
)
from src.fishing_v2.live.session_logger import (  # noqa: E402
    LiveSessionLogger,
    ProductionSessionLogger,
)
from src.fishing_v2.perception.prompt_bundle import load_prompt_bundle  # noqa: E402
from src.screen_capture import normalize_process_name  # noqa: E402


def _strict_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "false":
        return False
    if normalized == "true":
        return True
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--window-title",
        help="Exact visible borderless game-window title",
    )
    target.add_argument(
        "--process-name",
        help="Auto-resolve one visible top-level window by executable basename",
    )
    parser.add_argument(
        "--window-title-prefix",
        help="Optional title prefix used only with --process-name",
    )
    parser.add_argument(
        "--capture-backend",
        choices=CAPTURE_BACKENDS,
        default=MSS_REGION_BACKEND,
        help="Explicit capture source (default: mss-region desktop pixels)",
    )
    parser.add_argument(
        "--allow-mss-fallback",
        action="store_true",
        help="Allow WGC initialization failure to fall back to visible desktop-region pixels",
    )
    parser.add_argument("--duration-seconds", type=float, default=180.0)
    parser.add_argument(
        "--runtime-profile",
        choices=RUNTIME_PROFILES,
        default="production",
        help="Lightweight text-only production or full diagnostic runtime",
    )
    parser.add_argument(
        "--evidence-mode", choices=EVIDENCE_MODES, default="minimal",
        help="minimal keeps event-only screenshots; diagnostic adds video and dense ROI evidence",
    )
    parser.add_argument(
        "--evidence-video-fps", type=float, default=10.0,
        help="Diagnostic full-session video sampling rate (default: 10)",
    )
    parser.add_argument(
        "--max-completed-cycles", type=int,
        help="Stop after this many complete Runtime cycles; 0 means unlimited",
    )
    parser.add_argument("--log-repeat-window-seconds", type=float, default=10.0)
    parser.add_argument("--log-max-file-mb", type=float, default=10.0)
    parser.add_argument("--log-backup-count", type=int, default=5)
    parser.add_argument("--log-retention-days", type=int, default=14)
    parser.add_argument("--log-max-total-mb", type=float, default=100.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "fishing_v2" / "live_detect_only",
    )
    overlay = parser.add_mutually_exclusive_group()
    overlay.add_argument("--show-overlay", dest="show_overlay", action="store_true")
    overlay.add_argument("--no-overlay", dest="show_overlay", action="store_false")
    parser.set_defaults(show_overlay=True)
    parser.add_argument("--save-transition-frames", action="store_true")
    parser.add_argument(
        "--prompt-bundle", type=Path,
        default=PROJECT_ROOT / "artifacts" / "prompt_observer" / "prototype_v1",
    )
    parser.add_argument("--max-fps", type=float, default=25.0)
    parser.add_argument(
        "--hook-critical-fps",
        type=float,
        default=40.0,
        help=(
            "Native precise-ROI capture target while HOOK_PENDING/HOOK "
            "(minimum 30, default: 40)"
        ),
    )
    parser.add_argument(
        "--hook-action-stall-timeout-seconds",
        type=float,
        default=3.0,
        help=(
            "Re-evaluate or resynchronize an unconsumed HOOK action "
            "opportunity after this state age (default: 3.0)"
        ),
    )
    parser.add_argument(
        "--idle-recovery-window-size", type=int, default=5,
    )
    parser.add_argument(
        "--idle-recovery-required-count", type=int, default=4,
    )
    parser.add_argument(
        "--idle-recovery-min-window-seconds", type=float, default=0.5,
    )
    parser.add_argument(
        "--idle-recovery-freshness-ms", type=float, default=250.0,
    )
    parser.add_argument(
        "--idle-recovery-cast-cooldown-seconds", type=float, default=0.5,
    )
    parser.add_argument(
        "--idle-cast-retry-min-interval-seconds", type=float, default=3.0,
    )
    parser.add_argument("--emit-actions", type=_strict_bool, default=False)
    parser.add_argument(
        "--action-sink", choices=ACTION_SINKS, default=ACTION_SINK_NONE,
        help="Input backend; defaults to none and is never initialized in detect-only mode",
    )
    parser.add_argument(
        "--action-allowlist", default="",
        help=(
            "Comma-separated staged Live actions: "
            "CAST,START_HOOK,HOOK_ACTION,COLLECT; PRESS_SEQUENCE "
            "also requires --enable-live-press-sequence"
        ),
    )
    parser.add_argument(
        "--enable-live-press-sequence",
        action="store_true",
        help=(
            "Explicitly enable guarded PRESS_SEQUENCE emission; also "
            "requires emit-actions, sendinput, and PRESS_SEQUENCE in "
            "the action allowlist"
        ),
    )
    parser.add_argument(
        "--press-initial-delay-min-ms",
        type=int,
        default=300,
        help="Minimum non-blocking delay after PRESS freeze (default: 300)",
    )
    parser.add_argument(
        "--press-initial-delay-max-ms",
        type=int,
        default=500,
        help="Maximum non-blocking delay after PRESS freeze (default: 500)",
    )
    parser.add_argument(
        "--press-inter-key-gap-min-ms",
        type=int,
        default=90,
        help="Minimum key-up to next key-down gap (default: 90)",
    )
    parser.add_argument(
        "--press-inter-key-gap-max-ms",
        type=int,
        default=170,
        help="Maximum key-up to next key-down gap (default: 170)",
    )
    parser.add_argument(
        "--press-key-hold-ms",
        type=int,
        default=40,
        help="Key-down hold duration for each PRESS key (default: 40)",
    )
    parser.add_argument(
        "--press-anomaly-evidence",
        action="store_true",
        help=(
            "Save bounded PRESS ROI evidence only after an incomplete/timeout "
            "anomaly (default: disabled)"
        ),
    )
    parser.add_argument(
        "--press-anomaly-buffer-frames",
        type=int,
        default=12,
        help="Maximum in-memory PRESS ROI frames (default: 12)",
    )
    parser.add_argument(
        "--press-anomaly-max-episodes",
        type=int,
        default=20,
        help="Maximum anomaly episode directories per session (default: 20)",
    )
    parser.add_argument(
        "--panic-key", choices=("F12",), default="F12",
        help="Polling-only permanent session stop key for the action sink (default: F12)",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "fishing_v2.yaml")
    args = parser.parse_args(argv)
    if args.window_title_prefix and not args.process_name:
        parser.error("--window-title-prefix requires --process-name")
    if (
        args.press_initial_delay_min_ms < 0
        or args.press_initial_delay_min_ms
        > args.press_initial_delay_max_ms
    ):
        parser.error("invalid PRESS initial delay range")
    if (
        args.press_inter_key_gap_min_ms < 0
        or args.press_inter_key_gap_min_ms
        > args.press_inter_key_gap_max_ms
    ):
        parser.error("invalid PRESS inter-key gap range")
    if args.duration_seconds < 0:
        parser.error("--duration-seconds must be non-negative")
    if args.hook_action_stall_timeout_seconds <= 0:
        parser.error("--hook-action-stall-timeout-seconds must be positive")
    if args.idle_recovery_window_size < 1:
        parser.error("--idle-recovery-window-size must be positive")
    if not 1 <= args.idle_recovery_required_count <= args.idle_recovery_window_size:
        parser.error("--idle-recovery-required-count must fit the window")
    if args.idle_recovery_min_window_seconds < 0:
        parser.error("--idle-recovery-min-window-seconds must be non-negative")
    if args.idle_recovery_freshness_ms <= 0:
        parser.error("--idle-recovery-freshness-ms must be positive")
    if args.idle_recovery_cast_cooldown_seconds < 0:
        parser.error("--idle-recovery-cast-cooldown-seconds must be non-negative")
    if args.idle_cast_retry_min_interval_seconds <= 0:
        parser.error("--idle-cast-retry-min-interval-seconds must be positive")
    if args.max_completed_cycles is not None and args.max_completed_cycles < 0:
        parser.error("--max-completed-cycles must be non-negative")
    if args.log_repeat_window_seconds < 0 or args.log_max_file_mb <= 0:
        parser.error("invalid production log timing or size limit")
    if args.log_backup_count < 0 or args.log_retention_days < 0:
        parser.error("invalid production log retention count")
    if args.log_max_total_mb < 0:
        parser.error("--log-max-total-mb must be non-negative")
    if args.press_key_hold_ms <= 0:
        parser.error("--press-key-hold-ms must be positive")
    if args.press_anomaly_buffer_frames < 6:
        parser.error("--press-anomaly-buffer-frames must be at least 6")
    if args.press_anomaly_max_episodes < 1:
        parser.error("--press-anomaly-max-episodes must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.process_name
        and normalize_process_name(args.process_name)
        != normalize_process_name("BlackDesert64.exe")
    ):
        print(
            "REFUSED: --process-name must identify BlackDesert64.exe",
            file=sys.stderr,
        )
        return 2
    try:
        validate_emit_actions(
            args.emit_actions,
            args.action_sink,
            args.action_allowlist,
            enable_live_press_sequence=(
                args.enable_live_press_sequence
            ),
        )
    except LivePreflightError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    try:
        target = resolve_window_target(
            exact_title=args.window_title,
            process_name=args.process_name,
            title_prefix=args.window_title_prefix,
            expected_process_name="BlackDesert64",
        )
    except (WindowResolutionError, RuntimeError, ValueError) as exc:
        print(f"WINDOW RESOLUTION FAILED: {exc}", file=sys.stderr)
        if isinstance(exc, WindowResolutionError):
            print(
                "Candidates:\n"
                + format_window_candidates(exc.candidates),
                file=sys.stderr,
            )
        return 2
    try:
        bundle = load_prompt_bundle(args.prompt_bundle)
    except Exception as exc:
        print(f"PREFLIGHT FAILED: final Prompt bundle could not be loaded: {exc}", file=sys.stderr)
        return 2
    if args.runtime_profile == "production":
        logger = ProductionSessionLogger(
            args.output_dir,
            bundle_version=bundle.bundle_version,
            repeat_window_seconds=args.log_repeat_window_seconds,
            max_file_mb=args.log_max_file_mb,
            backup_count=args.log_backup_count,
            retention_days=args.log_retention_days,
            max_total_mb=args.log_max_total_mb,
        )
    else:
        logger = LiveSessionLogger(
            args.output_dir,
            bundle_version=bundle.bundle_version,
        )
    capture = create_live_capture_session(
        backend=args.capture_backend,
        window_title=target.window_title,
        resolved_window=target.window_info,
        expected_title_prefix=target.title_prefix,
        window_resolution_mode=target.resolution_mode,
        allow_mss_fallback=args.allow_mss_fallback,
    )
    if args.capture_backend == MSS_REGION_BACKEND and args.show_overlay:
        print(
            "WARNING: mss-region captures desktop pixels; the diagnostic overlay or other "
            "covering windows may appear in captured frames. Prefer --no-overlay.",
            file=sys.stderr,
        )
    if (
        args.capture_backend == WINDOWS_GRAPHICS_CAPTURE_BACKEND
        and args.allow_mss_fallback
    ):
        print(
            "WARNING: explicit MSS fallback is enabled; any fallback is recorded in the session.",
            file=sys.stderr,
        )
    runtime = LiveDetectOnlyRuntime(
        config_path=args.config,
        prompt_bundle=bundle,
        capture=capture,
        logger=logger,
        live_config=LiveDetectOnlyConfig(
            duration_seconds=args.duration_seconds,
            max_fps=args.max_fps,
            show_overlay=(
                args.show_overlay
                if args.runtime_profile == "diagnostic" else False
            ),
            save_transition_frames=(
                args.save_transition_frames
                if args.runtime_profile == "diagnostic" else False
            ),
            evidence_mode=args.evidence_mode,
            evidence_video_fps=args.evidence_video_fps,
            hook_critical_target_fps=args.hook_critical_fps,
            hook_action_stall_timeout_seconds=(
                args.hook_action_stall_timeout_seconds
            ),
            idle_recovery_window_size=args.idle_recovery_window_size,
            idle_recovery_required_count=(
                args.idle_recovery_required_count
            ),
            idle_recovery_min_window_seconds=(
                args.idle_recovery_min_window_seconds
            ),
            idle_recovery_freshness_ms=(
                args.idle_recovery_freshness_ms
            ),
            idle_recovery_cast_cooldown_seconds=(
                args.idle_recovery_cast_cooldown_seconds
            ),
            idle_cast_retry_min_interval_seconds=(
                args.idle_cast_retry_min_interval_seconds
            ),
            max_completed_cycles=args.max_completed_cycles,
            press_initial_delay_min_ms=(
                args.press_initial_delay_min_ms
            ),
            press_initial_delay_max_ms=(
                args.press_initial_delay_max_ms
            ),
            press_inter_key_gap_min_ms=(
                args.press_inter_key_gap_min_ms
            ),
            press_inter_key_gap_max_ms=(
                args.press_inter_key_gap_max_ms
            ),
            press_key_hold_ms=args.press_key_hold_ms,
            press_anomaly_evidence=args.press_anomaly_evidence,
            press_anomaly_buffer_frames=(
                args.press_anomaly_buffer_frames
            ),
            press_anomaly_max_episodes=(
                args.press_anomaly_max_episodes
            ),
            runtime_profile=args.runtime_profile,
        ),
        emit_actions=args.emit_actions,
        action_sink_name=args.action_sink,
        action_allowlist=args.action_allowlist,
        enable_live_press_sequence=args.enable_live_press_sequence,
        panic_key=args.panic_key,
    )
    summary = runtime.run()
    print(f"session: {logger.path}")
    print(f"result: {summary['result']}")
    print(f"capture_backend: {summary.get('capture_backend')}")
    print(f"capture_fallback_used: {summary.get('capture_fallback_used', False)}")
    print(f"evidence_mode: {summary.get('evidence_mode')}")
    print(f"runtime_profile: {summary.get('runtime_profile')}")
    print(f"video_path: {summary.get('video_path')}")
    print(f"completed_cycles: {summary.get('completed_cycles')}")
    print(f"actions_applied: {summary['actions_applied']}")
    if summary["result"] == "preflight_failed":
        print("preflight failed:", file=sys.stderr)
        print(summary.get("preflight_failure_reason"), file=sys.stderr)
        message = summary.get("preflight_failure_message")
        if message:
            lines = str(message).splitlines()
            if lines and lines[0] == summary.get("preflight_failure_reason"):
                lines = lines[1:]
            print("\n".join(lines), file=sys.stderr)
    return 0 if summary["result"] in {
        "completed", "completed_target_cycles", "interrupted_by_user",
        "panic_shutdown",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
