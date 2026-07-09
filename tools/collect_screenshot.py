"""Capture-only replay-frame collector; it never sends keyboard input."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import DEFAULT_CAPTURE_CONFIG_PATH, load_capture_config  # noqa: E402
from src.replay_session import DEFAULT_SESSION_ROOT, ReplaySession, apply_retention  # noqa: E402
from src.screen_capture import capture_screen, get_monitors  # noqa: E402


def _format_size(byte_count: int) -> str:
    return f"{byte_count / (1024 * 1024):.1f} MiB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture monitor frames into a replay session without input automation.")
    parser.add_argument("--duration", type=float, help="Capture duration in seconds")
    parser.add_argument("--interval", type=float, help="Seconds between frames")
    parser.add_argument("--monitor", type=int, default=1, help="One-based physical mss monitor index")
    parser.add_argument("--notes", default="", help="Optional manifest note")
    parser.add_argument("--list-monitors", action="store_true", help="List mss monitor indexes and exit")
    parser.add_argument("--capture-config", type=Path, default=DEFAULT_CAPTURE_CONFIG_PATH)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_monitors:
        for monitor in get_monitors():
            print(monitor)
        return 0

    config = load_capture_config(args.capture_config)
    duration = args.duration if args.duration is not None else config.capture.duration_sec
    interval = args.interval if args.interval is not None else config.capture.interval_sec
    if duration <= 0 or interval <= 0:
        raise ValueError("--duration and --interval must be greater than 0")
    monitors = get_monitors()
    monitor = next((item for item in monitors if item["index"] == args.monitor), None)
    if monitor is None:
        raise ValueError(f"Monitor {args.monitor} is unavailable; use --list-monitors")
    frame_limit = min(math.ceil(duration / interval), config.capture.max_frames_per_session)
    estimated_bytes = int(monitor["width"] * monitor["height"] * 3 * 0.12 * frame_limit)
    print(f"Capture-only session: up to {frame_limit} frame(s), estimated {_format_size(estimated_bytes)}")
    print(f"Monitor {args.monitor}: {monitor['width']}x{monitor['height']}; Ctrl+C safely stops collection.")

    session = ReplaySession.create(
        args.session_root,
        interval_sec=interval,
        duration_sec=duration,
        image_format=config.capture.image_format,
        screen_size=(monitor["width"], monitor["height"]),
        monitor_index=args.monitor,
        notes=args.notes,
    )
    apply_retention(args.session_root, config.retention, keep_session=session.path)
    started = time.monotonic()
    try:
        while session.manifest["frame_count"] < frame_limit and time.monotonic() - started < duration:
            frame_started = time.monotonic()
            frame = capture_screen(args.monitor)
            session.save_frame(frame, jpg_quality=config.capture.jpg_quality)
            size_bytes = session.size_bytes()
            elapsed = time.monotonic() - started
            print(f"elapsed={elapsed:6.1f}s frames={session.manifest['frame_count']}/{frame_limit} size={_format_size(size_bytes)}")
            if size_bytes >= config.capture.max_session_size_mb * 1024 * 1024:
                print("Stopped at max_session_size_mb limit.")
                break
            time.sleep(max(0.0, interval - (time.monotonic() - frame_started)))
    except KeyboardInterrupt:
        print("Capture interrupted safely; keeping frames already written.")
    finally:
        session.write_manifest()
        apply_retention(args.session_root, config.retention, keep_session=session.path)
    print(f"Session saved: {session.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
