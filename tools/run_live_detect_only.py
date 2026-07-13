"""Observe the real game window without ever emitting keyboard or mouse input."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.live.live_detect_only import (  # noqa: E402
    LiveDetectOnlyConfig,
    LiveDetectOnlyRuntime,
    LivePreflightError,
    validate_emit_actions,
)
from src.fishing_v2.live.session_logger import LiveSessionLogger  # noqa: E402
from src.fishing_v2.perception.prompt_bundle import load_prompt_bundle  # noqa: E402
from src.screen_capture import MSSCaptureSession  # noqa: E402


def _strict_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "false":
        return False
    if normalized == "true":
        return True
    raise argparse.ArgumentTypeError("expected true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-title", required=True, help="Exact visible borderless game-window title")
    parser.add_argument("--duration-seconds", type=float, default=180.0)
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
    parser.add_argument("--emit-actions", type=_strict_bool, default=False)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "fishing_v2.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validate_emit_actions(args.emit_actions)
    except LivePreflightError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    try:
        bundle = load_prompt_bundle(args.prompt_bundle)
    except Exception as exc:
        print(f"PREFLIGHT FAILED: final Prompt bundle could not be loaded: {exc}", file=sys.stderr)
        return 2
    logger = LiveSessionLogger(args.output_dir, bundle_version=bundle.bundle_version)
    capture = MSSCaptureSession(window_title=args.window_title)
    runtime = LiveDetectOnlyRuntime(
        config_path=args.config,
        prompt_bundle=bundle,
        capture=capture,
        logger=logger,
        live_config=LiveDetectOnlyConfig(
            duration_seconds=args.duration_seconds,
            max_fps=args.max_fps,
            show_overlay=args.show_overlay,
            save_transition_frames=args.save_transition_frames,
        ),
        emit_actions=False,
    )
    summary = runtime.run()
    print(f"session: {logger.path}")
    print(f"result: {summary['result']}")
    print(f"actions_applied: {summary['actions_applied']}")
    return 0 if summary["result"] in {"completed", "interrupted_by_user"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
