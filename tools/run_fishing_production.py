"""Run the existing fishing Live runtime with bounded Production defaults."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_live_detect_only import main as run_live_main  # noqa: E402


PRODUCTION_DEFAULTS = (
    "--runtime-profile", "production",
    "--capture-backend", "mss-region",
    "--hook-critical-fps", "40",
    "--press-initial-delay-min-ms", "300",
    "--press-initial-delay-max-ms", "500",
    "--press-inter-key-gap-min-ms", "90",
    "--press-inter-key-gap-max-ms", "170",
    "--press-key-hold-ms", "40",
    "--no-overlay",
    "--duration-seconds", "0",
    "--max-completed-cycles", "0",
    "--enable-live-press-sequence",
    "--emit-actions", "true",
    "--action-sink", "sendinput",
    "--action-allowlist",
    "CAST,START_HOOK,HOOK_ACTION,PRESS_SEQUENCE,COLLECT",
    "--panic-key", "F12",
)


def main(argv: list[str] | None = None) -> int:
    return run_live_main([*PRODUCTION_DEFAULTS, *(argv or sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
