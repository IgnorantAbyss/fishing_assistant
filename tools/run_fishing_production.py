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
    "--hook-action-stall-timeout-seconds", "3.0",
    "--idle-recovery-window-size", "5",
    "--idle-recovery-required-count", "4",
    "--idle-recovery-min-window-seconds", "0.5",
    "--idle-recovery-freshness-ms", "250",
    "--idle-recovery-cast-cooldown-seconds", "0.5",
    "--idle-cast-retry-min-interval-seconds", "3.0",
    "--idle-cast-liveness-timeout-seconds", "3.0",
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
    # PRESS defaults are resolved by the shared runtime profile, not this wrapper.
    return run_live_main([*PRODUCTION_DEFAULTS, *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
