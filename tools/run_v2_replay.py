"""Run Hybrid Runtime v2 over replay frames with no real action emission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.domain.runtime_state import RuntimeState  # noqa: E402
from src.fishing_v2.replay.v2_replay_runner import V2ReplayRunner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Hybrid Runtime v2 architecture on saved replay frames.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--session", help="Session id or directory")
    selection.add_argument("--all", action="store_true")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--use-scripted-prompt-observations", action="store_true")
    prompt.add_argument("--no-prompt-observer", action="store_true")
    parser.add_argument("--start-state", choices=[state.value for state in RuntimeState if state not in {RuntimeState.SYNCING, RuntimeState.SYNC_REQUIRED}])
    parser.add_argument("--session-root", type=Path, default=PROJECT_ROOT / "assets" / "replay" / "sessions")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "fishing_v2.yaml")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "fishing_v2" / "replay")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = "scripted_prompt" if args.use_scripted_prompt_observations else "no_prompt"
    if args.all:
        sessions = sorted(path for path in args.session_root.glob("session_*") if (path / "ground_truth.yaml").is_file())
    else:
        supplied = Path(args.session)
        sessions = [supplied if supplied.is_dir() else args.session_root / args.session]
    runner = V2ReplayRunner(args.config)
    start_state = RuntimeState(args.start_state) if args.start_state else None
    for session in sessions:
        run = runner.run(session, mode=mode, report_dir=args.report_dir, start_state=start_state)
        print(f"{run.session_id}: frames={len(run.rows)} report={run.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
