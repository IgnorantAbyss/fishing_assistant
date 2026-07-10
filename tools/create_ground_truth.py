"""Create validated frame-range labels for an existing replay session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.replay_session import DEFAULT_SESSION_ROOT, ReplaySession, latest_session  # noqa: E402
from src.replay_ground_truth import GROUND_TRUTH_STATES, validate_segments, write_ground_truth  # noqa: E402


def _parse_range(value: str) -> dict[str, int | str]:
    try:
        range_part, state = value.split(":", maxsplit=1)
        start_text, end_text = range_part.split("-", maxsplit=1)
        start, end = int(start_text), int(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid range '{value}'; use START-END:STATE") from exc
    state = state.upper()
    if start < 1 or end < start:
        raise argparse.ArgumentTypeError(f"Invalid frame range '{value}'")
    if state not in GROUND_TRUTH_STATES:
        raise argparse.ArgumentTypeError(f"Unsupported state '{state}'")
    return {"start": start, "end": end, "state": state}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write frame-range ground truth into an existing replay session.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--session", type=Path)
    selection.add_argument("--latest", action="store_true")
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--range", action="append", type=_parse_range, required=True, dest="ranges")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_path = args.session if args.session else latest_session(args.session_root)
    session = ReplaySession.load(session_path)
    segments = validate_segments(args.ranges, int(session.manifest["frame_count"]))
    destination = session.path / "ground_truth.yaml"
    write_ground_truth(destination, segments, int(session.manifest["frame_count"]))
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
