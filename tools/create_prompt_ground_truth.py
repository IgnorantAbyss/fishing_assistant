"""Create human-authored prompt observation ranges without global-state inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.data.prompt_annotation import PromptAnnotationKind, write_prompt_ground_truth  # noqa: E402
from src.replay_session import DEFAULT_SESSION_ROOT, ReplaySession, latest_session  # noqa: E402


def parse_range(value: str) -> dict[str, int | str]:
    try:
        span, raw_observation = value.rsplit(":", 1)
        raw_start, raw_end = span.split("-", 1)
        observation = PromptAnnotationKind(raw_observation.upper()).value
        return {"start": int(raw_start), "end": int(raw_end), "observation": observation}
    except (ValueError, KeyError) as exc:
        allowed = ", ".join(item.value for item in PromptAnnotationKind)
        raise argparse.ArgumentTypeError(f"Expected START-END:OBSERVATION; allowed: {allowed}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create prompt-specific ground truth from human-reviewed ROI content.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--latest", action="store_true")
    selection.add_argument("--session", help="Session id or directory")
    parser.add_argument("--range", dest="ranges", action="append", type=parse_range, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.latest:
        session_path = latest_session(args.session_root)
    else:
        supplied = Path(args.session)
        session_path = supplied if supplied.is_dir() else args.session_root / args.session
    session = ReplaySession.load(session_path)
    destination = write_prompt_ground_truth(
        session.path / "prompt_ground_truth.yaml",
        args.ranges,
        int(session.manifest["frame_count"]),
        force=args.force,
    )
    print(json.dumps({"session": session.path.name, "path": str(destination), "source": "human_ranges_only"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
