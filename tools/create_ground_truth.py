"""Create validated frame-range labels for an existing replay session."""

from __future__ import annotations

import argparse
from collections import Counter
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
    selection.add_argument("--session", help="Session id or session directory path")
    selection.add_argument("--latest", action="store_true")
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--range", action="append", type=_parse_range, required=True, dest="ranges")
    parser.add_argument("--force", action="store_true", help="Replace an existing ground_truth.yaml without prompting")
    return parser.parse_args()


def resolve_session_path(session_value: str | Path, session_root: str | Path) -> Path:
    candidate = Path(session_value)
    return candidate if candidate.is_dir() else Path(session_root) / candidate


def prepare_ground_truth(
    session_path: str | Path, ranges: list[dict[str, int | str]]
) -> tuple[ReplaySession, list[dict[str, int | str]], Path]:
    session = ReplaySession.load(session_path)
    frame_count = int(session.manifest["frame_count"])
    segments = validate_segments(ranges, frame_count)
    return session, segments, session.path / "ground_truth.yaml"


def preview_ground_truth(
    session: ReplaySession, segments: list[dict[str, int | str]], destination: Path
) -> str:
    frame_count = int(session.manifest["frame_count"])
    counts: Counter[str] = Counter()
    for segment in segments:
        counts[str(segment["state"])] += int(segment["end"]) - int(segment["start"]) + 1
    covered = sum(counts.values())
    lines = [
        f"session_id: {session.path.name}",
        f"frame_count: {frame_count}",
        "segments:",
        *[
            f"  - {segment['start']}-{segment['end']}:{segment['state']}"
            for segment in segments
        ],
        f"state_frame_counts: {dict(sorted(counts.items()))}",
        f"coverage: {covered}/{frame_count} ({covered / frame_count:.2%})",
        f"output_path: {destination.resolve()}",
    ]
    return "\n".join(lines)


def create_ground_truth_file(
    session_path: str | Path,
    ranges: list[dict[str, int | str]],
    *,
    force: bool = False,
) -> tuple[Path, list[dict[str, int | str]]]:
    session, segments, destination = prepare_ground_truth(session_path, ranges)
    if destination.exists() and not force:
        raise FileExistsError(
            f"Ground truth already exists and will not be replaced: {destination.resolve()}; use --force"
        )
    write_ground_truth(destination, segments, int(session.manifest["frame_count"]))
    return destination, segments


def main() -> int:
    args = parse_args()
    session_path = (
        resolve_session_path(args.session, args.session_root)
        if args.session
        else latest_session(args.session_root)
    )
    session, segments, destination = prepare_ground_truth(session_path, args.ranges)
    print(preview_ground_truth(session, segments, destination))
    force = args.force
    if destination.exists() and not force:
        print(f"Existing ground truth will be replaced: {destination.resolve()}")
        if not sys.stdin.isatty():
            raise FileExistsError(
                f"Non-interactive overwrite refused: {destination.resolve()}; use --force"
            )
        response = input("Type REPLACE to overwrite this file: ").strip()
        if response != "REPLACE":
            raise FileExistsError("Ground-truth overwrite cancelled")
        force = True
    output, written_segments = create_ground_truth_file(
        session.path, segments, force=force
    )
    counts = Counter(
        str(segment["state"])
        for segment in written_segments
        for _ in range(int(segment["start"]), int(segment["end"]) + 1)
    )
    print(f"ground_truth: {output.resolve()}")
    print(f"segment_count: {len(written_segments)}")
    print(f"frame_coverage: {sum(counts.values())}/{session.manifest['frame_count']}")
    print(f"state_frame_counts: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
