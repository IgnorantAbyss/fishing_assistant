"""Interactively mark state transitions on existing replay frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.replay_ground_truth import (  # noqa: E402
    GROUND_TRUTH_STATES,
    load_ground_truth,
    validate_segments,
    write_ground_truth,
)
from src.replay_session import DEFAULT_SESSION_ROOT, ReplaySession, latest_session  # noqa: E402
from src.state_detector import StateDetector  # noqa: E402


STATE_KEYS = {
    ord("i"): "IDLE",
    ord("w"): "WAITING",
    ord("r"): "READY",
    ord("h"): "HOOK",
    ord("p"): "PRESS",
    ord("g"): "GET",
    ord("x"): "IGNORE",
}


def labels_to_markers(labels: dict[int, str]) -> list[tuple[int, str]]:
    markers: list[tuple[int, str]] = []
    previous: str | None = None
    for frame_index in sorted(labels):
        state = labels[frame_index]
        if state != previous:
            markers.append((frame_index, state))
            previous = state
    return markers


def markers_to_segments(
    markers: list[tuple[int, str]], frame_count: int
) -> list[dict[str, int | str]]:
    latest_by_frame: dict[int, str] = {}
    for frame_index, state in markers:
        state = state.upper()
        if state not in GROUND_TRUTH_STATES:
            raise ValueError(f"Unknown marker state: {state}")
        latest_by_frame[frame_index] = state
    ordered = sorted(latest_by_frame.items())
    if not ordered or ordered[0][0] != 1:
        raise ValueError("The first marker must start at frame 1")
    segments = [
        {
            "start": frame_index,
            "end": ordered[index + 1][0] - 1 if index + 1 < len(ordered) else frame_count,
            "state": state,
        }
        for index, (frame_index, state) in enumerate(ordered)
    ]
    return validate_segments(segments, frame_count)


def _load_markers(session: ReplaySession) -> list[tuple[int, str]]:
    path = session.path / "ground_truth.yaml"
    if not path.is_file():
        return []
    return labels_to_markers(load_ground_truth(path, len(session.frame_paths())))


def _fit_for_display(frame):
    max_width, max_height = 1600, 900
    scale = min(1.0, max_width / frame.shape[1], max_height / frame.shape[0])
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _annotated_frame(frame, frame_index: int, frame_count: int, prediction, markers):
    display = _fit_for_display(frame)
    recent = sorted(markers, key=lambda item: item[0])[-5:]
    lines = [
        f"frame {frame_index}/{frame_count}",
        f"detector reference only: {prediction.state} ({prediction.confidence:.2f})",
        "I/W/R/H/P/G/X mark | S save | Backspace undo | Q quit",
        "markers: " + ", ".join(f"{index}:{state}" for index, state in recent),
    ]
    cv2.rectangle(display, (0, 0), (display.shape[1], 112), (0, 0, 0), -1)
    for line_index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (14, 25 + line_index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return display


def annotate_session(session_path: str | Path) -> None:
    session = ReplaySession.load(session_path)
    frames = session.frame_paths()
    if not frames:
        raise ValueError(f"Session has no frames: {session.path}")
    detector = StateDetector(PROJECT_ROOT / "assets" / "reference")
    markers = _load_markers(session)
    current = 0
    window = "Replay ground truth annotation"
    special_keys = {
        2424832: -1,
        2555904: 1,
        2162688: -10,
        2228224: 10,
        2359296: "home",
        2293760: "end",
    }
    try:
        while True:
            frame = cv2.imread(str(frames[current]), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Could not read frame: {frames[current]}")
            prediction = detector.detect_state(frame)
            cv2.imshow(
                window,
                _annotated_frame(frame, current + 1, len(frames), prediction, markers),
            )
            key = cv2.waitKeyEx(0)
            if key in special_keys:
                action = special_keys[key]
                if action == "home":
                    current = 0
                elif action == "end":
                    current = len(frames) - 1
                else:
                    current = max(0, min(len(frames) - 1, current + int(action)))
                continue
            character = key & 0xFF
            lower = ord(chr(character).lower()) if 0 <= character <= 255 else character
            if lower == ord("q"):
                break
            if lower == ord("s"):
                segments = markers_to_segments(markers, len(frames))
                destination = write_ground_truth(
                    session.path / "ground_truth.yaml", segments, len(frames)
                )
                print(f"Saved {destination}")
                continue
            if character == 8:
                if markers:
                    removed = markers.pop()
                    print(f"Removed marker {removed[0]}:{removed[1]}")
                continue
            state = STATE_KEYS.get(lower)
            if state is not None:
                markers.append((current + 1, state))
                print(f"Marker {current + 1}:{state}")
    finally:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline transition-marker editor for replay ground truth.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--session", type=Path)
    selection.add_argument("--latest", action="store_true")
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_path = args.session if args.session else latest_session(args.session_root)
    annotate_session(session_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
