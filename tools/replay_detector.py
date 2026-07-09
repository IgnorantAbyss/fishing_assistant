"""Run state and component detectors over a saved replay session offline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors.hook_detector import detect_hook_bar  # noqa: E402
from src.detectors.press_detector import detect_press_sequence  # noqa: E402
from src.replay_session import DEFAULT_SESSION_ROOT, ReplaySession, latest_session  # noqa: E402
from src.state_detector import StateDetector  # noqa: E402


@dataclass(frozen=True)
class ReplayRun:
    session_path: Path
    results_path: Path
    report_path: Path
    frame_count: int
    state_counts: dict[str, int]


def _state_transitions(rows: list[dict[str, Any]]) -> list[str]:
    transitions: list[str] = []
    previous: str | None = None
    for row in rows:
        state = str(row["state"])
        if state != previous:
            if previous is not None:
                transitions.append(f"{previous} -> {state} @ {float(row['timestamp_sec']):.2f}s")
            previous = state
    return transitions


def _write_report(
    session: ReplaySession,
    rows: list[dict[str, Any]],
    state_counts: Counter[str],
    low_confidence_count: int,
    hook_rows: list[dict[str, Any]],
    press_rows: list[dict[str, Any]],
) -> Path:
    frame_count = len(rows)
    unknown_count = state_counts.get("UNKNOWN", 0)
    transitions = _state_transitions(rows)
    sequences = Counter(row["press_sequence_text"] for row in press_rows if row["press_sequence_text"])
    fills = [float(row["hook_fill_ratio"]) for row in hook_rows if row["hook_fill_ratio"]]
    issues: list[str] = []
    if frame_count == 0:
        issues.append("No frames found in the session.")
    if frame_count and unknown_count / frame_count > 0.20:
        issues.append("UNKNOWN ratio exceeds 20%; recalibrate ROI or collect clearer frames.")
    if any("?" in str(row["press_sequence_text"]) for row in press_rows):
        issues.append("At least one PRESS sequence has an unresolved key glyph.")
    lines = [
        "# Replay Detector Report",
        "",
        f"- Session: `{session.manifest.get('session_id', session.path.name)}`",
        f"- Total frames: {frame_count}",
        f"- UNKNOWN ratio: {unknown_count / frame_count:.2%}" if frame_count else "- UNKNOWN ratio: n/a",
        f"- Low-confidence frames: {low_confidence_count}",
        "",
        "## State counts",
        "",
    ]
    for state in ("IDLE", "WAITING", "READY", "HOOK", "PRESS", "GET", "UNKNOWN"):
        lines.append(f"- {state}: {state_counts.get(state, 0)}")
    lines.extend(["", "## State transitions", ""])
    lines.extend([f"- {transition}" for transition in transitions] or ["- None"])
    lines.extend(["", "## HOOK frames", "", f"- Count: {len(hook_rows)}"])
    lines.append(f"- Mean fill ratio: {sum(fills) / len(fills):.4f}" if fills else "- Mean fill ratio: n/a")
    lines.extend(["", "## PRESS frames", "", f"- Count: {len(press_rows)}", "- Sequences:"])
    lines.extend([f"  - {sequence}: {count}" for sequence, count in sorted(sequences.items())] or ["  - None"])
    lines.extend(["", "## Possible problems", ""])
    lines.extend([f"- {issue}" for issue in issues] or ["- None"])
    destination = session.path / "replay_report.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def run_replay_detector(
    session_path: str | Path,
    *,
    reference_dir: str | Path = PROJECT_ROOT / "assets" / "reference",
) -> ReplayRun:
    """Evaluate each saved frame offline and write CSV plus Markdown results."""
    session = ReplaySession.load(session_path)
    detector = StateDetector(reference_dir)
    rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    low_confidence_count = 0
    hook_rows: list[dict[str, Any]] = []
    press_rows: list[dict[str, Any]] = []
    interval = float(session.manifest.get("interval_sec", 0.0))
    for frame_index, frame_path in enumerate(session.frame_paths(), start=1):
        result = detector.detect_state(frame_path)
        state = result.state
        confidence = result.confidence
        state_counts[state] += 1
        if confidence < detector.thresholds.min_confidence_for(state):
            low_confidence_count += 1
        hook_fill_ratio: float | None = None
        hook_divider_ratio: float | None = None
        press_sequence_text = ""
        if state == "HOOK":
            hook = detect_hook_bar(frame_path, detector.roi_config, detector.thresholds)
            hook_fill_ratio = hook["fill_ratio"]
            hook_divider_ratio = hook["divider_ratio"]
        elif state == "PRESS":
            press = detect_press_sequence(frame_path, detector.roi_config, detector.thresholds)
            press_sequence_text = press["sequence_text"]
        row = {
            "frame_index": frame_index,
            "filename": frame_path.name,
            "timestamp_sec": round((frame_index - 1) * interval, 6),
            "state": state,
            "confidence": confidence,
            "raw_scores": json.dumps(result.debug.get("raw_scores", {}), ensure_ascii=False, sort_keys=True),
            "hook_fill_ratio": hook_fill_ratio if hook_fill_ratio is not None else "",
            "hook_divider_ratio": hook_divider_ratio if hook_divider_ratio is not None else "",
            "press_sequence_text": press_sequence_text,
            "matched_features": "|".join(result.matched_features),
        }
        rows.append(row)
        if state == "HOOK":
            hook_rows.append(row)
        if state == "PRESS":
            press_rows.append(row)
    results_path = session.path / "replay_results.csv"
    fieldnames = [
        "frame_index", "filename", "timestamp_sec", "state", "confidence", "raw_scores",
        "hook_fill_ratio", "hook_divider_ratio", "press_sequence_text", "matched_features",
    ]
    with results_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report_path = _write_report(session, rows, state_counts, low_confidence_count, hook_rows, press_rows)
    return ReplayRun(session.path, results_path, report_path, len(rows), dict(state_counts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay saved frames through offline state/component detectors.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--session", type=Path, help="Replay session directory")
    selection.add_argument("--latest", action="store_true", help="Use most recently modified session")
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--reference-dir", type=Path, default=PROJECT_ROOT / "assets" / "reference")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_path = args.session if args.session else latest_session(args.session_root)
    run = run_replay_detector(session_path, reference_dir=args.reference_dir)
    print(f"frames: {run.frame_count}")
    print(f"state_counts: {run.state_counts}")
    print(f"replay_results: {run.results_path}")
    print(f"replay_report: {run.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
