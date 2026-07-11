#!/usr/bin/env python3
"""Cross-session PRESS panel/sequence diagnostics and manual review bundle."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import html
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_loader import load_roi_config, normalized_to_pixel_roi  # noqa: E402
from src.fishing_v2.legacy_adapters.press_detector_adapter import LegacyPressDetectorAdapter  # noqa: E402
from src.fishing_v2.legacy_adapters.replay_source_adapter import LegacyReplaySourceAdapter  # noqa: E402
from src.fishing_v2.runtime.press_sequence_aggregator import (  # noqa: E402
    PressSequenceAggregationConfig,
    PressSequenceTemporalAggregator,
)


DEFAULT_CONFIG = ROOT / "config" / "fishing_v2.yaml"
DEFAULT_SESSION_ROOT = ROOT / "assets" / "replay" / "sessions"
DEFAULT_REVIEW_ROOT = ROOT / "reports" / "fishing_v2" / "press_sequence_review"
SUMMARY_MD = ROOT / "reports" / "fishing_v2" / "press_detector_diagnostics_summary.md"
SUMMARY_JSON = ROOT / "reports" / "fishing_v2" / "press_detector_diagnostics_summary.json"
REVIEW_FIELDS = (
    "session_id", "press_start", "press_end", "review_frame",
    "predicted_sequence", "manually_confirmed_sequence", "review_status", "notes",
)


def _aggregation_config(path: Path) -> PressSequenceAggregationConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    press = data["press_detector"]
    return PressSequenceAggregationConfig(
        panel_confirmation_frames=int(press["panel_confirmation_frames"]),
        panel_geometry_tolerance=float(press["panel_geometry_tolerance"]),
        sequence_window_frames=int(press["sequence_window_frames"]),
        sequence_consensus_frames=int(press["sequence_consensus_frames"]),
        per_key_min_aggregated_confidence=float(press["per_key_min_aggregated_confidence"]),
        sequence_min_aggregated_confidence=float(press["sequence_min_aggregated_confidence"]),
    )


def _ranges(labels: dict[int, str], state: str) -> list[tuple[int, int]]:
    indices = [index for index, label in labels.items() if label == state]
    if not indices:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            ranges.append((start, previous))
            start = index
        previous = index
    ranges.append((start, previous))
    return ranges


def _percentile(values: list[float], percentile: float) -> float | None:
    return round(float(np.percentile(values, percentile)), 4) if values else None


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": round(min(values), 4) if values else None,
        "p25": _percentile(values, 25),
        "median": _percentile(values, 50),
        "p75": _percentile(values, 75),
        "max": round(max(values), 4) if values else None,
    }


def _mode_consistency(candidates: list[tuple[str, ...]]) -> tuple[str, float]:
    nonempty = [candidate for candidate in candidates if candidate]
    if not nonempty:
        return "", 0.0
    candidate, count = Counter(nonempty).most_common(1)[0]
    return "".join(candidate), round(count / len(nonempty), 4)


def _review_image(frame_path: Path, observation: Any, destination: Path) -> None:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(frame_path)
    annotated = frame.copy()
    panel = observation.evidence.get("panel_bbox")
    if isinstance(panel, (list, tuple)) and len(panel) == 4:
        cv2.rectangle(annotated, (int(panel[0]), int(panel[1])), (int(panel[2]), int(panel[3])), (0, 220, 0), 3)
    for item in observation.evidence.get("key_boxes", ()):
        box = item.get("bbox", ())
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(value) for value in box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (230, 80, 230), 2)
        cv2.putText(annotated, str(item.get("key", "?")), (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 80, 230), 2)
    roi = load_roi_config().rois["press_sequence"]
    x1, y1, x2, y2 = normalized_to_pixel_roi(roi, frame.shape[1], frame.shape[0])
    crop = annotated[y1:y2, x1:x2]
    full_width = 900
    full = cv2.resize(annotated, (full_width, round(annotated.shape[0] * full_width / annotated.shape[1])))
    crop = cv2.resize(crop, (full_width, round(crop.shape[0] * full_width / crop.shape[1])))
    canvas = np.vstack((full, crop))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise OSError(f"Could not write review image: {destination}")


def _best_review_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present = [row for row in rows if row["raw"].panel_present]
    candidates = present or rows
    return max(
        candidates,
        key=lambda row: (
            row["raw"].key_box_count,
            row["raw"].sequence_confidence,
            row["raw"].confidence,
            -row["frame"],
        ),
    )


def _episode_summary(
    session_id: str,
    start: int,
    end: int,
    rows: list[dict[str, Any]],
    interval: float,
) -> dict[str, Any]:
    first_candidate = next((row["frame"] for row in rows if row["raw"].panel_candidate), None)
    first_present = next((row["frame"] for row in rows if row["raw"].panel_present), None)
    first_confirmed = next((row["frame"] for row in rows if row["aggregation"].panel_confirmed), None)
    first_ready = next((row["frame"] for row in rows if row["aggregation"].sequence_ready), None)
    ready_row = next((row for row in rows if row["aggregation"].sequence_ready), None)
    modal_candidate, consistency = _mode_consistency([
        row["aggregation"].sequence_candidate for row in rows
    ])
    review = _best_review_row(rows)
    return {
        "session_id": session_id,
        "press_start": start,
        "press_end": end,
        "support": len(rows),
        "panel_candidate_first_frame": first_candidate,
        "panel_present_first_frame": first_present,
        "panel_confirmed_first_frame": first_confirmed,
        "sequence_ready_first_frame": first_ready,
        "panel_detection_latency_frames": None if first_confirmed is None else first_confirmed - start,
        "panel_detection_latency_seconds": None if first_confirmed is None else round((first_confirmed - start) * interval, 4),
        "panel_presence_recall": round(sum(row["raw"].panel_present for row in rows) / len(rows), 4),
        "key_box_counts": [row["raw"].key_box_count for row in rows],
        "grid_match_counts": [
            int(row["raw"].evidence.get("legacy_debug", {}).get("grid_match_count", 0))
            for row in rows
        ],
        "glyph_confidence": _distribution([
            float(box.get("confidence", 0.0))
            for row in rows
            for box in row["raw"].evidence.get("key_boxes", ())
        ]),
        "aggregated_confidence": _distribution([
            row["aggregation"].sequence_confidence for row in rows
            if row["aggregation"].sequence_candidate
        ]),
        "modal_sequence_candidate": modal_candidate,
        "sequence_candidate_consistency": consistency,
        "predicted_sequence": "" if ready_row is None else "".join(ready_row["aggregation"].sequence_candidate),
        "sequence_status": (
            "sequence_ready" if ready_row is not None else "sequence_not_recoverable_from_replay"
        ),
        "review_frame": review["frame"],
        "review_top_candidates": [
            item.get("top_candidates", []) for item in review["raw"].evidence.get("key_boxes", ())
        ],
    }


def _write_review(
    root: Path,
    episodes: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "review_items.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in REVIEW_FIELDS} for row in review_rows)
    sections = []
    for episode in episodes:
        image_name = f"{episode['session_id']}_{episode['press_start']}_{episode['press_end']}.jpg"
        top = html.escape(json.dumps(episode["review_top_candidates"], ensure_ascii=False))
        sections.append(f"""
<article>
  <h2>{html.escape(episode['session_id'])}: {episode['press_start']}-{episode['press_end']}</h2>
  <p>review frame: {episode['review_frame']} · panel first/confirmed: {episode['panel_present_first_frame']}/{episode['panel_confirmed_first_frame']} · sequence ready: {episode['sequence_ready_first_frame']}</p>
  <p>candidate: <code>{html.escape(episode['modal_sequence_candidate']) or '(none)'}</code> · predicted: <code>{html.escape(episode['predicted_sequence']) or '(not ready)'}</code> · status: <strong>{episode['sequence_status']}</strong></p>
  <img src="images/{html.escape(image_name)}" alt="PRESS review {html.escape(episode['session_id'])}">
  <details><summary>Per-box top-3 glyph candidates</summary><pre>{top}</pre></details>
</article>""")
    document = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>PRESS Sequence Review</title>
<style>body{{font-family:system-ui;margin:2rem;background:#111;color:#eee}}article{{margin:2rem 0;padding:1rem;background:#222}}img{{max-width:100%;height:auto}}code,pre{{white-space:pre-wrap;color:#9ee}}a{{color:#8cf}}</style></head>
<body><h1>PRESS Sequence Review</h1><p>Manual fields live in <a href="review_items.csv">review_items.csv</a>. Blank manually_confirmed_sequence is intentional and is not ground truth.</p>{''.join(sections)}</body></html>"""
    (root / "index.html").write_text(document, encoding="utf-8")


def run(config_path: Path, session_root: Path, review_root: Path) -> dict[str, Any]:
    aggregation_config = _aggregation_config(config_path)
    sessions = sorted(
        path for path in session_root.glob("session_*")
        if path.is_dir() and (path / "ground_truth.yaml").is_file()
    )
    if not sessions:
        raise FileNotFoundError(f"No ground-truth replay sessions under {session_root}")
    all_episodes: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    false_positive_raw: Counter[str] = Counter()
    false_positive_confirmed: Counter[str] = Counter()
    false_positive_frames: list[dict[str, Any]] = []
    session_summaries: list[dict[str, Any]] = []
    for session_path in sessions:
        source = LegacyReplaySourceAdapter(session_path)
        detector = LegacyPressDetectorAdapter()
        aggregator = PressSequenceTemporalAggregator(aggregation_config)
        rows: list[dict[str, Any]] = []
        for replay_frame in source.frames():
            frame = cv2.imread(str(replay_frame.path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(replay_frame.path)
            raw = detector.observe(frame, replay_frame.context)
            aggregation = aggregator.update(raw)
            row = {
                "frame": replay_frame.context.frame_index,
                "state": replay_frame.global_ground_truth,
                "path": replay_frame.path,
                "raw": raw,
                "aggregation": aggregation,
            }
            rows.append(row)
            if replay_frame.global_ground_truth not in {"PRESS", "IGNORE"}:
                if raw.panel_present:
                    false_positive_raw[replay_frame.global_ground_truth] += 1
                    false_positive_frames.append({
                        "session_id": session_path.name,
                        "frame": replay_frame.context.frame_index,
                        "global_state": replay_frame.global_ground_truth,
                        "panel_confidence": raw.confidence,
                        "key_box_count": raw.key_box_count,
                        "temporally_confirmed": bool(aggregation.panel_confirmed),
                    })
                if aggregation.panel_confirmed and raw.panel_present:
                    false_positive_confirmed[replay_frame.global_ground_truth] += 1
        episodes = []
        for start, end in _ranges(source.global_labels, "PRESS"):
            episode_rows = [row for row in rows if start <= row["frame"] <= end]
            episode = _episode_summary(session_path.name, start, end, episode_rows, source.interval)
            all_episodes.append(episode)
            episodes.append(episode)
            selected = next(row for row in episode_rows if row["frame"] == episode["review_frame"])
            image_name = f"{session_path.name}_{start}_{end}.jpg"
            _review_image(selected["path"], selected["raw"], review_root / "images" / image_name)
            review_rows.append({
                "session_id": session_path.name,
                "press_start": start,
                "press_end": end,
                "review_frame": episode["review_frame"],
                "predicted_sequence": episode["predicted_sequence"],
                "manually_confirmed_sequence": "",
                "review_status": "pending_manual_review",
                "notes": episode["sequence_status"],
            })
        support = sum(end - start + 1 for start, end in _ranges(source.global_labels, "PRESS"))
        present = sum(
            row["raw"].panel_present for row in rows if row["state"] == "PRESS"
        )
        session_summaries.append({
            "session_id": session_path.name,
            "frame_count": len(rows),
            "press_episode_count": len(episodes),
            "press_frame_support": support,
            "panel_present_frames": present,
            "panel_presence_recall": round(present / support, 4) if support else None,
            "sequence_ready_episodes": sum(item["sequence_status"] == "sequence_ready" for item in episodes),
        })
        print(f"{session_path.name}: frames={len(rows)} press={present}/{support} episodes={len(episodes)}", flush=True)

    total_support = sum(item["press_frame_support"] for item in session_summaries)
    total_present = sum(item["panel_present_frames"] for item in session_summaries)
    non_press_states = ("IDLE", "WAITING", "READY", "HOOK", "GET")
    summary = {
        "config": asdict(aggregation_config),
        "session_count": len(session_summaries),
        "episode_count": len(all_episodes),
        "press_frame_support": total_support,
        "panel_present_frames": total_present,
        "panel_presence_recall": round(total_present / total_support, 4) if total_support else None,
        "false_positive_raw_by_state": {state: false_positive_raw[state] for state in non_press_states},
        "false_positive_temporally_confirmed_by_state": {
            state: false_positive_confirmed[state] for state in non_press_states
        },
        "false_positive_frames": false_positive_frames,
        "runtime_fusion_false_positive_count": 0,
        "runtime_fusion_false_positive_reason": (
            "Non-PRESS diagnostic scans do not bypass detector activation; OFF evidence is never used by Fusion."
        ),
        "sequence_ready_episode_count": sum(item["sequence_status"] == "sequence_ready" for item in all_episodes),
        "sequence_accuracy": None,
        "sequence_accuracy_reason": "No manually confirmed sequence ground truth; accuracy is intentionally not computed.",
        "sessions": session_summaries,
        "episodes": all_episodes,
        "review_html": str(review_root / "index.html"),
        "review_csv": str(review_root / "review_items.csv"),
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# PRESS Detector Diagnostics Summary",
        "",
        f"- Sessions / PRESS episodes: **{summary['session_count']} / {summary['episode_count']}**",
        f"- Panel presence recall: **{total_present}/{total_support} ({summary['panel_presence_recall']:.2%})**",
        f"- Raw panel false positives by global state: `{summary['false_positive_raw_by_state']}`",
        f"- Temporally confirmed structural false positives by global state: `{summary['false_positive_temporally_confirmed_by_state']}`",
        "- Runtime Fusion false positives: **0**; diagnostic OFF evidence remains excluded from Fusion.",
        f"- Sequence-ready episodes: **{summary['sequence_ready_episode_count']}/{summary['episode_count']}**",
        "- Sequence accuracy: **N/A** — no manually confirmed sequence ground truth exists.",
        "- Existing per-frame glyph threshold was not lowered; panel presence is structural and sequence readiness uses temporal confidence plus ambiguity margin.",
        "",
        "## Sessions",
        "",
    ]
    lines.extend(
        f"- `{item['session_id']}`: PRESS frames {item['panel_present_frames']}/{item['press_frame_support']}; episodes {item['press_episode_count']}; sequence ready {item['sequence_ready_episodes']}"
        for item in session_summaries
    )
    lines.extend(["", "## Episodes", ""])
    lines.extend(
        f"- `{item['session_id']}` {item['press_start']}-{item['press_end']}: panel confirmed frame {item['panel_confirmed_first_frame']} (latency {item['panel_detection_latency_frames']} frames); boxes `{item['key_box_counts']}`; candidate `{item['modal_sequence_candidate'] or 'N/A'}`; status `{item['sequence_status']}`."
        for item in all_episodes
    )
    lines.extend([
        "",
        f"- Manual review: `{review_root / 'index.html'}`",
        f"- Review CSV: `{review_root / 'review_items.csv'}`",
    ])
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_review(review_root, all_episodes, review_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose PRESS panel presence and temporal sequence decoding across replay sessions.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args.config, args.session_root, args.review_root)
    print(json.dumps({
        "sessions": summary["session_count"],
        "episodes": summary["episode_count"],
        "panel_presence_recall": summary["panel_presence_recall"],
        "sequence_ready_episodes": summary["sequence_ready_episode_count"],
        "review_html": summary["review_html"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
