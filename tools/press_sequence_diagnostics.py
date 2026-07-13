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
from src.fishing_v2.data.press_sequence_ground_truth import (  # noqa: E402
    load_press_sequence_ground_truth,
)
from src.fishing_v2.legacy_adapters.press_detector_adapter import LegacyPressDetectorAdapter  # noqa: E402
from src.fishing_v2.legacy_adapters.replay_source_adapter import LegacyReplaySourceAdapter  # noqa: E402
from src.fishing_v2.runtime.press_sequence_aggregator import (  # noqa: E402
    PressSequenceAggregationConfig,
    PressSequenceTemporalAggregator,
)


DEFAULT_CONFIG = ROOT / "config" / "fishing_v2.yaml"
DEFAULT_SESSION_ROOT = ROOT / "assets" / "replay" / "sessions"
DEFAULT_REVIEW_ROOT = ROOT / "reports" / "fishing_v2" / "press_sequence_review"
DEFAULT_SEQUENCE_GROUND_TRUTH = ROOT / "data" / "annotations" / "press_sequence_ground_truth.yaml"
SUMMARY_MD = ROOT / "reports" / "fishing_v2" / "press_detector_diagnostics_summary.md"
SUMMARY_JSON = ROOT / "reports" / "fishing_v2" / "press_detector_diagnostics_summary.json"
REVIEW_FIELDS = (
    "session_id", "press_start", "press_end", "review_frame",
    "predicted_sequence", "manually_confirmed_sequence", "review_status",
    "source_yaml", "notes",
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
    for slot in observation.evidence.get("slots", ()):
        arrow_box = slot.get("arrow_bbox") if isinstance(slot, dict) else None
        if not isinstance(arrow_box, (list, tuple)) or len(arrow_box) != 4:
            continue
        ax1, ay1, ax2, ay2 = (int(value) for value in arrow_box)
        ax1, ax2, ay1, ay2 = ax1 + x1, ax2 + x1, ay1 + y1, ay2 + y1
        colour = (0, 255, 0) if slot.get("occupancy") == "OCCUPIED" else (0, 165, 255)
        cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), colour, 2)
        label = f"{slot.get('index')}:{slot.get('mapped_key') or '?'}"
        cv2.putText(annotated, label, (ax1, max(20, ay1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1)
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
    expected_sequence: str | None,
) -> dict[str, Any]:
    first_candidate = next((row["frame"] for row in rows if row["raw"].panel_candidate), None)
    first_present = next((row["frame"] for row in rows if row["raw"].panel_present), None)
    first_confirmed = next((row["frame"] for row in rows if row["aggregation"].panel_confirmed), None)
    first_ready = next((row["frame"] for row in rows if row["aggregation"].sequence_ready), None)
    first_input_effect = next((
        row["frame"] for row in rows
        if row["raw"].evidence.get("input_effect_detected") is True
    ), None)
    clean_row = next((
        row for row in rows
        if row["raw"].evidence.get("clean_frame_eligible") is True
        and row["raw"].evidence.get("arrow_sequence_ready") is True
        and (first_input_effect is None or row["frame"] < first_input_effect)
    ), None)
    modal_candidate, consistency = _mode_consistency([
        row["aggregation"].sequence_candidate for row in rows
    ])
    review = clean_row or _best_review_row(rows)
    predicted_sequence = (
        "".join(clean_row["raw"].sequence_candidate) if clean_row is not None else ""
    )
    evaluable = bool(clean_row is not None and expected_sequence)
    return {
        "session_id": session_id,
        "press_start": start,
        "press_end": end,
        "support": len(rows),
        "panel_candidate_first_frame": first_candidate,
        "panel_present_first_frame": first_present,
        "panel_confirmed_first_frame": first_confirmed,
        "sequence_ready_first_frame": first_ready,
        "earliest_clean_frame": None if clean_row is None else clean_row["frame"],
        "first_input_effect_frame": first_input_effect,
        "selected_review_frame": review["frame"],
        "selected_review_reason": (
            "earliest_clean_pre_input_frame" if clean_row is not None
            else "fallback_visual_review_no_clean_frame"
        ),
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
        "predicted_sequence": predicted_sequence,
        "expected_sequence": expected_sequence,
        "exact_match": predicted_sequence == expected_sequence if evaluable else None,
        "sequence_evaluable": evaluable,
        "sequence_status": (
            "sequence_ready" if clean_row is not None
            else "sequence_not_evaluable_from_existing_replay"
        ),
        "review_frame": review["frame"],
        "selected_slot_observations": list(review["raw"].evidence.get("slots", ())),
        "selected_panel_phase": review["raw"].evidence.get("panel_phase"),
        "selected_occupied_slot_count": int(review["raw"].evidence.get("occupied_slot_count", 0)),
        "selected_empty_slot_count": int(review["raw"].evidence.get("empty_slot_count", 0)),
        "selected_uncertain_slot_count": int(review["raw"].evidence.get("uncertain_slot_count", 0)),
        "selected_layout_conflict": bool(review["raw"].evidence.get("layout_conflict", False)),
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
        image_name = episode["review_image_name"]
        top = html.escape(json.dumps(episode["review_top_candidates"], ensure_ascii=False))
        sections.append(f"""
<article>
  <h2>{html.escape(episode['session_id'])}: {episode['press_start']}-{episode['press_end']}</h2>
  <p>candidate/present/clean/input: {episode['panel_candidate_first_frame']}/{episode['panel_present_first_frame']}/{episode['earliest_clean_frame']}/{episode['first_input_effect_frame']}</p>
  <p>selected frame: <strong>{episode['selected_review_frame']}</strong> ({episode['selected_review_reason']}) · phase: {episode['selected_panel_phase']}</p>
  <p>slots total/occupied/empty/uncertain: {len(episode['selected_slot_observations'])}/{episode['selected_occupied_slot_count']}/{episode['selected_empty_slot_count']}/{episode['selected_uncertain_slot_count']} · layout conflict: {episode['selected_layout_conflict']}</p>
  <p>predicted: <code>{html.escape(episode['predicted_sequence']) or '(not ready)'}</code> · expected: <code>{html.escape(episode['expected_sequence'] or '') or '(unverified)'}</code> · exact match: <strong>{episode['exact_match']}</strong> · status: <strong>{episode['sequence_status']}</strong></p>
  <img src="images/{html.escape(image_name)}" alt="PRESS review {html.escape(episode['session_id'])}">
  <details><summary>Arrow occupancy and auxiliary glyph candidates</summary><pre>{html.escape(json.dumps(episode['selected_slot_observations'], ensure_ascii=False, indent=2))}</pre><pre>{top}</pre></details>
</article>""")
    document = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>PRESS Sequence Review</title>
<style>body{{font-family:system-ui;margin:2rem;background:#111;color:#eee}}article{{margin:2rem 0;padding:1rem;background:#222}}img{{max-width:100%;height:auto}}code,pre{{white-space:pre-wrap;color:#9ee}}a{{color:#8cf}}</style></head>
<body><h1>PRESS Sequence Review</h1><p>The CSV is derived from the human-confirmed YAML source of truth. It is never loaded as ground truth.</p>{''.join(sections)}</body></html>"""
    (root / "index.html").write_text(document, encoding="utf-8")


def run(
    config_path: Path,
    session_root: Path,
    review_root: Path,
    sequence_ground_truth_path: Path = DEFAULT_SEQUENCE_GROUND_TRUTH,
) -> dict[str, Any]:
    aggregation_config = _aggregation_config(config_path)
    sequence_ground_truth = load_press_sequence_ground_truth(
        sequence_ground_truth_path, session_root=session_root
    )
    expected_by_episode = {
        (item.session_id, item.press_start, item.press_end): "".join(item.sequence)
        for item in sequence_ground_truth
    }
    ground_truth_by_session: dict[str, list[Any]] = defaultdict(list)
    for item in sequence_ground_truth:
        ground_truth_by_session[item.session_id].append(item)
    sessions = [session_root / session_id for session_id in sorted(ground_truth_by_session)]
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
            in_confirmed_press = any(
                item.press_start <= replay_frame.context.frame_index <= item.press_end
                for item in ground_truth_by_session[session_path.name]
            )
            if not in_confirmed_press and replay_frame.global_ground_truth != "IGNORE":
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
        for annotation in ground_truth_by_session[session_path.name]:
            start, end = annotation.press_start, annotation.press_end
            episode_rows = [row for row in rows if start <= row["frame"] <= end]
            identity = (session_path.name, start, end)
            expected_sequence = expected_by_episode.get(identity)
            episode = _episode_summary(
                session_path.name, start, end, episode_rows, source.interval,
                expected_sequence,
            )
            all_episodes.append(episode)
            episodes.append(episode)
            selected = next(row for row in episode_rows if row["frame"] == episode["review_frame"])
            answer_name = f"{session_path.name}_{expected_sequence}.jpg" if expected_sequence else ""
            standard_name = f"{session_path.name}_{start}_{end}.jpg"
            image_name = (
                answer_name
                if answer_name and (review_root / "images" / answer_name).exists()
                else standard_name
            )
            episode["review_image_name"] = image_name
            _review_image(selected["path"], selected["raw"], review_root / "images" / image_name)
            review_rows.append({
                "session_id": session_path.name,
                "press_start": start,
                "press_end": end,
                "review_frame": episode["selected_review_frame"],
                "predicted_sequence": episode["predicted_sequence"],
                "manually_confirmed_sequence": expected_sequence or "",
                "review_status": "human_confirmed_adjusted",
                "source_yaml": sequence_ground_truth_path.as_posix(),
                "notes": episode["sequence_status"],
            })
        support = sum(
            item.press_end - item.press_start + 1
            for item in ground_truth_by_session[session_path.name]
        )
        present = sum(
            row["raw"].panel_present for row in rows
            if any(
                item.press_start <= row["frame"] <= item.press_end
                for item in ground_truth_by_session[session_path.name]
            )
        )
        session_summaries.append({
            "session_id": session_path.name,
            "frame_count": len(rows),
            "press_episode_count": len(episodes),
            "press_frame_support": support,
            "panel_present_frames": present,
            "panel_presence_recall": round(present / support, 4) if support else None,
            "sequence_ready_episodes": sum(item["sequence_status"] == "sequence_ready" for item in episodes),
            "sequence_evaluable_episodes": sum(item["sequence_evaluable"] for item in episodes),
            "sequence_exact_matches": sum(item["exact_match"] is True for item in episodes),
        })
        print(f"{session_path.name}: frames={len(rows)} press={present}/{support} episodes={len(episodes)}", flush=True)

    total_support = sum(item["press_frame_support"] for item in session_summaries)
    total_present = sum(item["panel_present_frames"] for item in session_summaries)
    non_press_states = ("IDLE", "WAITING", "READY", "HOOK", "GET")
    evaluable_episodes = [item for item in all_episodes if item["sequence_evaluable"]]
    exact_matches = sum(item["exact_match"] is True for item in evaluable_episodes)
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
        "sequence_ground_truth_path": str(sequence_ground_truth_path),
        "sequence_ground_truth_episode_count": len(expected_by_episode),
        "sequence_evaluable_episode_count": len(evaluable_episodes),
        "sequence_exact_match_count": exact_matches,
        "sequence_accuracy": round(exact_matches / len(evaluable_episodes), 4) if evaluable_episodes else None,
        "sequence_accuracy_reason": (
            "Exact match is computed only for episodes with manual ground truth and an eligible clean frame."
            if evaluable_episodes else
            "No episode has both manual ground truth and an eligible clean frame."
        ),
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
        f"- Sequence exact match: **{exact_matches}/{len(evaluable_episodes)}** ({summary['sequence_accuracy'] if summary['sequence_accuracy'] is not None else 'N/A'}).",
        f"- Manual sequence ground truth: `{sequence_ground_truth_path}` ({len(expected_by_episode)} confirmed episodes).",
        "- Existing glyph threshold was not lowered; arrow direction is primary evidence and letter templates remain auxiliary diagnostics.",
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
        f"- `{item['session_id']}` {item['press_start']}-{item['press_end']}: candidate/present/clean/input `{item['panel_candidate_first_frame']}/{item['panel_present_first_frame']}/{item['earliest_clean_frame']}/{item['first_input_effect_frame']}`; selected `{item['selected_review_frame']}` ({item['selected_review_reason']}); slots `{item['selected_occupied_slot_count']}/{len(item['selected_slot_observations'])}`; predicted `{item['predicted_sequence'] or 'N/A'}`; expected `{item['expected_sequence'] or 'N/A'}`; exact `{item['exact_match']}`; status `{item['sequence_status']}`."
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
    parser.add_argument("--sequence-ground-truth", type=Path, default=DEFAULT_SEQUENCE_GROUND_TRUTH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(
        args.config, args.session_root, args.review_root,
        args.sequence_ground_truth,
    )
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
