#!/usr/bin/env python3
"""Summarize Hook crossing/action timing from human ranges and replay telemetry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fishing_v2.data.hook_bar_ground_truth import (  # noqa: E402
    HookBarEpisodeGroundTruth,
    load_hook_bar_ground_truth,
)
from src.fishing_v2.domain.observations import HookObservation  # noqa: E402
from src.fishing_v2.runtime.hook_action_policy import (  # noqa: E402
    HookActionPolicy,
    HookActionPolicyConfig,
)


GROUND_TRUTH = ROOT / "data" / "annotations" / "hook_bar_ground_truth.yaml"
SESSION_ROOT = ROOT / "assets" / "replay" / "sessions"
VISIBLE_VALIDATION = ROOT / "reports" / "fishing_v2" / "hook_bar_visible_validation_summary.json"
WORK_ROOT = ROOT / "tmp" / "scripted_replay_validation_frames"
OUTPUT_JSON = ROOT / "reports" / "fishing_v2" / "hook_threshold_crossing_summary.json"
OUTPUT_MD = ROOT / "reports" / "fishing_v2" / "hook_threshold_crossing_summary.md"
CONFIG = ROOT / "config" / "fishing_v2.yaml"
_HOOK_CONFIG = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["hook"]
DIVIDER_SAFETY_MARGIN_PX = int(_HOOK_CONFIG["divider_safety_margin_px"])
FALLBACK_TRIGGER_THRESHOLD = float(_HOOK_CONFIG["fallback_trigger_threshold"])


def classify_missing_observation_band(
    samples: Iterable[float],
    *,
    detector_missed_band_frames: bool = False,
    calibration_mismatch: bool = False,
    human_action_interrupted: bool = False,
) -> str:
    """Classify why the historical 0.65..0.85 observation band is absent."""
    values = list(samples)
    if calibration_mismatch:
        return "fill_ratio_calibration_error"
    if detector_missed_band_frames:
        return "detector_missing_band_frames"
    if human_action_interrupted:
        return "human_action_interrupted_curve"
    if any(left < 0.65 and right > 0.85 for left, right in zip(values, values[1:])):
        return "sampling_skipped_observation_band"
    return "other"


def _json(value: str) -> Any:
    return json.loads(value) if value else None


def _rows(work_root: Path, session_id: str) -> list[dict[str, Any]]:
    path = work_root / session_id / "v2_replay_frames.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run python tools/validate_scripted_replays.py first"
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                **raw,
                "frame_index": int(raw["frame_index"]),
                "raw_detected": _json(raw["raw_detected"]),
                "qualified_detected": _json(raw["qualified_detected"]),
                "used_by_fusion": _json(raw["used_by_fusion"]),
                "detector_activation_mode": _json(raw["detector_activation_mode"]),
                "hook_observation": _json(raw["hook_observation"]),
                "qualified_hook_observation": _json(raw["qualified_hook_observation"]),
                "action_applied": raw["action_applied"].lower() == "true",
            })
    return rows


def _hook_observation(data: dict[str, Any] | None) -> HookObservation | None:
    return HookObservation(**data) if data else None


def _geometry(row: dict[str, Any]) -> dict[str, Any]:
    evidence = (row["hook_observation"] or {}).get("evidence", {})
    divider = evidence.get("divider_line_x")
    endpoint = evidence.get("fill_endpoint_x")
    crossed = divider is not None and endpoint is not None and float(endpoint) >= float(divider)
    margin = crossed and float(endpoint) >= float(divider) + DIVIDER_SAFETY_MARGIN_PX
    return {
        "divider_line_detected": bool(evidence.get("divider_line_detected")),
        "divider_line_x": divider,
        "divider_confidence": float(evidence.get("divider_confidence", 0.0)),
        "fill_endpoint_x": endpoint,
        "divider_crossed": crossed,
        "divider_margin_passed": margin,
    }


def _first(rows: Iterable[dict[str, Any]], predicate) -> int | None:
    return next((row["frame_index"] for row in rows if predicate(row)), None)


def _raw_fill(row: dict[str, Any]) -> float | None:
    value = (row["hook_observation"] or {}).get("fill_ratio")
    return float(value) if value is not None else None


def _valid_raw_fill(row: dict[str, Any]) -> bool:
    raw = row["hook_observation"] or {}
    features = set((raw.get("evidence") or {}).get("matched_features", ()))
    value = _raw_fill(row)
    return bool("bar_fill" in features and value is not None and value > 0.0)


def _episode_summary(
    episode: HookBarEpisodeGroundTruth,
    session_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        row for row in session_rows
        if episode.visible_start <= row["frame_index"] <= episode.visible_end
    ]
    global_rows = [
        row for row in session_rows
        if episode.global_hook_start <= row["frame_index"] <= episode.global_hook_end
    ]
    valid = [row for row in rows if _valid_raw_fill(row)]
    fill_values = [_raw_fill(row) for row in valid]
    fills = [float(value) for value in fill_values if value is not None]
    jumps = [abs(right - left) for left, right in zip(fills, fills[1:])]
    first_cross = _first(rows, lambda row: _geometry(row)["divider_crossed"])
    first_margin = _first(rows, lambda row: _geometry(row)["divider_margin_passed"])
    intents = [
        row["frame_index"] for row in global_rows
        if row["proposed_intent"] == "HOOK_ACTION"
    ]
    detailed: list[dict[str, Any]] = []
    action_seen = False
    policy = HookActionPolicy(HookActionPolicyConfig(
        DIVIDER_SAFETY_MARGIN_PX,
        FALLBACK_TRIGGER_THRESHOLD,
    ))
    for row in rows:
        qualified = bool(row["qualified_detected"]["hook"])
        in_hook_state = row["previous_runtime_state"] == "HOOK"
        decision = policy.evaluate(
            _hook_observation(row["qualified_hook_observation"]),
            action_already_proposed=action_seen,
            hook_episode_active=in_hook_state and not action_seen,
        )
        action_ready = bool(in_hook_state and decision.action_ready)
        intent = row["proposed_intent"] == "HOOK_ACTION"
        detailed.append({
            "frame": row["frame_index"],
            "fill_ratio": _raw_fill(row),
            **_geometry(row),
            "qualified": qualified,
            "action_ready": action_ready,
            "intent": intent,
            "human_visible": episode.visible(row["frame_index"]),
            "human_clear": episode.clear(row["frame_index"]),
        })
        action_seen = action_seen or intent
    pre_threshold = sum(frame < (first_margin or episode.visible_start) for frame in intents)
    summary = {
        "session_id": episode.session_id,
        "episode_index": episode.episode_index,
        "first_visible_frame": episode.visible_start,
        "first_qualified_frame": _first(rows, lambda row: row["qualified_detected"]["hook"]),
        "first_valid_fill_frame": _first(rows, _valid_raw_fill),
        "first_divider_detected_frame": _first(
            rows, lambda row: _geometry(row)["divider_line_detected"]
        ),
        "first_threshold_cross_frame": first_cross,
        "first_margin_pass_frame": first_margin,
        "first_hook_action_intent_frame": intents[0] if intents else None,
        "action_latency_after_crossing": (
            intents[0] - first_margin if intents and first_margin is not None else None
        ),
        "pre_threshold_action_count": pre_threshold,
        "total_hook_action_intents": len(intents),
        "duplicate_action_count": max(0, len(intents) - 1),
        "last_visible_frame": episode.visible_end,
        "min_fill_ratio": round(min(fills), 4) if fills else None,
        "max_fill_ratio": round(max(fills), 4) if fills else None,
        "max_adjacent_sample_jump": round(max(jumps), 4) if jumps else 0.0 if fills else None,
        "has_0_65_to_0_85_sample": any(0.65 <= value <= 0.85 for value in fills),
        "action_method": "divider" if first_margin is not None else "fallback_or_unavailable",
        "historical_replay_not_fully_evaluable": not bool(intents),
    }
    return summary, detailed


def _outside_rows(
    episodes: tuple[HookBarEpisodeGroundTruth, ...],
    by_session: dict[str, list[dict[str, Any]]],
    visible_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output: list[dict[str, Any]] = []
    for session_id, item in visible_report["outside_visible"]["sessions"].items():
        rows = {row["frame_index"]: row for row in by_session[session_id]}
        session_episodes = [ep for ep in episodes if ep.session_id == session_id]
        for frame in item["qualified_frames"]:
            row = rows[frame]
            containing = next(
                (ep for ep in session_episodes if ep.global_hook_start <= frame <= ep.global_hook_end),
                None,
            )
            if containing and frame < containing.visible_start:
                category = "pre_visible_same_hook_episode"
                distance = frame - containing.visible_start
            elif containing and frame > containing.visible_end:
                category = "post_visible_same_hook_episode"
                distance = frame - containing.visible_end
            else:
                category = "outside_hook_episode"
                nearest = min(
                    session_episodes,
                    key=lambda ep: min(abs(frame - ep.visible_start), abs(frame - ep.visible_end)),
                )
                distance = (
                    frame - nearest.visible_start if frame < nearest.visible_start
                    else frame - nearest.visible_end
                )
            geometry = _geometry(row)
            in_hook_state = row["previous_runtime_state"] == "HOOK"
            action_ready = bool(
                containing
                and in_hook_state
                and row["qualified_detected"]["hook"]
                and geometry["divider_margin_passed"]
            )
            output.append({
                "session": session_id,
                "frame": frame,
                "category": category,
                "distance_from_visible_range": distance,
                "raw_detected": row["raw_detected"]["hook"],
                "qualified_detected": row["qualified_detected"]["hook"],
                "used_by_fusion": row["used_by_fusion"]["hook"],
                "activation_mode": row["detector_activation_mode"]["hook"],
                "fill_ratio": _raw_fill(row),
                "divider_crossed": geometry["divider_crossed"],
                "action_ready": action_ready,
                "hook_action_intent": row["proposed_intent"] == "HOOK_ACTION",
                "one_shot_guard_result": (
                    "not_in_hook_runtime_state" if not in_hook_state
                    else "no_action_ready_evidence" if not action_ready
                    else "eligible"
                ),
                "runtime_state": row["previous_runtime_state"],
            })
    stats = {
        "qualified_outside_visible_frames": len(output),
        "used_by_fusion_outside_visible_frames": sum(row["used_by_fusion"] for row in output),
        "action_ready_outside_visible_frames": sum(row["action_ready"] for row in output),
        "hook_action_intent_outside_visible_frames": sum(row["hook_action_intent"] for row in output),
        "pre_visible_action_intent_count": sum(
            row["hook_action_intent"] and row["category"] == "pre_visible_same_hook_episode"
            for row in output
        ),
        "post_visible_duplicate_intent_count": sum(
            row["hook_action_intent"] and row["category"] == "post_visible_same_hook_episode"
            for row in output
        ),
        "completely_outside_hook_intent_count": sum(
            row["hook_action_intent"] and row["category"] == "outside_hook_episode"
            for row in output
        ),
    }
    return output, stats


def run(work_root: Path = WORK_ROOT) -> dict[str, Any]:
    episodes = load_hook_bar_ground_truth(GROUND_TRUTH, session_root=SESSION_ROOT)
    by_session = {session: _rows(work_root, session) for session in {ep.session_id for ep in episodes}}
    summaries: list[dict[str, Any]] = []
    curves: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        summary, detailed = _episode_summary(episode, by_session[episode.session_id])
        summaries.append(summary)
        if episode.session_id in {"session_20260710_124419", "session_20260710_125441"}:
            curves[episode.session_id] = detailed

    special = {item["session_id"]: item for item in summaries if item["session_id"] in curves}
    special_analysis = {
        "session_20260710_124419": {
            "classification": classify_missing_observation_band(
                [row["fill_ratio"] for row in curves["session_20260710_124419"] if row["fill_ratio"] is not None],
                calibration_mismatch=True,
            ),
            "explanation": (
                "Raw fill_ratio remained 1.0 before/through the visually measured cyan crossing; "
                "the ratio is not calibrated to the action divider."
            ),
            **special["session_20260710_124419"],
        },
        "session_20260710_125441": {
            "classification": classify_missing_observation_band(
                [row["fill_ratio"] for row in curves["session_20260710_125441"] if row["fill_ratio"] is not None],
                detector_missed_band_frames=True,
            ),
            "secondary_finding": "fill_ratio_calibration_error",
            "explanation": (
                "The cyan fill safely crosses on frame 369, but raw/qualified Hook detection ends "
                "after frame 368; earlier raw ratios are already 1.0 and do not represent crossing."
            ),
            **special["session_20260710_125441"],
        },
    }
    outside, outside_stats = _outside_rows(
        episodes,
        by_session,
        json.loads(VISIBLE_VALIDATION.read_text(encoding="utf-8")),
    )
    pre = sum(item["pre_threshold_action_count"] for item in summaries)
    duplicates = sum(item["duplicate_action_count"] for item in summaries)
    applied = sum(row["action_applied"] for rows in by_session.values() for row in rows)
    ready = bool(
        all(item["first_qualified_frame"] is not None for item in summaries)
        and all(item["first_valid_fill_frame"] is not None for item in summaries)
        and pre == 0
        and all(item["total_hook_action_intents"] == 1 for item in summaries)
        and duplicates == 0
        and outside_stats["hook_action_intent_outside_visible_frames"] == 0
        and applied == 0
    )
    result = {
        "prior_action_policy": {
            "hard_gate": "0.65 <= fill_ratio <= 0.85",
            "upper_bound_existed": True,
            "divider_used": False,
            "one_shot_before_fix": "applied-action guard only",
        },
        "current_action_policy": {
            "primary": "qualified ACTIVE_HOOK_BAR and cyan endpoint >= white divider + margin",
            "divider_safety_margin_px": DIVIDER_SAFETY_MARGIN_PX,
            "fallback_trigger_threshold": FALLBACK_TRIGGER_THRESHOLD,
            "upper_bound": None,
            "one_shot": "proposal guard per Hook episode",
        },
        "episodes": summaries,
        "special_session_analysis": special_analysis,
        "special_session_fill_curves": curves,
        "outside_visible_frames": outside,
        "outside_visible_stats": outside_stats,
        "pre_threshold_action_count": pre,
        "duplicate_action_count": duplicates,
        "actions_applied": applied,
        "hook_action_policy_modified": True,
        "historical_replay_not_fully_evaluable": [
            f"{item['session_id']}#{item['episode_index']}"
            for item in summaries if item["historical_replay_not_fully_evaluable"]
        ],
        "hook_detector_operationally_ready": ready,
        "live_detect_only_remaining": (
            "Confirm crossing timing and raw Hook qualification continuity, especially the "
            "session_20260710_125441 pattern, without applying input."
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Hook Threshold Crossing Validation",
        "",
        "- Prior gate: `0.65 <= fill_ratio <= 0.85` (incorrect hard upper bound).",
        f"- Current primary gate: qualified bar + cyan endpoint past white divider + "
        f"{DIVIDER_SAFETY_MARGIN_PX} px.",
        f"- Fallback: `fill_ratio >= {FALLBACK_TRIGGER_THRESHOLD:.2f}` only when divider is "
        "unavailable; no upper bound.",
        "- One-shot: proposal guard per Hook episode.",
        "",
        "| session / episode | visible | qualified | valid fill | cross | margin | intent | latency | intents |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| `{item['session_id']}#{item['episode_index']}` | {item['first_visible_frame']} | "
            f"{item['first_qualified_frame']} | {item['first_valid_fill_frame']} | "
            f"{item['first_threshold_cross_frame']} | {item['first_margin_pass_frame']} | "
            f"{item['first_hook_action_intent_frame']} | {item['action_latency_after_crossing']} | "
            f"{item['total_hook_action_intents']} |"
        )
    lines.extend([
        "",
        "## Missing historical observation band",
        "",
        f"- 124419: `{special_analysis['session_20260710_124419']['classification']}`.",
        f"- 125441: `{special_analysis['session_20260710_125441']['classification']}` "
        "(secondary: `fill_ratio_calibration_error`).",
        "",
        "## Outside-visible operational risk",
        "",
        f"- Qualified: {outside_stats['qualified_outside_visible_frames']}",
        f"- Used by Fusion: {outside_stats['used_by_fusion_outside_visible_frames']}",
        f"- Action-ready: {outside_stats['action_ready_outside_visible_frames']}",
        f"- HOOK_ACTION intents: {outside_stats['hook_action_intent_outside_visible_frames']}",
        "",
        f"- Pre-threshold intents: **{pre}**",
        f"- Duplicate intents: **{duplicates}**",
        f"- Actions applied: **{applied}**",
        f"- `hook_detector_operationally_ready = {str(ready).lower()}`",
        f"- Historical replay not fully evaluable: "
        f"`{result['historical_replay_not_fully_evaluable']}`",
        "- Live detect-only remaining: " + result["live_detect_only_remaining"],
    ])
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args().work_root)
    print(f"episodes={len(result['episodes'])}")
    print(f"pre_threshold_action_count={result['pre_threshold_action_count']}")
    print(f"duplicate_action_count={result['duplicate_action_count']}")
    print(
        "hook_detector_operationally_ready="
        f"{str(result['hook_detector_operationally_ready']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
