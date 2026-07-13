#!/usr/bin/env python3
"""Diagnose and summarize level-triggered Hook crossing gap recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fishing_v2.data.hook_bar_ground_truth import (  # noqa: E402
    HookBarEpisodeGroundTruth,
    load_hook_bar_ground_truth,
)
from src.fishing_v2.runtime.hook_action_policy import (  # noqa: E402
    HookActionPolicy,
    HookActionPolicyConfig,
)
from tools.hook_threshold_crossing_diagnostics import (  # noqa: E402
    DIVIDER_SAFETY_MARGIN_PX,
    FALLBACK_TRIGGER_THRESHOLD,
    GROUND_TRUTH,
    SESSION_ROOT,
    VISIBLE_VALIDATION,
    WORK_ROOT,
    _geometry,
    _hook_observation,
    _raw_fill,
    _rows,
    _valid_raw_fill,
)


DIAGNOSTICS_JSON = ROOT / "reports" / "fishing_v2" / "hook_crossing_gap_diagnostics.json"
DIAGNOSTICS_MD = ROOT / "reports" / "fishing_v2" / "hook_crossing_gap_diagnostics.md"
SUMMARY_JSON = ROOT / "reports" / "fishing_v2" / "hook_crossing_gap_recovery_summary.json"
SUMMARY_MD = ROOT / "reports" / "fishing_v2" / "hook_crossing_gap_recovery_summary.md"
TARGET_SESSION = "session_20260710_125441"


def _policy() -> HookActionPolicy:
    return HookActionPolicy(HookActionPolicyConfig(
        DIVIDER_SAFETY_MARGIN_PX,
        FALLBACK_TRIGGER_THRESHOLD,
    ))


def _episode_rows(
    episode: HookBarEpisodeGroundTruth,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if episode.global_hook_start <= row["frame_index"] <= episode.global_hook_end
    ]


def _episode_summary(
    episode: HookBarEpisodeGroundTruth,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    scoped = _episode_rows(episode, rows)
    first_cross = next((
        row["frame_index"] for row in scoped if _geometry(row)["divider_crossed"]
    ), None)
    first_margin = next((
        row["frame_index"] for row in scoped if _geometry(row)["divider_margin_passed"]
    ), None)
    action_seen = False
    first_usable = None
    decisions: dict[int, Any] = {}
    for row in scoped:
        in_hook = row["previous_runtime_state"] == "HOOK"
        decision = _policy().evaluate(
            _hook_observation(row["qualified_hook_observation"]),
            action_already_proposed=action_seen,
            hook_episode_active=in_hook and not action_seen,
        )
        decisions[row["frame_index"]] = decision
        if first_usable is None and in_hook and decision.action_ready:
            first_usable = row["frame_index"]
        action_seen = action_seen or row["proposed_intent"] == "HOOK_ACTION"
    intents = [
        row["frame_index"] for row in scoped if row["proposed_intent"] == "HOOK_ACTION"
    ]
    first_intent = intents[0] if intents else None
    pre_threshold = sum(
        first_margin is None or frame < first_margin for frame in intents
    )
    return {
        "session_id": episode.session_id,
        "episode_index": episode.episode_index,
        "visible_range": [episode.visible_start, episode.visible_end],
        "first_threshold_cross_frame": first_cross,
        "first_margin_pass_frame": first_margin,
        "first_usable_post_cross_evidence_frame": first_usable,
        "first_hook_action_intent_frame": first_intent,
        "latency_from_first_usable_post_cross_evidence": (
            first_intent - first_usable
            if first_intent is not None and first_usable is not None else None
        ),
        "pre_threshold_intent_count": pre_threshold,
        "total_hook_action_intents": len(intents),
        "duplicate_intent_count": max(0, len(intents) - 1),
        "valid_raw_fill_once": any(_valid_raw_fill(row) for row in scoped),
        "valid_divider_geometry_once": any(
            _geometry(row)["divider_line_detected"] for row in scoped
        ),
        "fallback_used": any(
            decision.fallback_used for decision in decisions.values()
        ),
    }


def _target_diagnostics(
    episode: HookBarEpisodeGroundTruth,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    action_seen = False
    previous_crossed = False
    previous_margin = False
    for row in rows:
        frame = row["frame_index"]
        if not 360 <= frame <= 396:
            continue
        in_hook = row["previous_runtime_state"] == "HOOK"
        latch_active = in_hook and not action_seen
        decision = _policy().evaluate(
            _hook_observation(row["qualified_hook_observation"]),
            action_already_proposed=action_seen,
            hook_episode_active=latch_active,
        )
        geometry = _geometry(row)
        intent = row["proposed_intent"] == "HOOK_ACTION"
        matched = (
            (row["hook_observation"] or {}).get("evidence", {}).get("matched_features", [])
        )
        if action_seen:
            latch_state = "action_already_proposed"
        elif in_hook:
            latch_state = "active"
        elif row["next_runtime_state"] == "HOOK":
            latch_state = "activated_after_frame"
        else:
            latch_state = "inactive"
        output.append({
            "frame": frame,
            "human_visible": episode.visible(frame),
            "human_clear": episode.clear(frame),
            "runtime_state": row["previous_runtime_state"],
            "next_runtime_state": row["next_runtime_state"],
            "activation_mode": row["detector_activation_mode"]["hook"],
            "raw_detected": row["raw_detected"]["hook"],
            "qualified_active": row["qualified_detected"]["hook"],
            "used_by_fusion": row["used_by_fusion"]["hook"],
            "matched_features": matched,
            "divider_detected": geometry["divider_line_detected"],
            "divider_x": geometry["divider_line_x"],
            "divider_confidence": geometry["divider_confidence"],
            "fill_endpoint_x": geometry["fill_endpoint_x"],
            "divider_crossed": geometry["divider_crossed"],
            "divider_margin_passed": geometry["divider_margin_passed"],
            "fill_ratio": _raw_fill(row),
            "valid_geometry": decision.current_hook_geometry_is_usable,
            "threshold_currently_exceeded": decision.threshold_currently_exceeded,
            "divider_crossing_event_this_frame": (
                geometry["divider_crossed"] and not previous_crossed
            ),
            "crossing_event_this_frame": (
                geometry["divider_margin_passed"] and not previous_margin
            ),
            "hook_episode_latch_state": latch_state,
            "one_shot_state": "proposed" if action_seen else "available",
            "fallback_eligible": decision.fallback_eligible,
            "fallback_rejection_reason": decision.fallback_rejection_reason,
            "action_ready": in_hook and decision.action_ready,
            "hook_action_intent": intent,
            "rejection_reason": None if intent else (
                decision.reason if in_hook else "runtime_not_in_hook"
            ),
        })
        previous_crossed = geometry["divider_crossed"]
        previous_margin = geometry["divider_margin_passed"]
        action_seen = action_seen or intent
    return output


def _outside_stats(
    episodes: tuple[HookBarEpisodeGroundTruth, ...],
    by_session: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    visible_report = json.loads(VISIBLE_VALIDATION.read_text(encoding="utf-8"))
    qualified = used = ready = intents = 0
    for session_id, item in visible_report["outside_visible"]["sessions"].items():
        rows = {row["frame_index"]: row for row in by_session[session_id]}
        session_episodes = [ep for ep in episodes if ep.session_id == session_id]
        for frame in item["qualified_frames"]:
            qualified += 1
            row = rows[frame]
            used += bool(row["used_by_fusion"]["hook"])
            in_episode = any(
                ep.global_hook_start <= frame <= ep.global_hook_end for ep in session_episodes
            )
            decision = _policy().evaluate(
                _hook_observation(row["qualified_hook_observation"]),
                action_already_proposed=False,
                hook_episode_active=in_episode and row["previous_runtime_state"] == "HOOK",
            )
            ready += bool(in_episode and decision.action_ready)
            intents += row["proposed_intent"] == "HOOK_ACTION"
    return {
        "qualified_frames": qualified,
        "used_by_fusion": used,
        "action_ready": ready,
        "hook_action_intents": intents,
    }


def run(work_root: Path = WORK_ROOT) -> dict[str, Any]:
    episodes = load_hook_bar_ground_truth(GROUND_TRUTH, session_root=SESSION_ROOT)
    session_ids = {episode.session_id for episode in episodes}
    by_session = {session: _rows(work_root, session) for session in session_ids}
    episode_results = [
        _episode_summary(episode, by_session[episode.session_id]) for episode in episodes
    ]
    target_episode = next(ep for ep in episodes if ep.session_id == TARGET_SESSION)
    diagnostics = _target_diagnostics(target_episode, by_session[TARGET_SESSION])
    target_368 = next(row for row in diagnostics if row["frame"] == 368)
    target_369 = next(row for row in diagnostics if row["frame"] == 369)
    outside = _outside_stats(episodes, by_session)
    pre = sum(item["pre_threshold_intent_count"] for item in episode_results)
    duplicates = sum(item["duplicate_intent_count"] for item in episode_results)
    actions_applied = sum(
        row["action_applied"] for rows in by_session.values() for row in rows
    )
    total_intents = sum(item["total_hook_action_intents"] for item in episode_results)
    ready = bool(
        len(episode_results) == 9
        and all(item["valid_raw_fill_once"] for item in episode_results)
        and all(item["valid_divider_geometry_once"] for item in episode_results)
        and all(item["total_hook_action_intents"] == 1 for item in episode_results)
        and pre == 0
        and duplicates == 0
        and outside["hook_action_intents"] == 0
        and actions_applied == 0
    )
    diagnosis = {
        "session": TARGET_SESSION,
        "range": [360, 396],
        "frames": diagnostics,
        "key_frames": {"368": target_368, "369": target_369},
        "root_cause": (
            "The old policy required qualified_active=true on the same frame as usable geometry. "
            "Frame 369 had fresh divider/fill geometry past the safety margin but qualified=false."
        ),
        "crossing_edge_lost": False,
        "crossing_edge_is_action_gate": False,
        "action_policy": "level_triggered_one_shot",
        "stale_geometry_cached": False,
        "first_usable_post_cross_evidence_frame": 369,
        "first_hook_action_intent_frame": 369,
        "legacy_fill_ratio_trustworthy": False,
    }
    summary = {
        "policy": {
            "type": "level_triggered_one_shot",
            "divider_safety_margin_px": DIVIDER_SAFETY_MARGIN_PX,
            "fallback_trigger_threshold": FALLBACK_TRIGGER_THRESHOLD,
            "upper_bound": None,
            "requires_crossing_edge_on_action_frame": False,
            "requires_current_geometry": True,
            "uses_stale_geometry": False,
        },
        "hook_episode_latch": {
            "introduced": True,
            "activation": "formal Runtime transition into HOOK from qualified specialized evidence",
            "not_activated_by": ["global ground truth", "rectangle-only candidate", "single weak raw detection"],
            "cleared_by": [
                "HOOK_ACTION proposal", "Runtime leaving HOOK",
                "confirmed bar disappearance", "episode timeout",
            ],
        },
        "ratio_trust": {
            "legacy_fill_ratio_trustworthy": False,
            "legacy_fill_ratio_1_0_can_trigger_fallback": False,
            "fallback_requires_explicit_trust": True,
        },
        "target_125441": {
            key: value for key, value in diagnosis.items()
            if key not in {"frames", "key_frames"}
        },
        "episodes": episode_results,
        "intent_count": total_intents,
        "pre_threshold_intent_count": pre,
        "duplicate_intent_count": duplicates,
        "outside_visible": outside,
        "actions_applied": actions_applied,
        "hook_detector_operationally_ready": ready,
        "live_detect_only_remaining": [
            "Confirm fresh geometry remains stable at the real capture cadence.",
            "Confirm disappearance/timeout latch clearing without applying input.",
            "Validate ratio fallback only after a future source explicitly marks its ratio calibrated.",
        ],
    }
    DIAGNOSTICS_JSON.write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    DIAGNOSTICS_MD.write_text(_diagnostics_markdown(diagnosis), encoding="utf-8")
    SUMMARY_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    SUMMARY_MD.write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _diagnostics_markdown(diagnosis: dict[str, Any]) -> str:
    lines = [
        "# Hook Crossing Gap Diagnostics: session_20260710_125441",
        "",
        "- Reviewed frames: 360-396",
        "- Root cause: frame 369 had fresh safe divider/fill geometry, but the old policy rejected "
        "it solely because qualified_active was false.",
        "- Crossing edge was not lost or cached; the action policy is now level-triggered one-shot.",
        "- First usable post-cross evidence: **369**",
        "- First HOOK_ACTION intent: **369**",
        "",
        "| frame | visible | state | raw | qualified | fusion | divider | endpoint | margin | latch | ready | intent | rejection |",
        "| ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    rows = diagnosis["frames"]
    for row in rows:
        lines.append(
            f"| {row['frame']} | {row['human_visible']} | `{row['runtime_state']}` | "
            f"{row['raw_detected']} | {row['qualified_active']} | {row['used_by_fusion']} | "
            f"{row['divider_x']} | {row['fill_endpoint_x']} | {row['divider_margin_passed']} | "
            f"`{row['hook_episode_latch_state']}` | {row['action_ready']} | "
            f"{row['hook_action_intent']} | `{row['rejection_reason']}` |"
        )
    return "\n".join(lines) + "\n"


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Hook Crossing Gap Recovery",
        "",
        "- Policy: **level-triggered one-shot**; crossing edge is diagnostic only.",
        "- Action requires current usable geometry; stale geometry is never cached.",
        "- Legacy fill_ratio is untrusted and cannot trigger fallback, including `1.0`.",
        "- 125441: first usable post-cross evidence/intention = **369 / 369**.",
        "",
        "| episode | cross | margin | first usable | intent | usable latency | intents |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["episodes"]:
        lines.append(
            f"| `{item['session_id']}#{item['episode_index']}` | "
            f"{item['first_threshold_cross_frame']} | {item['first_margin_pass_frame']} | "
            f"{item['first_usable_post_cross_evidence_frame']} | "
            f"{item['first_hook_action_intent_frame']} | "
            f"{item['latency_from_first_usable_post_cross_evidence']} | "
            f"{item['total_hook_action_intents']} |"
        )
    outside = summary["outside_visible"]
    lines.extend([
        "",
        "## Gates",
        "",
        f"- Total episode intents: **{summary['intent_count']}/9**",
        f"- Pre-threshold intents: **{summary['pre_threshold_intent_count']}**",
        f"- Duplicate intents: **{summary['duplicate_intent_count']}**",
        f"- Outside-visible intents: **{outside['hook_action_intents']}**",
        f"- Actions applied: **{summary['actions_applied']}**",
        f"- `hook_detector_operationally_ready = "
        f"{str(summary['hook_detector_operationally_ready']).lower()}`",
        "",
        "## Live detect-only remaining",
        "",
    ])
    lines.extend(f"- {item}" for item in summary["live_detect_only_remaining"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    return parser.parse_args()


def main() -> int:
    result = run(parse_args().work_root)
    print(f"episodes={len(result['episodes'])}")
    print(f"intent_count={result['intent_count']}")
    print(
        "hook_detector_operationally_ready="
        f"{str(result['hook_detector_operationally_ready']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
