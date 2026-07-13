"""Compare isolated prototype Prompt predictions with scripted Prompt replay."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.data.prompt_annotation import load_prompt_ground_truth  # noqa: E402
from src.fishing_v2.data.prompt_observation_dataset import (  # noqa: E402
    FORMAL_SESSION_IDS,
)
from src.fishing_v2.data.prompt_roi import load_approved_prompt_roi  # noqa: E402
from src.fishing_v2.evaluation.prompt_prototype_evaluation import evaluate_loso  # noqa: E402
from src.fishing_v2.perception.prototype_prompt_observer import PrototypePromptObserver  # noqa: E402
from src.fishing_v2.replay.v2_replay_runner import V2ReplayRunner  # noqa: E402
from src.fishing_v2.runtime.runtime_controller import ActionExecutionMode  # noqa: E402


PROMPT_TO_INTENT = {
    "CAST": "IDLE_CAST",
    "START_HOOK": "READY_BITE",
}


def _transitions(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, str, str]]:
    return [
        (int(row["frame_index"]), str(row["previous_runtime_state"]), str(row["next_runtime_state"]))
        for row in rows
        if row["previous_runtime_state"] != row["next_runtime_state"]
    ]


def _false_intents(
    rows: Sequence[Mapping[str, Any]], prompt_labels: Mapping[int, Any]
) -> dict[str, list[int]]:
    result = {name: [] for name in ("CAST", "START_HOOK", "HOOK_ACTION", "PRESS_SEQUENCE", "COLLECT")}
    for row in rows:
        intent = str(row["proposed_intent"])
        if intent not in result:
            continue
        frame = int(row["frame_index"])
        expected_prompt = prompt_labels[frame].value
        global_state = str(row["global_ground_truth"])
        get_present = bool(row["qualified_detected"]["get"])
        invalid = False
        if intent in PROMPT_TO_INTENT:
            invalid = expected_prompt != PROMPT_TO_INTENT[intent]
        elif intent == "HOOK_ACTION":
            invalid = global_state != "HOOK"
        elif intent == "PRESS_SEQUENCE":
            invalid = global_state != "PRESS"
        elif intent == "COLLECT":
            invalid = global_state != "GET"
        if intent == "CAST" and get_present:
            invalid = True
        if invalid:
            result[intent].append(frame)
    return result


def _prompt_event_outcomes(
    events: Sequence[Mapping[str, Any]],
    session_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for label in ("READY_BITE", "HOOK_INSTRUCTION", "PRESS_INSTRUCTION"):
        relevant = [item for item in events if item["session_id"] == session_id and item["label"] == label]
        details = []
        for item in relevant:
            first_prompt = item["first_correct_predicted_frame"]
            detector = "hook" if label == "HOOK_INSTRUCTION" else "press" if label == "PRESS_INSTRUCTION" else None
            specialized_frames = [
                int(row["frame_index"])
                for row in rows
                if detector
                and int(item["ground_truth_start"]) <= int(row["frame_index"]) <= int(item["ground_truth_end"])
                and (
                    bool(row["qualified_detected"][detector])
                    if detector == "hook"
                    else bool(row["press_panel_present_raw"])
                )
            ]
            first_specialized = min(specialized_frames) if specialized_frames else None
            details.append({
                "ground_truth_start": item["ground_truth_start"],
                "ground_truth_end": item["ground_truth_end"],
                "first_prompt_frame": first_prompt,
                "first_specialized_evidence_frame": first_specialized,
                "prompt_before_or_at_specialized_evidence": (
                    first_prompt is not None
                    and (first_specialized is None or int(first_prompt) <= first_specialized)
                ),
                "armed_fallback_evidence_available": first_specialized is not None,
            })
        result[label] = {
            "episodes": len(relevant),
            "detected": sum(bool(item["detected_at_least_once"]) for item in relevant),
            "latency_frames": [item["detection_latency_frames"] for item in relevant],
            "details": details,
        }
    return result


def _get_outcomes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    episodes: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for row in rows:
        if row["global_ground_truth"] == "GET":
            current.append(row)
        elif current:
            episodes.append(current)
            current = []
    if current:
        episodes.append(current)
    return {
        "episodes": len(episodes),
        "runtime_get_episodes": sum(any(row["next_runtime_state"] == "GET" for row in episode) for episode in episodes),
        "false_cast_frames": [
            int(row["frame_index"])
            for episode in episodes
            for row in episode
            if row["proposed_intent"] == "CAST"
        ],
    }


def compare_runs(
    session_id: str,
    scripted_rows: Sequence[Mapping[str, Any]],
    predicted_rows: Sequence[Mapping[str, Any]],
    prompt_labels: Mapping[int, Any],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(scripted_rows) != len(predicted_rows):
        raise ValueError(f"Replay lengths differ for {session_id}")
    agreement = sum(
        left["next_runtime_state"] == right["next_runtime_state"]
        for left, right in zip(scripted_rows, predicted_rows)
    )
    first_divergence = next((
        {
            "frame": int(predicted["frame_index"]),
            "scripted_state": scripted["next_runtime_state"],
            "predicted_state": predicted["next_runtime_state"],
            "scripted_reason": scripted["transition_reason"],
            "predicted_reason": predicted["transition_reason"],
            "observer": predicted.get("prompt_observer_raw"),
        }
        for scripted, predicted in zip(scripted_rows, predicted_rows)
        if scripted["next_runtime_state"] != predicted["next_runtime_state"]
    ), None)
    scripted_transitions = set(_transitions(scripted_rows))
    predicted_transitions = set(_transitions(predicted_rows))
    union = scripted_transitions | predicted_transitions
    predicted_intents = Counter(
        row["proposed_intent"] for row in predicted_rows if row["proposed_intent"] != "NONE"
    )
    scripted_intents = Counter(
        row["proposed_intent"] for row in scripted_rows if row["proposed_intent"] != "NONE"
    )
    false_intents = _false_intents(predicted_rows, prompt_labels)
    missed_types = sorted(
        intent for intent, count in scripted_intents.items() if count and not predicted_intents[intent]
    )
    prompt_events = _prompt_event_outcomes(events, session_id, predicted_rows)
    get_outcome = _get_outcomes(predicted_rows)
    complete = bool(predicted_rows) and int(predicted_rows[-1]["frame_index"]) == 600
    sync_frames = sum(row["next_runtime_state"] == "SYNC_REQUIRED" for row in predicted_rows)
    fail_frames = sum(row["next_runtime_state"] == "FAIL" for row in predicted_rows)
    actions_applied = sum(bool(row["action_applied"]) for row in predicted_rows)
    final_match = scripted_rows[-1]["next_runtime_state"] == predicted_rows[-1]["next_runtime_state"]
    event_safe = all(item["episodes"] == item["detected"] for item in prompt_events.values())
    get_safe = get_outcome["episodes"] == get_outcome["runtime_get_episodes"] and not get_outcome["false_cast_frames"]
    false_count = sum(len(items) for items in false_intents.values())
    passed = all((complete, not sync_frames, not fail_frames, not actions_applied, final_match, event_safe, get_safe, not false_count, not missed_types))
    return {
        "session": session_id,
        "frames": len(predicted_rows),
        "scripted_final_state": scripted_rows[-1]["next_runtime_state"],
        "predicted_final_state": predicted_rows[-1]["next_runtime_state"],
        "replay_completed": complete,
        "runtime_state_frame_agreement": agreement / len(predicted_rows),
        "transition_agreement": len(scripted_transitions & predicted_transitions) / len(union) if union else 1.0,
        "scripted_transitions": len(scripted_transitions),
        "predicted_transitions": len(predicted_transitions),
        "prompt_events": prompt_events,
        "get_outcome": get_outcome,
        "scripted_proposed_intents": dict(scripted_intents),
        "predicted_proposed_intents": dict(predicted_intents),
        "false_intents": false_intents,
        "false_intent_count": false_count,
        "missed_expected_intent_types": missed_types,
        "sync_required_frames": sync_frames,
        "fail_frames": fail_frames,
        "actions_applied": actions_applied,
        "first_divergence": first_divergence,
        "result": "PASS" if passed else "FAIL",
    }


def build_summary(comparisons: Sequence[Mapping[str, Any]], prompt_summary: Mapping[str, Any]) -> dict[str, Any]:
    complete = sum(bool(item["replay_completed"]) for item in comparisons)
    false_intents = sum(int(item["false_intent_count"]) for item in comparisons)
    missed_intents = sum(len(item["missed_expected_intent_types"]) for item in comparisons)
    sync_frames = sum(int(item["sync_required_frames"]) for item in comparisons)
    fail_frames = sum(int(item["fail_frames"]) for item in comparisons)
    actions_applied = sum(int(item["actions_applied"]) for item in comparisons)
    ready_events = sum(item["prompt_events"]["READY_BITE"]["episodes"] for item in comparisons)
    ready_detected = sum(item["prompt_events"]["READY_BITE"]["detected"] for item in comparisons)
    hook_events = sum(item["prompt_events"]["HOOK_INSTRUCTION"]["episodes"] for item in comparisons)
    hook_detected = sum(item["prompt_events"]["HOOK_INSTRUCTION"]["detected"] for item in comparisons)
    press_events = sum(item["prompt_events"]["PRESS_INSTRUCTION"]["episodes"] for item in comparisons)
    press_detected = sum(item["prompt_events"]["PRESS_INSTRUCTION"]["detected"] for item in comparisons)
    get_events = sum(item["get_outcome"]["episodes"] for item in comparisons)
    get_detected = sum(item["get_outcome"]["runtime_get_episodes"] for item in comparisons)
    ready = all((
        len(comparisons) == 7,
        complete == 7,
        fail_frames == 0,
        actions_applied == 0,
        false_intents == 0,
        missed_intents == 0,
        sync_frames == 0,
        ready_events == ready_detected,
        hook_events == hook_detected,
        press_events == press_detected,
        get_events == get_detected,
        all(item["scripted_final_state"] == item["predicted_final_state"] for item in comparisons),
        prompt_summary["ignore"]["false_acceptance_rate"]["IDLE_CAST"] < 0.10,
    ))
    return {
        "observer": "prototype_v1",
        "mode": "predicted_prompt",
        "formal_sessions": list(FORMAL_SESSION_IDS),
        "replay_complete_count": complete,
        "fail_frames": fail_frames,
        "actions_applied": actions_applied,
        "false_intents": false_intents,
        "missed_expected_intents": missed_intents,
        "sync_required_frames": sync_frames,
        "event_outcomes": {
            "READY_BITE": [ready_detected, ready_events],
            "HOOK_INSTRUCTION": [hook_detected, hook_events],
            "PRESS_INSTRUCTION": [press_detected, press_events],
            "GET": [get_detected, get_events],
        },
        "mean_runtime_state_frame_agreement": float(np.mean([item["runtime_state_frame_agreement"] for item in comparisons])),
        "mean_transition_agreement": float(np.mean([item["transition_agreement"] for item in comparisons])),
        "result_counts": dict(Counter(item["result"] for item in comparisons)),
        "prompt_observer_ready_for_live_detect_only": ready,
        "sessions": list(comparisons),
    }


def write_reports(summary: Mapping[str, Any], report_root: Path) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "predicted_prompt_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Predicted Prompt Replay Summary",
        "",
        "| session | scripted final | predicted final | READY | HOOK hint | PRESS hint | false intents | sync | result |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in summary["sessions"]:
        event = item["prompt_events"]
        lines.append(
            f"| {item['session']} | {item['scripted_final_state']} | {item['predicted_final_state']} | "
            f"{event['READY_BITE']['detected']}/{event['READY_BITE']['episodes']} | "
            f"{event['HOOK_INSTRUCTION']['detected']}/{event['HOOK_INSTRUCTION']['episodes']} | "
            f"{event['PRESS_INSTRUCTION']['detected']}/{event['PRESS_INSTRUCTION']['episodes']} | "
            f"{item['false_intent_count']} | {item['sync_required_frames']} | {item['result']} |"
        )
    lines.extend([
        "",
        f"- Complete replays: **{summary['replay_complete_count']}/7**",
        f"- FAIL frames: **{summary['fail_frames']}**",
        f"- Actions applied: **{summary['actions_applied']}**",
        f"- False / missed intents: **{summary['false_intents']} / {summary['missed_expected_intents']}**",
        f"- SYNC_REQUIRED frames: **{summary['sync_required_frames']}**",
        f"- Mean Runtime state agreement: **{summary['mean_runtime_state_frame_agreement']:.4f}**",
        f"- Mean transition agreement: **{summary['mean_transition_agreement']:.4f}**",
        f"- Gate `prompt_observer_ready_for_live_detect_only`: **{str(summary['prompt_observer_ready_for_live_detect_only']).lower()}**",
        "",
        "## First divergences",
        "",
    ])
    for item in summary["sessions"]:
        lines.append(f"- {item['session']}: `{item['first_divergence']}`")
    (report_root / "predicted_prompt_replay_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "fishing_v2.yaml")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "prompt_observation_v1")
    parser.add_argument("--session-root", type=Path, default=PROJECT_ROOT / "assets" / "replay" / "sessions")
    parser.add_argument("--work-root", type=Path, default=PROJECT_ROOT / "tmp" / "predicted_prompt_replay")
    parser.add_argument("--report-root", type=Path, default=PROJECT_ROOT / "reports" / "fishing_v2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt_summary, _, fold_models = evaluate_loso(
        manifest_path=args.dataset / "manifest.csv",
        dataset_root=args.dataset,
        session_root=args.session_root,
        feature_cache=args.dataset / "features_prototype_v1.npz",
    )
    roi = load_approved_prompt_roi(args.config)
    if roi is None:
        raise ValueError("Predicted Prompt replay requires approved ROI")
    comparisons = []
    for session_id in FORMAL_SESSION_IDS:
        session_path = args.session_root / session_id
        print(f"replay {session_id}: scripted_prompt", flush=True)
        scripted = V2ReplayRunner(args.config).run(
            session_path,
            mode="scripted_prompt",
            report_dir=args.work_root / "scripted",
            action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        )
        print(f"replay {session_id}: predicted_prompt", flush=True)
        predicted = V2ReplayRunner(args.config).run(
            session_path,
            mode="predicted_prompt",
            report_dir=args.work_root / "predicted",
            action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
            prompt_observer=PrototypePromptObserver(fold_models[session_id], roi),
        )
        prompt_labels = load_prompt_ground_truth(
            session_path / "prompt_ground_truth.yaml", len(predicted.rows)
        )
        comparisons.append(compare_runs(
            session_id, scripted.rows, predicted.rows, prompt_labels, prompt_summary["events"]
        ))
    summary = build_summary(comparisons, prompt_summary)
    write_reports(summary, args.report_root)
    print(f"replays_complete: {summary['replay_complete_count']}/7")
    print(f"false_intents: {summary['false_intents']}")
    print(f"sync_required_frames: {summary['sync_required_frames']}")
    print(f"prompt_observer_ready_for_live_detect_only: {summary['prompt_observer_ready_for_live_detect_only']}")
    return 0 if summary["prompt_observer_ready_for_live_detect_only"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
