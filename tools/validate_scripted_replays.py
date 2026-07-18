"""Validate scripted Prompt replay across every formally annotated session.

The full per-frame replay artifacts are written under ``tmp``.  Only compact
per-session summaries and the cross-session gate report are written to the
tracked reports directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.data.prompt_annotation import (  # noqa: E402
    FINAL_PROMPT_ANNOTATION_KINDS,
    validate_prompt_segments,
)
from src.fishing_v2.replay.v2_replay_runner import V2ReplayRunner  # noqa: E402
from src.fishing_v2.runtime.runtime_controller import ActionExecutionMode  # noqa: E402


FORMAL_SESSION_IDS = (
    "session_20260709_192315",
    "session_20260710_061220",
    "session_20260710_123210",
    "session_20260710_124419",
    "session_20260710_125441",
    "session_20260710_130308",
    "session_20260710_131254",
)
TRIAL_SESSION_ID = "session_20260709_192231"

INTENT_PROMPT_SUPPORT = {
    "CAST": "IDLE_CAST",
    "START_HOOK": "READY_BITE",
    "HOOK_ACTION": "HOOK_INSTRUCTION",
    "PRESS_SEQUENCE": "PRESS_INSTRUCTION",
}


def _manifest_frame_count(session_path: Path) -> int:
    manifest = json.loads((session_path / "manifest.json").read_text(encoding="utf-8"))
    frame_count = manifest.get("frame_count")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 1:
        raise ValueError(f"Invalid manifest frame_count: {session_path}")
    image_format = str(manifest.get("image_format", "jpg"))
    actual_count = len(list((session_path / "frames").glob(f"*.{image_format}")))
    if actual_count != frame_count:
        raise ValueError(
            f"Manifest/frame count mismatch for {session_path.name}: "
            f"manifest={frame_count}, files={actual_count}"
        )
    return frame_count


def validate_annotation(session_path: Path) -> dict[str, Any]:
    """Validate complete, final-vocabulary Prompt ground truth without editing it."""
    frame_count = _manifest_frame_count(session_path)
    source = session_path / "prompt_ground_truth.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError(f"Prompt ground truth requires version: 1: {source}")
    segments = raw.get("segments")
    if not isinstance(segments, list):
        raise ValueError(f"Prompt ground truth requires segments: {source}")
    parsed = validate_prompt_segments(segments, frame_count, allow_deprecated=False)
    counts = Counter()
    for segment in parsed:
        counts[str(segment["observation"])] += int(segment["end"]) - int(segment["start"]) + 1
    legal = {kind.value for kind in FINAL_PROMPT_ANNOTATION_KINDS}
    if set(counts) - legal:
        raise ValueError(f"Unsupported Prompt observations: {sorted(set(counts) - legal)}")
    coverage = sum(counts.values())
    return {
        "valid": coverage == frame_count,
        "total_frames": frame_count,
        "coverage_frames": coverage,
        "coverage_percentage": round(100.0 * coverage / frame_count, 2),
        "label_frame_counts": {label: counts.get(label, 0) for label in sorted(legal)},
        "segment_count": len(parsed),
    }


def _detector_metrics(rows: Sequence[dict[str, Any]], detector: str, state: str) -> dict[str, Any]:
    support = [row for row in rows if row["global_ground_truth"] == state]
    outside = [row for row in rows if row["global_ground_truth"] != state]
    raw = sum(bool(row["raw_detected"][detector]) for row in support)
    qualified = sum(bool(row["qualified_detected"][detector]) for row in support)
    used = sum(bool(row["used_by_fusion"][detector]) for row in support)
    missed_frames = [
        int(row["frame_index"])
        for row in support
        if not bool(row["qualified_detected"][detector])
    ]
    false_frames = [
        int(row["frame_index"])
        for row in outside
        if bool(row["qualified_detected"][detector])
    ]
    return {
        "support_frames": len(support),
        "raw_detected": raw,
        "qualified_detected": qualified,
        "used_by_fusion": used,
        "missed_qualified_frames": len(missed_frames),
        "missed_qualified_frame_samples": missed_frames[:10],
        "false_qualified_frames": len(false_frames),
        "false_qualified_frame_samples": false_frames[:10],
    }


def _first_clean_sequence(rows: Sequence[dict[str, Any]]) -> tuple[int | None, list[str]]:
    for row in rows:
        qualified = row.get("qualified_press_observation") or {}
        evidence = qualified.get("evidence") or {}
        selected = evidence.get("selected_clean_frame")
        candidate = row.get("press_sequence_candidate") or []
        if isinstance(selected, int) and candidate:
            return selected, [str(key) for key in candidate]
    return None, []


def summarize_replay(
    session_id: str,
    rows: Sequence[dict[str, Any]],
    annotation: dict[str, Any],
) -> dict[str, Any]:
    transitions = [
        {
            "frame": int(row["frame_index"]),
            "from": row["previous_runtime_state"],
            "to": row["next_runtime_state"],
            "reason": row["transition_reason"],
        }
        for row in rows
        if row["previous_runtime_state"] != row["next_runtime_state"]
    ]
    proposed = Counter(row["proposed_intent"] for row in rows if row["proposed_intent"] != "NONE")
    sync_rows = [row for row in rows if row["next_runtime_state"] == "SYNC_REQUIRED"]
    false_intents: dict[str, list[int]] = {}
    for intent, expected_prompt in INTENT_PROMPT_SUPPORT.items():
        false_intents[intent] = [
            int(row["frame_index"])
            for row in rows
            if row["proposed_intent"] == intent and row["prompt_observation"] != expected_prompt
        ]
    hook = _detector_metrics(rows, "hook", "HOOK")
    get = _detector_metrics(rows, "get", "GET")
    press = _detector_metrics(rows, "press", "PRESS")
    press_rows = [row for row in rows if row["global_ground_truth"] == "PRESS"]
    non_press_rows = [row for row in rows if row["global_ground_truth"] != "PRESS"]
    clean_frame, frozen_sequence = _first_clean_sequence(rows)
    press.update({
        "panel_candidate_frames": sum(bool(row["press_panel_candidate"]) for row in press_rows),
        "panel_present_frames": sum(bool(row["press_panel_present_raw"]) for row in press_rows),
        "sequence_ready_frames": sum(bool(row["press_sequence_ready"]) for row in press_rows),
        "sequence_intent_count": proposed.get("PRESS_SEQUENCE", 0),
        "frozen_sequence_candidate": frozen_sequence,
        "clean_pre_input_frame": clean_frame,
        "false_panel_candidate_frames": sum(
            bool(row["press_panel_candidate"]) for row in non_press_rows
        ),
        "false_panel_candidate_frame_samples": [
            int(row["frame_index"])
            for row in non_press_rows
            if row["press_panel_candidate"]
        ][:10],
        "false_panel_present_frames": sum(
            bool(row["press_panel_present_raw"]) for row in non_press_rows
        ),
        "false_panel_present_frame_samples": [
            int(row["frame_index"])
            for row in non_press_rows
            if row["press_panel_present_raw"]
        ][:10],
    })
    actions_applied = sum(bool(row["action_applied"]) for row in rows)
    completed = bool(rows) and int(rows[-1]["frame_index"]) == annotation["total_frames"]
    detector_warnings: dict[str, list[str]] = {"hook": [], "press": [], "get": []}
    if hook["missed_qualified_frames"]:
        detector_warnings["hook"].append(
            f"qualified miss {hook['missed_qualified_frames']}/{hook['support_frames']} HOOK frames"
        )
    if hook["false_qualified_frames"]:
        detector_warnings["hook"].append(
            f"false qualified on {hook['false_qualified_frames']} non-HOOK frames"
        )
    if press["panel_present_frames"] < press["support_frames"]:
        detector_warnings["press"].append(
            f"panel present {press['panel_present_frames']}/{press['support_frames']} PRESS frames"
        )
    if press["support_frames"] and clean_frame is None:
        detector_warnings["press"].append("no clean pre-input sequence frame")
    if press["false_qualified_frames"]:
        detector_warnings["press"].append(
            f"false qualified on {press['false_qualified_frames']} non-PRESS frames"
        )
    if press["false_panel_candidate_frames"]:
        detector_warnings["press"].append(
            f"panel candidate on {press['false_panel_candidate_frames']} non-PRESS frames"
        )
    if press["false_panel_present_frames"]:
        detector_warnings["press"].append(
            f"panel present on {press['false_panel_present_frames']} non-PRESS frames"
        )
    if get["missed_qualified_frames"]:
        detector_warnings["get"].append(
            f"qualified miss {get['missed_qualified_frames']}/{get['support_frames']} GET frames"
        )
    if get["false_qualified_frames"]:
        detector_warnings["get"].append(
            f"false qualified on {get['false_qualified_frames']} non-GET frames"
        )
    specialized_targets = {"HOOK": "HOOK", "PRESS": "PRESS", "GET": "GET"}
    unexplained_transitions = [
        {
            **transition,
            "global_ground_truth": row["global_ground_truth"],
            "issue": "specialized_state_entered_outside_global_support",
        }
        for row, transition in zip(
            (row for row in rows if row["previous_runtime_state"] != row["next_runtime_state"]),
            transitions,
        )
        if transition["to"] in specialized_targets
        and row["global_ground_truth"] != specialized_targets[transition["to"]]
    ]
    historical_result_pending_not_fully_evaluable = bool(
        rows
        and rows[-1]["next_runtime_state"] == "RESULT_PENDING"
        and rows[-1]["global_ground_truth"] == "IDLE"
        and rows[-1]["prompt_observation"] == "UNKNOWN"
    )
    terminal_mismatch = bool(
        rows
        and rows[-1]["global_ground_truth"] in {"IDLE", "WAITING", "READY", "HOOK", "PRESS", "GET"}
        and rows[-1]["next_runtime_state"] != rows[-1]["global_ground_truth"]
        and not historical_result_pending_not_fully_evaluable
    )
    if terminal_mismatch:
        unexplained_transitions.append({
            "frame": int(rows[-1]["frame_index"]),
            "from": rows[-1]["next_runtime_state"],
            "to": rows[-1]["global_ground_truth"],
            "reason": "replay_ended_in_state_inconsistent_with_global_ground_truth",
            "global_ground_truth": rows[-1]["global_ground_truth"],
            "issue": "terminal_runtime_state_mismatch",
        })
    has_false_intents = any(false_intents.values())
    failure_reasons: list[str] = []
    if not annotation["valid"]:
        failure_reasons.append("invalid_annotation")
    if not completed:
        failure_reasons.append("incomplete_replay")
    if actions_applied:
        failure_reasons.append("actions_applied_nonzero")
    if sync_rows:
        failure_reasons.append("sync_required")
    if unexplained_transitions:
        failure_reasons.append("unexplained_runtime_transition")
    if has_false_intents:
        failure_reasons.append("false_action_intent")
    runtime_warnings = (
        ["historical_result_pending_not_fully_evaluable"]
        if historical_result_pending_not_fully_evaluable else []
    )
    warnings = [
        item for values in detector_warnings.values() for item in values
    ] + runtime_warnings
    result = "FAIL" if failure_reasons else "PASS_WITH_WARNINGS" if warnings else "PASS"
    return {
        "session": session_id,
        "annotation": annotation,
        "replay_completed": completed,
        "frame_count": len(rows),
        "final_frame": int(rows[-1]["frame_index"]) if rows else None,
        "final_state": rows[-1]["next_runtime_state"] if rows else None,
        "transitions": transitions,
        "proposed_intents": dict(proposed),
        "actions_applied": actions_applied,
        "sync_required_frames": len(sync_rows),
        "first_sync_required_frame": int(sync_rows[0]["frame_index"]) if sync_rows else None,
        "first_sync_required_reason": sync_rows[0]["transition_reason"] if sync_rows else None,
        "hook": hook,
        "press": press,
        "get": get,
        "false_action_intents": false_intents,
        "detector_warnings": detector_warnings,
        "runtime_warnings": runtime_warnings,
        "unexplained_runtime_transitions": unexplained_transitions,
        "failure_reasons": failure_reasons,
        "result": result,
    }


def _metric_cell(metric: dict[str, Any], *, panel: bool = False, sequence: bool = False) -> str:
    if panel:
        return f"{metric['panel_present_frames']}/{metric['support_frames']}"
    if sequence:
        clean = metric["clean_pre_input_frame"]
        return f"ready={metric['sequence_ready_frames']}, clean={clean if clean is not None else 'none'}"
    return f"{metric['qualified_detected']}/{metric['support_frames']}"


def write_session_summary(summary: dict[str, Any], output_root: Path) -> tuple[Path, Path]:
    root = output_root / summary["session"]
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "replay_summary.json"
    md_path = root / "replay_report.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Scripted Prompt Replay: {summary['session']}",
        "",
        f"- Result: **{summary['result']}**",
        f"- Annotation: **{'PASS' if summary['annotation']['valid'] else 'FAIL'}**",
        f"- Frames/final: {summary['frame_count']} / {summary['final_frame']} `{summary['final_state']}`",
        f"- Replay completed: **{summary['replay_completed']}**",
        f"- Actions applied: **{summary['actions_applied']}**",
        f"- SYNC_REQUIRED frames: **{summary['sync_required_frames']}**",
        f"- Proposed intents: `{summary['proposed_intents']}`",
        f"- HOOK qualified/support: {_metric_cell(summary['hook'])}",
        f"- PRESS panel/support: {_metric_cell(summary['press'], panel=True)}",
        f"- PRESS sequence: {_metric_cell(summary['press'], sequence=True)}",
        f"- GET qualified/support: {_metric_cell(summary['get'])}",
        f"- False action intents: `{summary['false_action_intents']}`",
        "",
        "## Warnings",
        "",
    ]
    warnings = [
        f"- {detector.upper()}: {warning}"
        for detector, values in summary["detector_warnings"].items()
        for warning in values
    ]
    warnings.extend(f"- RUNTIME: {warning}" for warning in summary.get("runtime_warnings", []))
    lines.extend(warnings or ["- None"])
    if summary["failure_reasons"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {reason}" for reason in summary["failure_reasons"])
        lines.extend(
            f"- frame {item['frame']}: {item['issue']} "
            f"({item['from']} -> {item['to']}, {item['reason']})"
            for item in summary["unexplained_runtime_transitions"]
        )
    lines.extend(["", "## Runtime transitions", ""])
    lines.extend(
        f"- frame {item['frame']}: {item['from']} -> {item['to']} ({item['reason']})"
        for item in summary["transitions"]
    )
    if not summary["transitions"]:
        lines.append("- None")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def build_cross_session_summary(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(summary["result"] for summary in summaries)
    annotations_valid = all(summary["annotation"]["valid"] for summary in summaries)
    replays_complete = all(summary["replay_completed"] for summary in summaries)
    actions_safe = all(summary["actions_applied"] == 0 for summary in summaries)
    sync_safe = all(summary["sync_required_frames"] == 0 for summary in summaries)
    runtime_safe = all(
        not summary["unexplained_runtime_transitions"]
        and not any(summary["false_action_intents"].values())
        for summary in summaries
    )
    ready = (
        len(summaries) == len(FORMAL_SESSION_IDS)
        and annotations_valid
        and replays_complete
        and actions_safe
        and sync_safe
        and runtime_safe
    )
    detector_totals = {
        detector: {
            field: sum(int(summary.get(detector, {}).get(field, 0)) for summary in summaries)
            for field in ("support_frames", "raw_detected", "qualified_detected", "used_by_fusion")
        }
        for detector in ("hook", "press", "get")
    }
    detector_totals["press"].update({
        field: sum(int(summary.get("press", {}).get(field, 0)) for summary in summaries)
        for field in ("panel_candidate_frames", "panel_present_frames", "sequence_ready_frames", "sequence_intent_count")
    })
    return {
        "formal_sessions": list(FORMAL_SESSION_IDS),
        "trial_excluded": TRIAL_SESSION_ID,
        "annotation_pass_count": sum(summary["annotation"]["valid"] for summary in summaries),
        "replay_complete_count": sum(summary["replay_completed"] for summary in summaries),
        "result_counts": {name: counts.get(name, 0) for name in ("PASS", "PASS_WITH_WARNINGS", "FAIL")},
        "actions_applied_total": sum(summary["actions_applied"] for summary in summaries),
        "sync_required_frames_total": sum(summary["sync_required_frames"] for summary in summaries),
        "detector_totals": detector_totals,
        "false_action_intents_present": any(
            any(summary["false_action_intents"].values()) for summary in summaries
        ),
        "unexplained_runtime_transitions": [
            {"session": summary["session"], **transition}
            for summary in summaries
            for transition in summary["unexplained_runtime_transitions"]
        ],
        "specialized_detector_warnings_do_not_block_prompt_observer_gate": True,
        "specialized_detector_warning_impact": (
            "Warnings remain relevant to Runtime Fusion, but do not invalidate future PromptObserver "
            "prediction scoring against independent human Prompt annotations."
        ),
        "ready_for_prompt_observer": ready,
        "sessions": list(summaries),
    }


def write_cross_session_summary(summary: dict[str, Any], output_root: Path) -> tuple[Path, Path]:
    json_path = output_root.parent / "scripted_replay_validation_summary.json"
    md_path = output_root.parent / "scripted_replay_validation_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Scripted Prompt Replay Validation",
        "",
        "| session | annotation | replay | final state | sync frames | hook | press panel | press sequence | get | result |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for item in summary["sessions"]:
        lines.append(
            f"| `{item['session']}` | {'PASS' if item['annotation']['valid'] else 'FAIL'} | "
            f"{'PASS' if item['replay_completed'] else 'FAIL'} | `{item['final_state']}` | "
            f"{item['sync_required_frames']} | {_metric_cell(item['hook'])} | "
            f"{_metric_cell(item['press'], panel=True)} | {_metric_cell(item['press'], sequence=True)} | "
            f"{_metric_cell(item['get'])} | **{item['result']}** |"
        )
    counts = summary["result_counts"]
    lines.extend([
        "",
        f"- Annotation PASS: **{summary['annotation_pass_count']}/{len(FORMAL_SESSION_IDS)}**",
        f"- Replay complete: **{summary['replay_complete_count']}/{len(FORMAL_SESSION_IDS)}**",
        f"- PASS / PASS_WITH_WARNINGS / FAIL: **{counts['PASS']} / {counts['PASS_WITH_WARNINGS']} / {counts['FAIL']}**",
        f"- Actions applied: **{summary['actions_applied_total']}**",
        f"- SYNC_REQUIRED frames: **{summary['sync_required_frames_total']}**",
        f"- HOOK raw / qualified / used / support: **{summary['detector_totals']['hook']['raw_detected']} / "
        f"{summary['detector_totals']['hook']['qualified_detected']} / "
        f"{summary['detector_totals']['hook']['used_by_fusion']} / "
        f"{summary['detector_totals']['hook']['support_frames']}**",
        f"- PRESS candidate / present / sequence-ready / intents / support: "
        f"**{summary['detector_totals']['press']['panel_candidate_frames']} / "
        f"{summary['detector_totals']['press']['panel_present_frames']} / "
        f"{summary['detector_totals']['press']['sequence_ready_frames']} / "
        f"{summary['detector_totals']['press']['sequence_intent_count']} / "
        f"{summary['detector_totals']['press']['support_frames']}**",
        f"- GET raw / qualified / used / support: **{summary['detector_totals']['get']['raw_detected']} / "
        f"{summary['detector_totals']['get']['qualified_detected']} / "
        f"{summary['detector_totals']['get']['used_by_fusion']} / "
        f"{summary['detector_totals']['get']['support_frames']}**",
        f"- False action intents present: **{summary['false_action_intents_present']}**",
        f"- Unexplained Runtime transitions: **{len(summary['unexplained_runtime_transitions'])}**",
        f"- Trial excluded: `{summary['trial_excluded']}`",
        f"- Specialized warning impact: {summary['specialized_detector_warning_impact']}",
        f"- `ready_for_prompt_observer = {str(summary['ready_for_prompt_observer']).lower()}`",
        "",
        "## Detector warnings",
        "",
    ])
    warnings = [
        f"- `{item['session']}` {detector.upper()}: {warning}"
        for item in summary["sessions"]
        for detector, values in item["detector_warnings"].items()
        for warning in values
    ]
    lines.extend(warnings or ["- None"])
    lines.extend(["", "## SYNC_REQUIRED and Runtime issues", ""])
    issues = []
    for item in summary["sessions"]:
        if item["sync_required_frames"]:
            issues.append(
                f"- `{item['session']}` frame {item['first_sync_required_frame']}: "
                f"{item['first_sync_required_reason']}"
            )
        if any(item["false_action_intents"].values()):
            issues.append(f"- `{item['session']}` false action intents: {item['false_action_intents']}")
        issues.extend(
            f"- `{item['session']}` frame {transition['frame']}: "
            f"{transition['issue']} ({transition['from']} -> {transition['to']}, "
            f"{transition['reason']})"
            for transition in item["unexplained_runtime_transitions"]
        )
    lines.extend(issues or ["- None"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def run_batch(
    *,
    session_root: Path,
    config: Path,
    output_root: Path,
    work_root: Path,
    session_ids: Iterable[str] = FORMAL_SESSION_IDS,
) -> dict[str, Any]:
    session_ids = tuple(session_ids)
    if session_ids != FORMAL_SESSION_IDS:
        raise ValueError("Batch validation must contain exactly the seven formal sessions in fixed order")
    if (session_root / TRIAL_SESSION_ID / "prompt_ground_truth.yaml").exists():
        raise ValueError(f"Trial session must not have formal Prompt annotation: {TRIAL_SESSION_ID}")
    runner = V2ReplayRunner(config)
    summaries: list[dict[str, Any]] = []
    for session_id in session_ids:
        session_path = session_root / session_id
        annotation = validate_annotation(session_path)
        run = runner.run(
            session_path,
            mode="scripted_prompt",
            report_dir=work_root,
            action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
            flat_report=False,
        )
        summary = summarize_replay(session_id, run.rows, annotation)
        write_session_summary(summary, output_root)
        summaries.append(summary)
        print(
            f"{session_id}: {summary['result']} frames={summary['final_frame']} "
            f"state={summary['final_state']} sync={summary['sync_required_frames']}",
            flush=True,
        )
    cross_session = build_cross_session_summary(summaries)
    write_cross_session_summary(cross_session, output_root)
    return cross_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate seven formal scripted Prompt replay sessions.")
    parser.add_argument(
        "--session-root", type=Path,
        default=PROJECT_ROOT / "assets" / "replay" / "sessions",
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "fishing_v2.yaml")
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "reports" / "fishing_v2" / "scripted_replay_validation",
    )
    parser.add_argument(
        "--work-root", type=Path,
        default=PROJECT_ROOT / "tmp" / "scripted_replay_validation_frames",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_batch(
        session_root=args.session_root,
        config=args.config,
        output_root=args.output_root,
        work_root=args.work_root,
    )
    print(
        "ready_for_prompt_observer="
        f"{str(summary['ready_for_prompt_observer']).lower()}",
        flush=True,
    )
    return 0 if summary["result_counts"]["FAIL"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
