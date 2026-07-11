from __future__ import annotations

import csv
from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence


REPLAY_FIELDS = (
    "frame_index", "global_ground_truth", "prompt_observation", "action_mode",
    "detector_activation_mode", "raw_detected", "qualified_detected",
    "qualification_reason", "used_by_fusion", "diagnostic_only",
    "press_panel_candidate", "press_panel_present_raw", "press_panel_present_qualified",
    "press_panel_qualification_reason", "press_key_box_count", "press_stable_key_box_count",
    "press_sequence_candidate", "press_sequence_ready", "press_sequence_confidence",
    "press_sequence_qualification_reason", "press_used_by_fusion",
    "hook_observation", "press_observation", "get_observation",
    "qualified_hook_observation", "qualified_press_observation", "qualified_get_observation",
    "previous_runtime_state", "state_evidence", "next_runtime_state",
    "proposed_intent", "action_intent", "action_applied", "safety_decision",
    "visual_acknowledgement", "transition_reason",
)


def write_v2_replay_report(
    report_dir: str | Path,
    session_id: str,
    rows: Sequence[dict[str, Any]],
    *,
    mode: str,
    action_mode: str = "standard",
    flat_output: bool = False,
) -> tuple[Path, Path]:
    root = Path(report_dir) if flat_output else Path(report_dir) / session_id
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "v2_replay_frames.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REPLAY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: json.dumps(row.get(name), ensure_ascii=False, sort_keys=True)
                if isinstance(row.get(name), (dict, list)) else row.get(name)
                for name in REPLAY_FIELDS
            })
    transitions = [
        row for row in rows if row["previous_runtime_state"] != row["next_runtime_state"]
    ]
    detector_counts = {
        detector: {
            "raw_detected": sum(bool(row["raw_detected"][detector]) for row in rows),
            "qualified_detected": sum(bool(row["qualified_detected"][detector]) for row in rows),
            "used_by_fusion": sum(bool(row["used_by_fusion"][detector]) for row in rows),
        }
        for detector in ("hook", "press", "get")
    }
    proposed = Counter(row["proposed_intent"] for row in rows if row["proposed_intent"] != "NONE")
    applied_count = sum(bool(row["action_applied"]) for row in rows)
    sync_required_frames = sum(row["next_runtime_state"] == "SYNC_REQUIRED" for row in rows)
    press_frames = [row for row in rows if row["global_ground_truth"] == "PRESS"]
    get_frames = [row for row in rows if row["global_ground_truth"] == "GET"]
    summary = {
        "session": session_id,
        "mode": mode,
        "action_mode": action_mode,
        "frames": len(rows),
        "final_frame": rows[-1]["frame_index"] if rows else None,
        "final_state": rows[-1]["next_runtime_state"] if rows else None,
        "proposed_intents": dict(proposed),
        "actions_applied": applied_count,
        "sync_required_frames": sync_required_frames,
        "detectors": detector_counts,
        "global_press": {
            "frames": len(press_frames),
            "raw_detected": sum(bool(row["raw_detected"]["press"]) for row in press_frames),
            "qualified_detected": sum(bool(row["qualified_detected"]["press"]) for row in press_frames),
            "panel_present": sum(bool(row.get("press_panel_present_raw")) for row in press_frames),
            "sequence_ready": sum(bool(row.get("press_sequence_ready")) for row in press_frames),
            "sequence_intents": sum(row["proposed_intent"] == "PRESS_SEQUENCE" for row in press_frames),
        },
        "global_get": {
            "frames": len(get_frames),
            "raw_detected": sum(bool(row["raw_detected"]["get"]) for row in get_frames),
            "qualified_detected": sum(bool(row["qualified_detected"]["get"]) for row in get_frames),
        },
        "complete_replay_processed": bool(rows) and rows[-1]["frame_index"] == len(rows),
    }
    lines = [
        "# Hybrid Runtime v2 Replay",
        "",
        f"- Session: `{session_id}`",
        f"- Mode: `{mode}`",
        f"- Action mode: `{action_mode}`",
        f"- Frames: {len(rows)}",
        "- Architecture validation only; this is not Prompt model evaluation.",
        "- Proposed intents are reported separately from action_applied.",
        "- recorded_observation validates recorded perception/Fusion/state flow; it is not real action execution.",
        f"- Final frame/state: {summary['final_frame']} / `{summary['final_state']}`",
        f"- Proposed intents: `{summary['proposed_intents']}`",
        f"- Actions applied: **{applied_count}**",
        f"- SYNC_REQUIRED frames: **{sync_required_frames}**",
        f"- Detector raw/qualified counts: `{detector_counts}`",
        f"- Global PRESS raw/qualified: **{summary['global_press']['raw_detected']}/{summary['global_press']['qualified_detected']}** of {len(press_frames)}",
        f"- Global PRESS panel present: **{summary['global_press']['panel_present']}/{len(press_frames)}**",
        f"- Global PRESS sequence ready frames/intents: **{summary['global_press']['sequence_ready']}/{summary['global_press']['sequence_intents']}**",
        f"- Global GET raw/qualified: **{summary['global_get']['raw_detected']}/{summary['global_get']['qualified_detected']}** of {len(get_frames)}",
        "",
        "## Runtime transitions",
        "",
    ]
    lines.extend(
        f"- frame {row['frame_index']}: {row['previous_runtime_state']} -> "
        f"{row['next_runtime_state']} ({row['transition_reason']})"
        for row in transitions
    )
    if not transitions:
        lines.append("- None")
    md_path = root / "v2_replay_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if action_mode == "recorded_observation":
        small_root = Path(report_dir).parent
        small_root.mkdir(parents=True, exist_ok=True)
        (small_root / "pilot_replay_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (small_root / "pilot_replay_summary.md").write_text(
            "\n".join(lines[:18]) + "\n",
            encoding="utf-8",
        )
    return csv_path, md_path
