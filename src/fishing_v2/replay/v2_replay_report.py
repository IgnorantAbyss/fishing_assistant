from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence


REPLAY_FIELDS = (
    "frame_index", "global_ground_truth", "prompt_observation",
    "hook_observation", "press_observation", "get_observation",
    "previous_runtime_state", "state_evidence", "next_runtime_state",
    "action_intent", "safety_decision", "transition_reason",
)


def write_v2_replay_report(
    report_dir: str | Path,
    session_id: str,
    rows: Sequence[dict[str, Any]],
    *,
    mode: str,
) -> tuple[Path, Path]:
    root = Path(report_dir) / session_id
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "v2_replay_frames.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REPLAY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                name: json.dumps(row[name], ensure_ascii=False, sort_keys=True)
                if isinstance(row[name], (dict, list)) else row[name]
                for name in REPLAY_FIELDS
            })
    transitions = [
        row for row in rows if row["previous_runtime_state"] != row["next_runtime_state"]
    ]
    lines = [
        "# Hybrid Runtime v2 Replay",
        "",
        f"- Session: `{session_id}`",
        f"- Mode: `{mode}`",
        f"- Frames: {len(rows)}",
        "- Architecture validation only; this is not Prompt model evaluation.",
        "- Action emission: disabled",
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
    return csv_path, md_path
