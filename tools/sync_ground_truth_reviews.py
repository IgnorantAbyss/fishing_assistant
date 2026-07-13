#!/usr/bin/env python3
"""Regenerate review CSVs from human-confirmed YAML ground truth."""

from __future__ import annotations

import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fishing_v2.data.hook_bar_ground_truth import load_hook_bar_ground_truth  # noqa: E402
from src.fishing_v2.data.press_sequence_ground_truth import load_press_sequence_ground_truth  # noqa: E402


HOOK_YAML = ROOT / "data" / "annotations" / "hook_bar_ground_truth.yaml"
PRESS_YAML = ROOT / "data" / "annotations" / "press_sequence_ground_truth.yaml"
SESSION_ROOT = ROOT / "assets" / "replay" / "sessions"
HOOK_CSV = ROOT / "reports" / "fishing_v2" / "hook_bar_candidate_review" / "review_items.csv"
PRESS_CSV = ROOT / "reports" / "fishing_v2" / "press_sequence_review" / "review_items.csv"
HOOK_SUMMARY = ROOT / "reports" / "fishing_v2" / "hook_bar_candidate_review" / "summary.md"


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sync_review_csvs(
    hook_yaml: Path = HOOK_YAML,
    press_yaml: Path = PRESS_YAML,
    hook_csv: Path = HOOK_CSV,
    press_csv: Path = PRESS_CSV,
    session_root: Path = SESSION_ROOT,
    hook_summary: Path | None = HOOK_SUMMARY,
) -> tuple[Path, Path]:
    hooks = load_hook_bar_ground_truth(hook_yaml, session_root=session_root)
    presses = load_press_sequence_ground_truth(press_yaml, session_root=session_root)
    _write(hook_csv, [{
        "session_id": item.session_id,
        "episode_index": item.episode_index,
        "global_hook_start": item.global_hook_start,
        "global_hook_end": item.global_hook_end,
        "proposed_start": item.visible_start,
        "proposed_end": item.visible_end,
        "first_clear_frame": item.first_clear_frame,
        "last_clear_frame": item.last_clear_frame,
        "confidence": item.confidence,
        "uncertain_frames": ";".join(map(str, item.uncertain_frames)),
        "human_confirmed_start": item.visible_start,
        "human_confirmed_end": item.visible_end,
        "review_status": "human_confirmed",
        "source_yaml": hook_yaml.as_posix(),
        "notes": item.notes,
    } for item in hooks])
    _write(press_csv, [{
        "session_id": item.session_id,
        "press_start": item.press_start,
        "press_end": item.press_end,
        "review_frame": item.press_start,
        "predicted_sequence": "",
        "manually_confirmed_sequence": "".join(item.sequence),
        "review_status": "human_confirmed_adjusted",
        "source_yaml": press_yaml.as_posix(),
        "notes": "YAML source of truth; prompt/panel-absent IGNORE tail excluded",
    } for item in presses])
    if hook_summary is not None:
        hook_summary.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Human-confirmed Hook Bar ranges", "",
            "This review artifact is regenerated from `data/annotations/hook_bar_ground_truth.yaml`.",
            "The YAML is the sole source of truth; this summary and its CSV never override it.", "",
            "- Status: **human_confirmed**.",
            f"- Formal sessions / episodes: **7 / {len(hooks)}**.",
            "- Original candidates came from raw-frame visual review without Hook detector output.",
            "- All visible and clear ranges below retain the user's manual confirmation.", "",
            "| Session | Episode | Global Hook | Visible | Clear | Status |", 
            "|---|---:|---:|---:|---:|---|",
        ]
        lines.extend(
            f"| {item.session_id} | {item.episode_index} | {item.global_hook_start}-{item.global_hook_end} | "
            f"{item.visible_start}-{item.visible_end} | {item.first_clear_frame}-{item.last_clear_frame} | human_confirmed |"
            for item in hooks
        )
        hook_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hook_csv, press_csv


def main() -> int:
    hook_csv, press_csv = sync_review_csvs()
    print(f"hook_review_csv={hook_csv}")
    print(f"press_review_csv={press_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
