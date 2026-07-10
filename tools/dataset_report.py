"""Generate dataset capacity, balance, split, and warning reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DEFAULT_DATASET_CONFIG_PATH, load_dataset_config  # noqa: E402
from src.dataset.validation import SessionSplit, write_session_split  # noqa: E402
from src.replay_session import DEFAULT_SESSION_ROOT  # noqa: E402
from tools.build_training_dataset import _write_reports, build_dataset, print_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a report for the planned or existing ROI dataset.")
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG_PATH)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = load_dataset_config(args.dataset_config).dataset.output_dir
    training_report = output_dir / "training_dataset_report.json"
    if training_report.is_file():
        report = json.loads(training_report.read_text(encoding="utf-8"))
        print(f"training_report: {training_report}")
        print(f"formal_sessions: {report['formal_session_count']}")
        print(f"trial_sessions: {report['trial_sessions']}")
        print(f"split: {report['split']}")
        print(f"prompt_train_before: {report['prompt_train_before']}")
        print(f"prompt_train_after: {report['prompt_train_after']}")
        print(f"special_train_before: {report['special_train_before']}")
        print(f"special_train_after: {report['special_train_after']}")
        print(f"validation_prompt: {report['validation_prompt_labels']}")
        print(f"test_prompt: {report['test_prompt_labels']}")
        print(f"validation_special: {report['validation_special_labels']}")
        print(f"test_special: {report['test_special_labels']}")
        print(f"session_leakage: {report['session_leakage']}")
        for warning in report["warnings"]:
            print(f"WARNING: {warning}")
        return 0
    result = build_dataset(
        config_path=args.dataset_config,
        session_root=args.session_root,
        dry_run=True,
        write_reports=False,
        strict_split=False,
    )
    _write_reports(result.output_dir, result.report)
    session_ids = [session["session_id"] for session in result.report["sessions"]]
    split_path = result.output_dir / "session_split.yaml"
    if not split_path.exists():
        write_session_split(split_path, SessionSplit((), (), (), tuple(sorted(session_ids))))
    print_summary(result)
    print(f"json_report: {result.output_dir / 'dataset_report.json'}")
    print(f"markdown_report: {result.output_dir / 'dataset_report.md'}")
    for warning in result.report["warnings"]:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
