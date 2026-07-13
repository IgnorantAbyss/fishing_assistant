"""Evaluate prototype_v1 with seven-fold leave-one-session-out isolation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.evaluation.prompt_prototype_evaluation import (  # noqa: E402
    evaluate_loso,
    write_prompt_observer_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "prompt_observation_v1")
    parser.add_argument("--session-root", type=Path, default=PROJECT_ROOT / "assets" / "replay" / "sessions")
    parser.add_argument("--report-root", type=Path, default=PROJECT_ROOT / "reports" / "fishing_v2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, _, _ = evaluate_loso(
        manifest_path=args.dataset / "manifest.csv",
        dataset_root=args.dataset,
        session_root=args.session_root,
        feature_cache=args.dataset / "features_prototype_v1.npz",
    )
    write_prompt_observer_reports(
        summary,
        args.report_root / "prompt_observer_prototype_summary.json",
        args.report_root / "prompt_observer_prototype_summary.md",
    )
    print(f"macro_f1: {summary['macro_f1']:.6f}")
    print(f"ignore_rejection_rate: {summary['ignore']['unknown_rejection_rate']:.6f}")
    print(f"report: {args.report_root / 'prompt_observer_prototype_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
