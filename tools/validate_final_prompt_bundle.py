"""Deployment regression for the immutable all-session Prompt bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.data.prompt_annotation import load_prompt_ground_truth  # noqa: E402
from src.fishing_v2.data.prompt_observation_dataset import FORMAL_SESSION_IDS  # noqa: E402
from src.fishing_v2.perception.prompt_bundle import load_prompt_bundle  # noqa: E402
from src.fishing_v2.perception.prototype_prompt_observer import PrototypePromptObserver  # noqa: E402
from src.fishing_v2.replay.v2_replay_runner import V2ReplayRunner  # noqa: E402
from src.fishing_v2.runtime.runtime_controller import ActionExecutionMode  # noqa: E402
from tools.validate_predicted_prompt_replays import build_summary, compare_runs  # noqa: E402


def write_report(summary: dict, report_root: Path) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / "final_prompt_bundle_validation.json"
    md_path = report_root / "final_prompt_bundle_validation.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Final Prompt Bundle Deployment Validation",
        "",
        f"- Bundle version: `{summary['bundle_version']}`",
        f"- Bundle SHA-256: `{summary['bundle_sha256']}`",
        f"- Prototypes: **{summary['prototype_count']}**",
        "- Threshold aggregation: per-class LOSO median; ambiguity LOSO median; IDLE stability LOSO maximum.",
        "- This all-session replay is deployment regression only; it does not replace LOSO generalization results.",
        "",
        "| session | final scripted | final bundle | state agreement | false intents | missed events | sync | result |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for item in summary["sessions"]:
        lines.append(
            f"| {item['session']} | {item['scripted_final_state']} | {item['predicted_final_state']} | "
            f"{item['runtime_state_frame_agreement']:.4f} | {item['false_intent_count']} | "
            f"{len(item['missed_expected_intent_types'])} | {item['sync_required_frames']} | {item['result']} |"
        )
    lines.extend([
        "",
        f"- Complete: **{summary['replay_complete_count']}/7**",
        f"- False intents: **{summary['false_intents']}**",
        f"- Missed expected events: **{summary['missed_expected_intents']}**",
        f"- SYNC_REQUIRED frames: **{summary['sync_required_frames']}**",
        f"- Actions applied: **{summary['actions_applied']}**",
        f"- Deployment regression passed: **{str(summary['deployment_regression_passed']).lower()}**",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=PROJECT_ROOT / "artifacts" / "prompt_observer" / "prototype_v1")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "fishing_v2.yaml")
    parser.add_argument("--session-root", type=Path, default=PROJECT_ROOT / "assets" / "replay" / "sessions")
    parser.add_argument("--work-root", type=Path, default=PROJECT_ROOT / "tmp" / "final_prompt_bundle_replay")
    parser.add_argument("--report-root", type=Path, default=PROJECT_ROOT / "reports" / "fishing_v2")
    parser.add_argument("--prompt-summary", type=Path, default=PROJECT_ROOT / "reports" / "fishing_v2" / "prompt_observer_prototype_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loaded = load_prompt_bundle(args.bundle)
    prompt_summary = json.loads(args.prompt_summary.read_text(encoding="utf-8"))
    comparisons = []
    for session_id in FORMAL_SESSION_IDS:
        session_path = args.session_root / session_id
        print(f"deployment regression {session_id}: scripted", flush=True)
        scripted = V2ReplayRunner(args.config).run(
            session_path,
            mode="scripted_prompt",
            report_dir=args.work_root / "scripted",
            action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        )
        print(f"deployment regression {session_id}: final bundle", flush=True)
        predicted = V2ReplayRunner(args.config).run(
            session_path,
            mode="predicted_prompt",
            report_dir=args.work_root / "predicted",
            action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
            prompt_observer=PrototypePromptObserver(loaded.model, loaded.roi),
        )
        prompt_labels = load_prompt_ground_truth(
            session_path / "prompt_ground_truth.yaml", len(predicted.rows)
        )
        comparisons.append(compare_runs(
            session_id,
            scripted.rows,
            predicted.rows,
            prompt_labels,
            prompt_summary["events"],
        ))
    operational = build_summary(comparisons, prompt_summary)
    deployment_passed = all((
        operational["replay_complete_count"] == 7,
        operational["false_intents"] == 0,
        operational["missed_expected_intents"] == 0,
        operational["sync_required_frames"] == 0,
        operational["actions_applied"] == 0,
        operational["fail_frames"] == 0,
    ))
    summary = {
        "bundle_version": loaded.bundle_version,
        "bundle_sha256": loaded.bundle_sha256,
        "prototype_count": len(loaded.model.prototypes),
        "class_thresholds": dict(loaded.model.class_thresholds),
        "ambiguity_margin": loaded.model.ambiguity_threshold,
        "idle_stability_frames": loaded.model.idle_stability_frames,
        "deployment_regression_only": True,
        "loso_generalization_report": "reports/fishing_v2/prompt_observer_prototype_summary.json",
        **operational,
        "deployment_regression_passed": deployment_passed,
    }
    write_report(summary, args.report_root)
    print(f"complete: {summary['replay_complete_count']}/7")
    print(f"false_intents: {summary['false_intents']}")
    print(f"missed_expected_intents: {summary['missed_expected_intents']}")
    print(f"sync_required_frames: {summary['sync_required_frames']}")
    print(f"actions_applied: {summary['actions_applied']}")
    print(f"deployment_regression_passed: {deployment_passed}")
    return 0 if deployment_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
