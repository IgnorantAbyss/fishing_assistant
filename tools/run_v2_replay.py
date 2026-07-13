"""Run Hybrid Runtime v2 over replay frames with no real action emission."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.domain.runtime_state import RuntimeState  # noqa: E402
from src.fishing_v2.data.prompt_observation_dataset import (  # noqa: E402
    FORMAL_SESSION_IDS,
    read_prompt_observation_manifest,
)
from src.fishing_v2.data.prompt_roi import load_approved_prompt_roi  # noqa: E402
from src.fishing_v2.evaluation.prompt_prototype_evaluation import load_or_build_features  # noqa: E402
from src.fishing_v2.perception.prototype_prompt_observer import (  # noqa: E402
    PrototypePromptObserver,
    fit_prototype_model,
)
from src.fishing_v2.replay.v2_replay_runner import V2ReplayRunner  # noqa: E402
from src.fishing_v2.runtime.runtime_controller import ActionExecutionMode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Hybrid Runtime v2 architecture on saved replay frames.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--session", help="Session id or directory")
    selection.add_argument("--all", action="store_true")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--mode", choices=("scripted_prompt", "predicted_prompt", "no_prompt"))
    prompt.add_argument("--use-scripted-prompt-observations", action="store_true")
    prompt.add_argument("--no-prompt-observer", action="store_true")
    parser.add_argument(
        "--action-mode",
        choices=[item.value for item in ActionExecutionMode],
        default=ActionExecutionMode.STANDARD.value,
    )
    parser.add_argument("--start-state", choices=[state.value for state in RuntimeState if state not in {RuntimeState.SYNCING, RuntimeState.SYNC_REQUIRED}])
    parser.add_argument("--session-root", type=Path, default=PROJECT_ROOT / "assets" / "replay" / "sessions")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "fishing_v2.yaml")
    parser.add_argument("--prompt-observer", choices=("prototype_v1",))
    parser.add_argument("--prompt-dataset", type=Path, default=PROJECT_ROOT / "datasets" / "prompt_observation_v1")
    parser.add_argument("--report-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = args.mode or ("scripted_prompt" if args.use_scripted_prompt_observations else "no_prompt")
    action_mode = ActionExecutionMode(args.action_mode)
    report_dir = args.report_dir or (
        PROJECT_ROOT / "reports" / "fishing_v2" / "predicted_prompt_replay"
        if mode == "predicted_prompt"
        else PROJECT_ROOT / "reports" / "fishing_v2" / "pilot_replay"
        if action_mode == ActionExecutionMode.RECORDED_OBSERVATION
        else PROJECT_ROOT / "reports" / "fishing_v2" / "replay"
    )
    if args.all:
        sessions = sorted(path for path in args.session_root.glob("session_*") if (path / "ground_truth.yaml").is_file())
    else:
        supplied = Path(args.session)
        sessions = [supplied if supplied.is_dir() else args.session_root / args.session]
    runner = V2ReplayRunner(args.config)
    start_state = RuntimeState(args.start_state) if args.start_state else None
    rows = features = roi = None
    if mode == "predicted_prompt":
        if args.prompt_observer != "prototype_v1":
            raise ValueError("predicted_prompt requires --prompt-observer prototype_v1")
        rows = read_prompt_observation_manifest(args.prompt_dataset / "manifest.csv")
        features = load_or_build_features(
            rows,
            args.prompt_dataset,
            cache_path=args.prompt_dataset / "features_prototype_v1.npz",
        )
        roi = load_approved_prompt_roi(args.config)
        if roi is None:
            raise ValueError("predicted_prompt requires the approved Prompt ROI")
    for session in sessions:
        observer = None
        if mode == "predicted_prompt":
            if session.name not in FORMAL_SESSION_IDS:
                raise ValueError(f"No isolated prototype fold exists for {session.name}")
            training_sessions = tuple(item for item in FORMAL_SESSION_IDS if item != session.name)
            observer = PrototypePromptObserver(
                fit_prototype_model(rows, features, training_sessions), roi
            )
        run = runner.run(
            session,
            mode=mode,
            report_dir=report_dir,
            start_state=start_state,
            action_mode=action_mode,
            flat_report=(action_mode == ActionExecutionMode.RECORDED_OBSERVATION and not args.all),
            prompt_observer=observer,
        )
        print(f"{run.session_id}: frames={len(run.rows)} report={run.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
