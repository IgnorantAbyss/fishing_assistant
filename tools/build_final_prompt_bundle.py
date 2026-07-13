"""Build the immutable all-session deployment bundle for Prototype PromptObserver."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.perception.prompt_bundle import build_final_prompt_bundle, load_prompt_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "fishing_v2.yaml")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets" / "prompt_observation_v1")
    parser.add_argument("--session-root", type=Path, default=PROJECT_ROOT / "assets" / "replay" / "sessions")
    parser.add_argument("--loso-summary", type=Path, default=PROJECT_ROOT / "reports" / "fishing_v2" / "prompt_observer_prototype_summary.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "prompt_observer" / "prototype_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = build_final_prompt_bundle(
        config_path=args.config,
        manifest_path=args.dataset / "manifest.csv",
        dataset_root=args.dataset,
        session_root=args.session_root,
        loso_summary_path=args.loso_summary,
        output_dir=args.output,
        feature_cache=args.dataset / "features_prototype_v1.npz",
    )
    bundle = load_prompt_bundle(output)
    print(f"bundle: {output}")
    print(f"version: {bundle.bundle_version}")
    print(f"prototypes: {len(bundle.model.prototypes)}")
    print(f"bundle_sha256: {bundle.bundle_sha256}")
    print(f"class_thresholds: {dict(bundle.model.class_thresholds)}")
    print(f"ambiguity_margin: {bundle.model.ambiguity_threshold}")
    print(f"idle_stability_frames: {bundle.model.idle_stability_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
