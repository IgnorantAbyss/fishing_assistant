"""Report offline fishing-state predictions for supplied reference screenshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.state_detector import StateDetector, expected_reference_images  # noqa: E402
from src.debug_overlay import save_debug_overlay  # noqa: E402
from src.config_loader import DEFAULT_ROI_CONFIG_PATH, DEFAULT_THRESHOLDS_CONFIG_PATH  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify fishing UI screenshots without input automation.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--image", type=Path, help="Screenshot to classify")
    selection.add_argument("--all", action="store_true", help="Classify all bundled reference screenshots")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=PROJECT_ROOT / "assets" / "reference",
        help="Directory that contains labelled reference PNGs",
    )
    parser.add_argument("--json", action="store_true", help="Print newline-delimited JSON instead of readable text")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "static_reports",
        help="Directory for annotated debug screenshots",
    )
    parser.add_argument(
        "--roi-config",
        type=Path,
        default=DEFAULT_ROI_CONFIG_PATH,
        help="Normalized ROI YAML; missing files use built-in defaults",
    )
    parser.add_argument(
        "--thresholds-config",
        type=Path,
        default=DEFAULT_THRESHOLDS_CONFIG_PATH,
        help="Detection threshold YAML; missing files use built-in defaults",
    )
    return parser.parse_args()


def emit(path: Path, result: dict[str, object], debug_image_path: Path, as_json: bool) -> None:
    payload = {"image": str(path), **result, "debug_image_path": str(debug_image_path)}
    if as_json:
        print(json.dumps(payload, ensure_ascii=False))
        return
    print(f"image: {path}")
    print(f"state: {payload['state']}")
    print(f"confidence: {payload['confidence']:.2%}")
    print(f"matched_features: {', '.join(payload['matched_features'])}")
    print(f"debug image path: {payload['debug_image_path']}")
    print(f"best reference: {payload['debug']['best_reference']}")
    print()


def main() -> int:
    args = parse_args()
    detector = StateDetector(
        args.reference_dir,
        roi_config_path=args.roi_config,
        thresholds_config_path=args.thresholds_config,
    )
    images = [args.image] if args.image else [path for path, _ in expected_reference_images(args.reference_dir)]
    for path in images:
        result = detector.detect(path)
        debug_image_path = save_debug_overlay(path, result, args.debug_dir, roi_config=detector.roi_config)
        emit(path, result.to_dict(), debug_image_path, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
