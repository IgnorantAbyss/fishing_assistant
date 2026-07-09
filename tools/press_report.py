"""Print an offline WASD press-sequence parsing report for one image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors.press_detector import detect_press_sequence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse one WASD sequence from an image file.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = detect_press_sequence(args.image)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        for key in ("detected", "confidence", "sequence_text", "key_boxes", "matched_features"):
            print(f"{key}: {result[key]}")
        print(f"debug_image_path: {result['debug']['debug_image_path']}")
    return 0 if result["detected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
