"""Print an offline hook-bar detection report for one existing image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detectors.hook_detector import detect_hook_bar  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one hook timing bar from an image file.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = detect_hook_bar(args.image)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        for key in ("detected", "confidence", "fill_ratio", "divider_ratio", "matched_features"):
            print(f"{key}: {result[key]}")
        print(f"debug_image_path: {result['debug']['debug_image_path']}")
    return 0 if result["detected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
