"""Create ROI-annotated copies of existing screenshots without modifying them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import DEFAULT_ROI_CONFIG_PATH, ROI_NAMES, load_roi_config, normalized_to_pixel_roi  # noqa: E402
from src.state_detector import expected_reference_images  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw configured normalized ROIs on existing screenshots.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--image", type=Path, help="One image to preview")
    selection.add_argument("--all", action="store_true", help="Preview every bundled reference image")
    parser.add_argument("--reference-dir", type=Path, default=PROJECT_ROOT / "assets" / "reference")
    parser.add_argument("--roi-config", type=Path, default=DEFAULT_ROI_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "logs" / "roi_preview")
    return parser.parse_args()


def save_preview(image_path: Path, output_dir: Path, *, roi_config_path: Path) -> Path:
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Could not read preview image: {image_path}")
    config = load_roi_config(roi_config_path)
    height, width = frame.shape[:2]
    colours = ((40, 220, 40), (40, 180, 255), (255, 120, 40), (220, 60, 220), (60, 220, 220), (255, 255, 255))
    for index, name in enumerate(ROI_NAMES):
        if name not in config.rois:
            continue
        left, top, right, bottom = normalized_to_pixel_roi(config.rois[name], width, height)
        colour = colours[index % len(colours)]
        cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
        cv2.putText(frame, name, (left, max(20, top - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{image_path.stem}_roi_preview.png"
    if not cv2.imwrite(str(destination), frame):
        raise OSError(f"Could not write ROI preview: {destination}")
    return destination.resolve()


def main() -> int:
    args = parse_args()
    images = [args.image] if args.image else [path for path, _ in expected_reference_images(args.reference_dir)]
    for image_path in images:
        print(save_preview(image_path, args.output_dir, roi_config_path=args.roi_config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
