"""Interactively calibrate normalized UI regions from a static screenshot.

This tool opens only an image explicitly passed with --image.  It never
captures the screen and never sends keyboard input to another application.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import (  # noqa: E402
    DEFAULT_ROI_CONFIG_PATH,
    ROI_NAMES,
    ROIConfig,
    load_roi_config,
    save_roi_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate normalized fishing UI regions from a static image.")
    parser.add_argument("--image", required=True, type=Path, help="Existing screenshot used for mouse selection")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--roi", choices=ROI_NAMES, help="One ROI to replace")
    selection.add_argument("--all", action="store_true", help="Select every ROI in the standard order")
    parser.add_argument(
        "--roi-config",
        type=Path,
        default=DEFAULT_ROI_CONFIG_PATH,
        help="ROI YAML file to update",
    )
    return parser.parse_args()


def select_normalized_roi(image, roi_name: str) -> tuple[float, float, float, float]:
    """Open OpenCV's mouse selector and turn its pixel rectangle into fractions."""
    height, width = image.shape[:2]
    window_name = f"Select ROI: {roi_name} (Enter/Space confirm, C cancel)"
    x, y, roi_width, roi_height = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError(f"Selection for '{roi_name}' was cancelled or empty; no configuration was changed")
    return (
        round(x / width, 6),
        round(y / height, 6),
        round((x + roi_width) / width, 6),
        round((y + roi_height) / height, 6),
    )


def main() -> int:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read calibration image: {args.image}")

    current_config = load_roi_config(args.roi_config)
    rois = dict(current_config.rois)
    targets = ROI_NAMES if args.all else (args.roi,)
    try:
        for roi_name in targets:
            rois[roi_name] = select_normalized_roi(image, roi_name)
    finally:
        cv2.destroyAllWindows()

    save_roi_config(
        args.roi_config,
        ROIConfig(
            screen_reference=current_config.screen_reference,
            rois=rois,
            source=args.roi_config,
        ),
    )
    print(f"Updated {args.roi_config}")
    for roi_name in targets:
        x1, y1, x2, y2 = rois[roi_name]
        print(f"{roi_name}: x={x1:.6f}..{x2:.6f}, y={y1:.6f}..{y2:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
