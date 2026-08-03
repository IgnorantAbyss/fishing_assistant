"""Offline PRESS V3 analysis.  This tool never constructs an ActionSink."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_roi_config  # noqa: E402
from src.detectors.press_background_subtraction_v3 import (  # noqa: E402
    PressBackgroundSubtractionDetectorV3,
    serializable_v3_result,
)


def _images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        item for item in source.rglob("*")
        if item.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def _press_roi(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= 900 and height <= 400:
        return image
    x1, y1, x2, y2 = load_roi_config().pixel_roi(
        "press_sequence", width, height
    )
    return image[y1:y2, x1:x2]


def _background_patch(lab_values: object) -> np.ndarray:
    values = np.asarray(lab_values, dtype=np.uint8).reshape(1, 1, 3)
    bgr = cv2.cvtColor(values, cv2.COLOR_LAB2BGR)[0, 0]
    return np.full((80, 240, 3), bgr, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = _images(args.input)
    if not paths:
        parser.error(f"no images found: {args.input}")
    detector = PressBackgroundSubtractionDetectorV3()
    rows: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        panel = _press_roi(image)
        result = detector.detect(panel)
        target = args.output / f"{index:03d}_{path.stem}"
        target.mkdir(parents=True, exist_ok=True)
        located = panel.copy()
        locator = result.get("locator", {})
        bbox = locator.get("key_strip_bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(located, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
        for slot_index, slot_bbox in enumerate(locator.get("slot_bboxes", ())):
            x1, y1, x2, y2 = map(int, slot_bbox)
            cv2.rectangle(located, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 0), 1)
            cv2.putText(located, str(slot_index), (x1 + 2, y1 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        cv2.imwrite(str(target / "key_strip_location.png"), located)
        strip = result.get("key_strip_crop")
        if isinstance(strip, np.ndarray):
            cv2.imwrite(str(target / "key_strip.png"), strip)
        masks = [
            slot.get("binary_mask") for slot in result.get("slots", ())
            if isinstance(slot.get("binary_mask"), np.ndarray)
        ]
        if masks:
            cv2.imwrite(str(target / "binary_foreground.png"), np.hstack(masks))
        background = result.get("background_model", {})
        if background.get("global_background_lab") is not None:
            cv2.imwrite(
                str(target / "background_lab_patch.png"),
                _background_patch(background["global_background_lab"]),
            )
        payload = serializable_v3_result(result)
        payload["source"] = str(path)
        (target / "classification.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append({
            "source": str(path),
            "located": bool(result.get("panel_present")),
            "sequence": "".join(result.get("sequence_candidate", ())),
            "occupied_count": int(result.get("occupied_slot_count", 0)),
            "complete": bool(result.get("frame_complete", False)),
            "background_stability": background.get("background_stability"),
            "latency_ms": result.get("processing_latency_ms"),
            "output": str(target),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PRESS Background Subtraction V3",
        "",
        "| Source | Located | Sequence | Occupied | Complete | Background stability | Latency ms |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['source']}` | {row['located']} | `{row['sequence']}` | "
            f"{row['occupied_count']} | {row['complete']} | "
            f"{row['background_stability']} | {row['latency_ms']} |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"analyzed={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
