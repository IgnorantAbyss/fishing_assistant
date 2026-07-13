"""Diagnose one saved live Prompt frame without changing model calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.data.prompt_observation_dataset import read_prompt_observation_manifest  # noqa: E402
from src.fishing_v2.evaluation.prompt_prototype_evaluation import load_or_build_features  # noqa: E402
from src.fishing_v2.perception.prompt_bundle import load_prompt_bundle  # noqa: E402
from src.fishing_v2.perception.prototype_prompt_observer import (  # noqa: E402
    PrototypePromptModel,
    extract_prompt_feature,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _preprocessing_images(crop: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 4)).apply(gray)
    background = cv2.GaussianBlur(clahe, (0, 0), sigmaX=7.0, sigmaY=7.0)
    positive = cv2.subtract(clahe, background)
    bright = np.where((gray >= 145) & (positive >= 8), positive, 0).astype(np.uint8)
    grad_x = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.clip(cv2.magnitude(grad_x, grad_y), 0, 255).astype(np.uint8)
    return {"normalized": clahe, "bright_mask": bright, "sobel": sobel}


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write diagnostic image: {path}")


def _image_stats(image: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "channel_count": int(image.shape[2]) if image.ndim == 3 else 1,
        "channel_order": "BGR" if image.ndim == 3 and image.shape[2] == 3 else "unsupported",
        "contiguous": bool(image.flags.c_contiguous),
        "strides": list(image.strides),
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean()),
    }


def _runtime_prompt_event(events_path: Path) -> dict[str, object] | None:
    if not events_path.is_file():
        return None
    for line in events_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("event_type") == "prompt_label_change":
            return dict(item.get("prompt_evidence", {}))
    return None


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path,
        default=PROJECT_ROOT / "reports" / "fishing_v2" / "live_detect_only"
        / "session_20260713_172154" / "screenshots" / "000035_sync_required.jpg",
    )
    parser.add_argument(
        "--bundle", type=Path,
        default=PROJECT_ROOT / "artifacts" / "prompt_observer" / "prototype_v1",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "reports" / "fishing_v2" / "live_prompt_diagnostics",
    )
    parser.add_argument(
        "--candidate-output", type=Path,
        help="Optional explicit output for the reviewed live ROI crop; no label is inferred.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise FileNotFoundError(args.image)
    loaded = load_prompt_bundle(args.bundle)
    x1, y1, x2, y2 = loaded.roi.pixel_bounds(frame.shape[1], frame.shape[0])
    crop = np.ascontiguousarray(frame[y1:y2, x1:x2])
    feature = extract_prompt_feature(crop)
    baseline_model = PrototypePromptModel(
        tuple(
            item for item in loaded.model.prototypes
            if item.source_session in loaded.model.training_sessions
        ),
        loaded.model.class_thresholds,
        loaded.model.ambiguity_threshold,
        loaded.model.training_sessions,
        loaded.model.calibration_sessions,
        loaded.model.idle_stability_frames,
    )
    prediction = baseline_model.predict_feature(feature)
    ranked = sorted(
        [
            {
                "rank": 0,
                "class": prototype.label,
                "prototype_id": prototype.prototype_id,
                "prototype_session": prototype.source_session,
                "prototype_frame": prototype.source_frame,
                "similarity": float(feature @ prototype.feature),
            }
            for prototype in baseline_model.prototypes
        ],
        key=lambda item: (-float(item["similarity"]), str(item["prototype_id"])),
    )
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index

    args.output.mkdir(parents=True, exist_ok=True)
    _write(args.output / "live_raw_roi.png", crop)
    for name, image in _preprocessing_images(crop).items():
        _write(args.output / f"live_{name}.png", image)
    if args.candidate_output:
        _write(args.candidate_output, crop)

    nearest: dict[str, dict[str, object]] = {}
    for short_name, label in (
        ("idle", "IDLE_CAST"),
        ("hook", "HOOK_INSTRUCTION"),
        ("press", "PRESS_INSTRUCTION"),
    ):
        match = next(item for item in ranked if item["class"] == label)
        nearest[short_name] = match
        source = (
            PROJECT_ROOT / "assets" / "replay" / "sessions"
            / str(match["prototype_session"]) / "frames"
            / f"{int(match['prototype_frame']):06d}.jpg"
        )
        source_frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if source_frame is None:
            raise FileNotFoundError(source)
        source_crop = source_frame[y1:y2, x1:x2]
        _write(args.output / f"nearest_{short_name}_raw.png", source_crop)
        _write(
            args.output / f"nearest_{short_name}_normalized.png",
            _preprocessing_images(source_crop)["normalized"],
        )

    dataset = PROJECT_ROOT / "datasets" / "prompt_observation_v1"
    rows = read_prompt_observation_manifest(dataset / "manifest.csv")
    features = load_or_build_features(
        rows, dataset, cache_path=dataset / "features_prototype_v1.npz"
    )
    idle_indices = [index for index, row in enumerate(rows) if row["label"] == "IDLE_CAST"]
    idle_coverage = sorted(
        [
            {
                "session": rows[index]["session_id"],
                "frame": int(rows[index]["frame_index"]),
                "similarity": float(feature @ features[index]),
            }
            for index in idle_indices
        ],
        key=lambda item: -item["similarity"],
    )
    events_path = args.image.parents[1] / "events.jsonl"
    runtime = _runtime_prompt_event(events_path)
    diagnostic = {
        "source_image": _display_path(args.image),
        "source_image_sha256": _sha256(args.image.read_bytes()),
        "input_contract": _image_stats(frame),
        "roi": [x1, y1, x2, y2],
        "roi_stats": _image_stats(crop),
        "feature_sha256": _sha256(feature.tobytes()),
        "saved_screenshot_prediction": prediction.__dict__,
        "corrected_bundle_prediction": loaded.model.predict_feature(feature).__dict__,
        "runtime_recorded_prediction": runtime,
        "runtime_offline_categorical_match": bool(
            runtime
            and runtime.get("predicted_label") == prediction.predicted_label
            and runtime.get("prototype_id") == prediction.prototype_id
            and runtime.get("second_label") == prediction.second_label
        ),
        "top_10_prototype_similarities": ranked[:10],
        "class_scores": baseline_model.raw_scores(feature)[0],
        "idle_prototype_similarities": [item for item in ranked if item["class"] == "IDLE_CAST"],
        "nearest_by_class": nearest,
        "all_annotated_idle_frame_count": len(idle_coverage),
        "nearest_annotated_idle_frames": idle_coverage[:10],
        "diagnosis": "new_live_variant_not_covered",
    }
    (args.output / "baseline_diagnostics.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
