"""Pure analysis helpers for PromptClassifier v1 failure diagnostics."""

from __future__ import annotations

import hashlib
import heapq
import itertools
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


BOUNDARY_BUCKETS = ("0-3", "4-5", "6-10", ">10")
NONE_STATES = ("HOOK", "PRESS", "GET")
NEAR_DUPLICATE_LABEL_PAIRS = {
    frozenset(("IDLE", "READY")),
    frozenset(("READY", "NONE")),
    frozenset(("WAITING", "NONE")),
}


def uniform_sample(items: Sequence[Any], maximum: int) -> list[Any]:
    """Select across the full ordered sequence, including both endpoints."""
    if maximum <= 0 or not items:
        return []
    if len(items) <= maximum:
        return list(items)
    indices = np.linspace(0, len(items) - 1, maximum).round().astype(int)
    return [items[int(index)] for index in indices]


def group_confusion_errors(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["label"] != record["prediction"]:
            grouped[f"{record['label']}_to_{record['prediction']}"] .append(record)
    return dict(grouped)


def none_breakdown(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {state: [] for state in NONE_STATES}
    for record in records:
        if record.get("label") == "NONE" and record.get("original_state") in grouped:
            grouped[str(record["original_state"])].append(record)
    result: dict[str, dict[str, Any]] = {}
    for state, items in grouped.items():
        predictions = Counter(str(item["prediction"]) for item in items)
        confidences = np.asarray([float(item["confidence"]) for item in items], dtype=float)
        errors = [item for item in items if item["prediction"] != "NONE"]
        result[state] = {
            "total_samples": len(items),
            "predicted_NONE": predictions.get("NONE", 0),
            "predicted_READY": predictions.get("READY", 0),
            "predicted_WAITING": predictions.get("WAITING", 0),
            "predicted_IDLE": predictions.get("IDLE", 0),
            "predicted_UNKNOWN": predictions.get("UNKNOWN", 0),
            "mean_confidence": float(confidences.mean()) if confidences.size else 0.0,
            "median_confidence": float(np.median(confidences)) if confidences.size else 0.0,
            "boundary_errors": sum(bool(item.get("is_boundary")) for item in errors),
            "non_boundary_errors": sum(not bool(item.get("is_boundary")) for item in errors),
        }
    return result


def transition_frames(segments: Sequence[dict[str, Any]]) -> list[int]:
    ordered = sorted(segments, key=lambda item: int(item["start"]))
    return [int(segment["start"]) for segment in ordered[1:]]


def boundary_distance(frame_index: int, transitions: Sequence[int]) -> int:
    if not transitions:
        return 10**9
    return min(abs(int(frame_index) - int(point)) for point in transitions)


def boundary_bucket(distance: int) -> str:
    if distance <= 3:
        return "0-3"
    if distance <= 5:
        return "4-5"
    if distance <= 10:
        return "6-10"
    return ">10"


def boundary_analysis(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    categories = {
        "IDLE_to_READY": lambda item: item["label"] == "IDLE" and item["prediction"] == "READY",
        "NONE_to_READY": lambda item: item["label"] == "NONE" and item["prediction"] == "READY",
        "READY_to_other": lambda item: item["label"] == "READY" and item["prediction"] != "READY",
        "WAITING_to_other": lambda item: item["label"] == "WAITING" and item["prediction"] != "WAITING",
    }
    result: dict[str, Any] = {}
    for name, predicate in categories.items():
        errors = [item for item in records if predicate(item)]
        buckets = Counter(boundary_bucket(int(item["boundary_distance"])) for item in errors)
        result[name] = {
            "total": len(errors),
            "boundary": sum(bool(item["is_boundary"]) for item in errors),
            "non_boundary": sum(not bool(item["is_boundary"]) for item in errors),
            "distance_buckets": {bucket: buckets.get(bucket, 0) for bucket in BOUNDARY_BUCKETS},
            "nearest_transition_counts": dict(
                sorted(Counter(str(item.get("nearest_transition", "unknown")) for item in errors).items())
            ),
            "transition_counts_by_distance_bucket": {
                bucket: dict(
                    sorted(
                        Counter(
                            str(item.get("nearest_transition", "unknown"))
                            for item in errors
                            if boundary_bucket(int(item["boundary_distance"])) == bucket
                        ).items()
                    )
                )
                for bucket in BOUNDARY_BUCKETS
            },
            "representative_frames": [
                f"{item['session_id']}:{item['frame_index']}"
                for item in uniform_sample(sorted(errors, key=lambda item: (item["session_id"], item["frame_index"])), 8)
            ],
        }
    boundary_records = [item for item in records if item["is_boundary"]]
    interior_records = [item for item in records if not item["is_boundary"]]
    result["error_rates"] = {
        "boundary": _error_rate(boundary_records),
        "non_boundary": _error_rate(interior_records),
    }
    return result


def _error_rate(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    errors = sum(item["label"] != item["prediction"] for item in records)
    return {
        "samples": len(records),
        "errors": errors,
        "rate": errors / len(records) if records else 0.0,
    }


def preprocess_views(image: np.ndarray) -> dict[str, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError("Prompt crop must be a non-empty image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0, 0, 155)), np.array((179, 150, 255)))
    edges = cv2.Canny(clahe, 80, 180)
    return {"rgb": image, "gray": gray, "contrast": clahe, "text_mask": mask, "edges": edges}


def image_statistics(image: np.ndarray) -> dict[str, Any]:
    views = preprocess_views(image)
    gray = views["gray"]
    mask = views["text_mask"]
    edges = views["edges"]
    height, width = gray.shape
    nonzero = cv2.findNonZero(mask)
    if nonzero is None:
        centroid = [None, None]
        box = None
        touches_edge = False
    else:
        x, y, box_width, box_height = cv2.boundingRect(nonzero)
        moments = cv2.moments(mask, binaryImage=True)
        centroid = [
            float(moments["m10"] / moments["m00"] / width),
            float(moments["m01"] / moments["m00"] / height),
        ]
        box = [x / width, y / height, (x + box_width) / width, (y + box_height) / height]
        margin_x, margin_y = max(2, round(width * 0.01)), max(2, round(height * 0.04))
        components, _, component_stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        touches_edge = False
        for component in range(1, components):
            component_x, component_y, component_width, component_height, area = component_stats[component]
            text_sized = (
                4 <= area <= width * height * 0.01
                and component_width <= width * 0.20
                and component_height <= height * 0.35
            )
            if text_sized and (
                component_x <= margin_x
                or component_y <= margin_y
                or component_x + component_width >= width - margin_x
                or component_y + component_height >= height - margin_y
            ):
                touches_edge = True
                break
    histogram = cv2.calcHist([gray], [0], None, [32], [0, 256]).reshape(-1)
    histogram = histogram / max(float(histogram.sum()), 1.0)
    return {
        "width": width,
        "height": height,
        "mean_brightness": float(gray.mean()),
        "grayscale_std": float(gray.std()),
        "black_pixel_ratio": float(np.mean(gray <= 10)),
        "bright_pixel_ratio": float(np.mean(gray >= 200)),
        "edge_density": float(np.mean(edges > 0)),
        "text_mask_occupancy": float(np.mean(mask > 0)),
        "text_mask_centroid": centroid,
        "text_mask_box": box,
        "text_mask_touches_edge": bool(touches_edge),
        "suspected_truncation": bool(touches_edge and np.mean(mask > 0) > 0.002),
        "histogram": histogram.tolist(),
    }


def perceptual_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequency = cv2.dct(resized)[:8, :8]
    values = low_frequency.flatten()
    median = float(np.median(values[1:]))
    return _bits_to_int(values > median)


def difference_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return _bits_to_int((resized[:, 1:] > resized[:, :-1]).flatten())


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def hash_distance(first: int, second: int) -> int:
    return (int(first) ^ int(second)).bit_count()


def find_cross_label_near_duplicates(
    records: Sequence[dict[str, Any]], maximum: int = 200
) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[str(record["label"])].append(record)
    heap: list[tuple[int, int, dict[str, Any]]] = []
    sequence = itertools.count()
    for label_a, label_b in (("IDLE", "READY"), ("READY", "NONE"), ("WAITING", "NONE")):
        for first in by_label[label_a]:
            for second in by_label[label_b]:
                phash_distance = hash_distance(first["phash"], second["phash"])
                dhash_distance = hash_distance(first["dhash"], second["dhash"])
                score = phash_distance + dhash_distance
                pair = {
                    "session_a": first["session_id"],
                    "frame_a": first["frame_index"],
                    "label_a": first["label"],
                    "original_state_a": first["original_state"],
                    "session_b": second["session_id"],
                    "frame_b": second["frame_index"],
                    "label_b": second["label"],
                    "original_state_b": second["original_state"],
                    "phash_distance": phash_distance,
                    "dhash_distance": dhash_distance,
                    "is_boundary_a": first["is_boundary"],
                    "is_boundary_b": second["is_boundary"],
                    "crop_path_a": first["crop_path"],
                    "crop_path_b": second["crop_path"],
                }
                item = (-score, -phash_distance, next(sequence), pair)
                if len(heap) < maximum:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    return sorted(
        (item[3] for item in heap),
        key=lambda item: (item["phash_distance"] + item["dhash_distance"], item["phash_distance"], item["session_a"], item["frame_a"]),
    )


def histogram_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return float(cv2.compareHist(np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32), cv2.HISTCMP_BHATTACHARYYA))


def summarize_session_roi(records: Sequence[dict[str, Any]], normalized_roi: Sequence[float]) -> dict[str, Any]:
    if not records:
        raise ValueError("Session ROI summary requires records")
    numeric = ("mean_brightness", "grayscale_std", "black_pixel_ratio", "bright_pixel_ratio", "edge_density", "text_mask_occupancy")
    result: dict[str, Any] = {
        "sample_count": len(records),
        "crop_widths": sorted({int(item["stats"]["width"]) for item in records}),
        "crop_heights": sorted({int(item["stats"]["height"]) for item in records}),
        "source_widths": sorted({int(item["source_width"]) for item in records}),
        "source_heights": sorted({int(item["source_height"]) for item in records}),
        "normalized_roi": list(normalized_roi),
        "crop_out_of_frame_count": sum(bool(item["crop_out_of_frame"]) for item in records),
        "suspected_truncation_count": sum(bool(item["stats"]["suspected_truncation"]) for item in records),
        "manual_review_required": True,
    }
    for name in numeric:
        values = np.asarray([float(item["stats"][name]) for item in records])
        result[name] = {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()), "max": float(values.max())}
    centroids = [item["stats"]["text_mask_centroid"] for item in records if item["stats"]["text_mask_centroid"][0] is not None]
    result["mean_text_centroid"] = [float(np.mean([value[axis] for value in centroids])) for axis in (0, 1)] if centroids else [None, None]
    result["label_counts"] = dict(sorted(Counter(str(item["label"]) for item in records).items()))
    return result


def protected_file_hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        source = Path(path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        result[str(source.resolve())] = digest
    return result


def assert_hashes_unchanged(before: dict[str, str]) -> None:
    after = protected_file_hashes(before)
    changed = [path for path, digest in before.items() if after[path] != digest]
    if changed:
        raise RuntimeError(f"Protected analysis inputs changed: {changed}")
