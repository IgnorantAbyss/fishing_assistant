"""Visual and statistical diagnostics for the frozen PromptClassifier v1."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_roi_config  # noqa: E402
from src.dataset.manifest import ManifestRow, read_manifest  # noqa: E402
from src.ml.failure_analysis import (  # noqa: E402
    assert_hashes_unchanged,
    boundary_analysis,
    boundary_distance,
    difference_hash,
    find_cross_label_near_duplicates,
    group_confusion_errors,
    hash_distance,
    histogram_distance,
    image_statistics,
    none_breakdown,
    perceptual_hash,
    preprocess_views,
    protected_file_hashes,
    summarize_session_roi,
    transition_frames,
    uniform_sample,
)
from src.ml.prompt_dataset import CLASS_NAMES  # noqa: E402
from src.ml.prompt_model import load_prompt_checkpoint  # noqa: E402
from src.ml.prompt_transforms import LetterboxResize, build_prompt_transform  # noqa: E402


DATASET_ROOT = PROJECT_ROOT / "datasets" / "fishing_v2"
MANIFEST_PATH = DATASET_ROOT / "prompt" / "manifest.csv"
SESSION_ROOT = PROJECT_ROOT / "assets" / "replay" / "sessions"
SPLIT_PATH = PROJECT_ROOT / "config" / "session_split.yaml"
ROI_PATH = PROJECT_ROOT / "config" / "roi.yaml"
MODEL_PATH = PROJECT_ROOT / "models" / "prompt_classifier_v1" / "best_model.pt"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "prompt_classifier_v1" / "confidence_threshold.json"
REPORT_DIR = PROJECT_ROOT / "reports" / "prompt_classifier_v1"
SHEET_DIR = REPORT_DIR / "failure_contact_sheets"
SALIENCY_DIR = REPORT_DIR / "saliency"
NEAR_DUPLICATE_FIELDS = (
    "session_a", "frame_a", "label_a", "original_state_a",
    "session_b", "frame_b", "label_b", "original_state_b",
    "phash_distance", "dhash_distance", "is_boundary_a", "is_boundary_b",
    "crop_path_a", "crop_path_b",
)


def _session_metadata(session_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    session = SESSION_ROOT / session_id
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    ground_truth = yaml.safe_load((session / "ground_truth.yaml").read_text(encoding="utf-8"))
    if not isinstance(ground_truth, dict) or not isinstance(ground_truth.get("segments"), list):
        raise ValueError(f"Invalid ground truth for {session_id}")
    return manifest, ground_truth["segments"]


def _fixed_inference(rows: Sequence[ManifestRow], threshold: float) -> list[dict[str, Any]]:
    if abs(threshold - 0.55) > 1e-12:
        raise ValueError(f"Failure analysis requires frozen threshold 0.55, got {threshold}")
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model, checkpoint = load_prompt_checkpoint(MODEL_PATH, device=device)
    model.eval()
    transform = build_prompt_transform(
        False,
        width=int(checkpoint.get("input_width", 320)),
        height=int(checkpoint.get("input_height", 96)),
    )
    session_cache: dict[str, tuple[dict[str, Any], list[dict[str, Any]], list[int]]] = {}
    records: list[dict[str, Any]] = []
    batch_size = 32
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        tensors = []
        for row in batch_rows:
            with Image.open(DATASET_ROOT / row.crop_path) as image:
                tensors.append(transform(image.convert("RGB")))
        with torch.inference_mode():
            probabilities = torch.softmax(model(torch.stack(tensors)), dim=1).cpu().numpy()
        for row, values in zip(batch_rows, probabilities, strict=True):
            confidence = float(values.max())
            prediction = CLASS_NAMES[int(values.argmax())] if confidence >= threshold else "UNKNOWN"
            if row.session_id not in session_cache:
                session_manifest, segments = _session_metadata(row.session_id)
                session_cache[row.session_id] = (session_manifest, segments, transition_frames(segments))
            session_manifest, segments, transitions = session_cache[row.session_id]
            transition_details = [
                (
                    int(segments[index]["start"]),
                    f"{segments[index - 1]['state']}_to_{segments[index]['state']}",
                )
                for index in range(1, len(segments))
            ]
            nearest_transition = min(
                transition_details,
                key=lambda item: abs(row.frame_index - item[0]),
            )[1] if transition_details else "none"
            source_width, source_height = [int(value) for value in session_manifest["screen_size"]]
            records.append(
                {
                    "session_id": row.session_id,
                    "frame_index": row.frame_index,
                    "label": row.label,
                    "original_state": row.original_state,
                    "prediction": prediction,
                    "confidence": confidence,
                    "probabilities": {name: float(values[index]) for index, name in enumerate(CLASS_NAMES)},
                    "is_boundary": row.is_boundary,
                    "boundary_distance": boundary_distance(row.frame_index, transitions),
                    "nearest_transition": nearest_transition,
                    "crop_path": row.crop_path,
                    "crop_file": str(DATASET_ROOT / row.crop_path),
                    "source_frame": str(PROJECT_ROOT / row.source_frame),
                    "source_width": source_width,
                    "source_height": source_height,
                    "split": row.split,
                }
            )
    return records


def _feature_records(rows: Sequence[ManifestRow]) -> list[dict[str, Any]]:
    roi_config = load_roi_config(ROI_PATH)
    normalized = roi_config.rois["top_prompt"]
    session_manifests: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        image = cv2.imread(str(DATASET_ROOT / row.crop_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read prompt crop: {row.crop_path}")
        if row.session_id not in session_manifests:
            session_manifests[row.session_id] = _session_metadata(row.session_id)[0]
        source_width, source_height = [int(value) for value in session_manifests[row.session_id]["screen_size"]]
        raw_left = normalized[0] * source_width
        raw_top = normalized[1] * source_height
        raw_right = normalized[2] * source_width
        raw_bottom = normalized[3] * source_height
        records.append(
            {
                "session_id": row.session_id,
                "frame_index": row.frame_index,
                "label": row.label,
                "original_state": row.original_state,
                "split": row.split,
                "is_boundary": row.is_boundary,
                "crop_path": row.crop_path,
                "crop_file": str(DATASET_ROOT / row.crop_path),
                "source_frame": str(PROJECT_ROOT / row.source_frame),
                "source_width": source_width,
                "source_height": source_height,
                "crop_out_of_frame": raw_left < 0 or raw_top < 0 or raw_right > source_width or raw_bottom > source_height,
                "stats": image_statistics(image),
                "phash": perceptual_hash(image),
                "dhash": difference_hash(image),
            }
        )
        if index % 500 == 0:
            print(f"feature_crops: {index}/{len(rows)}")
    return records


def _fit_panel(image: np.ndarray, width: int = 215, height: int = 105) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _write_failure_sheet(records: Sequence[dict[str, Any]], destination: Path) -> None:
    ordered = sorted(records, key=lambda item: (item["session_id"], item["frame_index"]))
    selected = uniform_sample(ordered, 50)
    if not selected:
        placeholder = np.zeros((160, 900, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No matching errors", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), placeholder)
        return
    tile_width, tile_height = 1075, 180
    columns = 2
    sheet = np.zeros((((len(selected) + 1) // 2) * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for position, record in enumerate(selected):
        source = cv2.imread(record["source_frame"], cv2.IMREAD_COLOR)
        crop = cv2.imread(record["crop_file"], cv2.IMREAD_COLOR)
        if source is None or crop is None:
            continue
        views = preprocess_views(crop)
        panels = [source, crop, views["gray"], views["contrast"], views["text_mask"]]
        tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
        for panel_index, panel in enumerate(panels):
            tile[:105, panel_index * 215 : (panel_index + 1) * 215] = _fit_panel(panel)
        names = ("full frame", "crop", "gray", "CLAHE", "text mask")
        for panel_index, name in enumerate(names):
            cv2.putText(tile, name, (panel_index * 215 + 4, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 220, 255), 1)
        probs = record["probabilities"]
        line1 = f"{record['session_id']} #{record['frame_index']} GT={record['label']} original={record['original_state']} pred={record['prediction']} conf={record['confidence']:.3f}"
        line2 = f"P I={probs['IDLE']:.2f} W={probs['WAITING']:.2f} R={probs['READY']:.2f} N={probs['NONE']:.2f} boundary={record['is_boundary']} distance={record['boundary_distance']}"
        cv2.putText(tile, line1, (4, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(tile, line2, (4, 161), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        row_index, column_index = divmod(position, columns)
        sheet[row_index * tile_height : (row_index + 1) * tile_height, column_index * tile_width : (column_index + 1) * tile_width] = tile
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])


def _write_roi_overview(records: Sequence[dict[str, Any]], destination: Path) -> None:
    selected: list[dict[str, Any]] = []
    for label in CLASS_NAMES:
        label_records = sorted((item for item in records if item["label"] == label), key=lambda item: item["frame_index"])
        selected.extend(uniform_sample(label_records, 10))
    cell_width, cell_height, columns = 340, 140, 5
    rows = max(1, (len(selected) + columns - 1) // columns)
    sheet = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
    for position, record in enumerate(selected):
        crop = cv2.imread(record["crop_file"], cv2.IMREAD_COLOR)
        if crop is None:
            continue
        cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        fitted = _fit_panel(crop, 340, 102)
        cell[:102] = fitted
        stats = record["stats"]
        cv2.putText(cell, f"{record['label']}/{record['original_state']} #{record['frame_index']}", (4, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(cell, f"mask={stats['text_mask_occupancy']:.3f} edge={stats['edge_density']:.3f} trunc?={stats['suspected_truncation']}", (4, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 220, 255), 1)
        y, x = divmod(position, columns)
        sheet[y * cell_height : (y + 1) * cell_height, x * cell_width : (x + 1) * cell_width] = cell
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 86])


def _write_duplicate_csv(pairs: Sequence[dict[str, Any]]) -> None:
    destination = REPORT_DIR / "cross_label_near_duplicates.csv"
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=NEAR_DUPLICATE_FIELDS)
        writer.writeheader()
        writer.writerows(pairs)


def _write_duplicate_sheet(pairs: Sequence[dict[str, Any]]) -> None:
    selected = list(pairs[:50])
    cell_width, cell_height, columns = 680, 145, 2
    sheet = np.zeros((max(1, (len(selected) + 1) // 2) * cell_height, columns * cell_width, 3), dtype=np.uint8)
    for position, pair in enumerate(selected):
        first = cv2.imread(str(DATASET_ROOT / pair["crop_path_a"]), cv2.IMREAD_COLOR)
        second = cv2.imread(str(DATASET_ROOT / pair["crop_path_b"]), cv2.IMREAD_COLOR)
        if first is None or second is None:
            continue
        cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        cell[:100, :340] = _fit_panel(first, 340, 100)
        cell[:100, 340:] = _fit_panel(second, 340, 100)
        text_a = f"{pair['label_a']}/{pair['original_state_a']} {pair['session_a']}#{pair['frame_a']}"
        text_b = f"{pair['label_b']}/{pair['original_state_b']} {pair['session_b']}#{pair['frame_b']}"
        cv2.putText(cell, text_a, (4, 117), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        cv2.putText(cell, text_b, (342, 117), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        cv2.putText(cell, f"pHash={pair['phash_distance']} dHash={pair['dhash_distance']} boundaries={pair['is_boundary_a']}/{pair['is_boundary_b']}", (4, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 255), 1)
        y, x = divmod(position, columns)
        sheet[y * cell_height : (y + 1) * cell_height, x * cell_width : (x + 1) * cell_width] = cell
    cv2.imwrite(str(REPORT_DIR / "cross_label_near_duplicates.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])


def _text_separation(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    features = ("mean_brightness", "grayscale_std", "black_pixel_ratio", "bright_pixel_ratio", "edge_density", "text_mask_occupancy")
    label_summary: dict[str, Any] = {}
    for label in CLASS_NAMES:
        items = [item for item in records if item["label"] == label]
        label_summary[label] = {
            name: {
                "mean": float(np.mean([item["stats"][name] for item in items])),
                "std": float(np.std([item["stats"][name] for item in items])),
            }
            for name in features
        }
    fisher: dict[str, float] = {}
    for name in features:
        groups = [np.asarray([item["stats"][name] for item in records if item["label"] == label]) for label in CLASS_NAMES]
        overall = float(np.mean(np.concatenate(groups)))
        between = sum(len(group) * (float(group.mean()) - overall) ** 2 for group in groups)
        within = sum(float(((group - group.mean()) ** 2).sum()) for group in groups)
        fisher[name] = between / max(within, 1e-12)
    pair_distances: dict[str, float] = {}
    for left_index, left in enumerate(CLASS_NAMES):
        for right in CLASS_NAMES[left_index + 1 :]:
            first = uniform_sample([item for item in records if item["label"] == left], 30)
            second = uniform_sample([item for item in records if item["label"] == right], 30)
            distances = [histogram_distance(a["stats"]["histogram"], b["stats"]["histogram"]) for a in first for b in second]
            pair_distances[f"{left}_vs_{right}"] = float(np.mean(distances)) if distances else 0.0
    same_label_cross_session: list[float] = []
    different_label_same_session: list[float] = []
    sessions = sorted({item["session_id"] for item in records})
    for label in CLASS_NAMES:
        representatives = [uniform_sample([item for item in records if item["session_id"] == session and item["label"] == label], 3) for session in sessions]
        for index, first in enumerate(representatives):
            for second in representatives[index + 1 :]:
                same_label_cross_session.extend(histogram_distance(a["stats"]["histogram"], b["stats"]["histogram"]) for a in first for b in second)
    for session in sessions:
        groups = {label: uniform_sample([item for item in records if item["session_id"] == session and item["label"] == label], 3) for label in CLASS_NAMES}
        for index, left in enumerate(CLASS_NAMES):
            for right in CLASS_NAMES[index + 1 :]:
                different_label_same_session.extend(histogram_distance(a["stats"]["histogram"], b["stats"]["histogram"]) for a in groups[left] for b in groups[right])
    return {
        "label_statistics": label_summary,
        "separation_fisher_ratio": fisher,
        "histogram_distance_by_label_pair": pair_distances,
        "mean_same_label_cross_session_histogram_distance": float(np.mean(same_label_cross_session)),
        "mean_different_label_same_session_histogram_distance": float(np.mean(different_label_same_session)),
    }


def _saliency(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    model, checkpoint = load_prompt_checkpoint(MODEL_PATH, device="cpu")
    model.eval()
    transform = build_prompt_transform(False, width=int(checkpoint["input_width"]), height=int(checkpoint["input_height"]))
    letterbox = LetterboxResize(int(checkpoint["input_width"]), int(checkpoint["input_height"]))
    categories = {
        "correct_WAITING": [item for item in records if item["label"] == "WAITING" and item["prediction"] == "WAITING"],
        "correct_READY": [item for item in records if item["label"] == "READY" and item["prediction"] == "READY"],
        "IDLE_to_READY": [item for item in records if item["label"] == "IDLE" and item["prediction"] == "READY"],
        "NONE_to_READY": [item for item in records if item["label"] == "NONE" and item["prediction"] == "READY"],
    }
    output: dict[str, Any] = {"method": "absolute input-gradient saliency; descriptive, not causal", "categories": {}}
    SALIENCY_DIR.mkdir(parents=True, exist_ok=True)
    for category, candidates in categories.items():
        chosen = uniform_sample(sorted(candidates, key=lambda item: (item["session_id"], item["frame_index"])), 10)
        cells: list[np.ndarray] = []
        stats: list[dict[str, Any]] = []
        for record in chosen:
            with Image.open(record["crop_file"]) as source:
                rgb = source.convert("RGB")
            tensor = transform(rgb).unsqueeze(0).requires_grad_(True)
            logits = model(tensor)
            target = int(logits.argmax(dim=1).item())
            gradient = torch.autograd.grad(logits[0, target], tensor, retain_graph=False)[0]
            saliency = gradient.detach().abs().mean(dim=1)[0].numpy()
            saliency /= max(float(saliency.max()), 1e-12)
            base_rgb = np.asarray(letterbox(rgb))
            crop_bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
            mask = preprocess_views(crop_bgr)["text_mask"]
            mask_letterbox = np.asarray(letterbox(Image.fromarray(mask).convert("RGB")))[:, :, 0] > 0
            total = max(float(saliency.sum()), 1e-12)
            border = np.zeros_like(mask_letterbox)
            border[:10, :] = True
            border[-10:, :] = True
            border[:, :16] = True
            border[:, -16:] = True
            text_fraction = float(saliency[mask_letterbox].sum() / total)
            text_occupancy = float(mask_letterbox.mean())
            border_fraction = float(saliency[border].sum() / total)
            stats.append({
                "session_id": record["session_id"], "frame_index": record["frame_index"],
                "text_attention_fraction": text_fraction, "text_mask_occupancy": text_occupancy,
                "text_enrichment": text_fraction / max(text_occupancy, 1e-12), "border_attention_fraction": border_fraction,
            })
            heat = cv2.applyColorMap(np.uint8(saliency * 255), cv2.COLORMAP_JET)
            base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(base_bgr, 0.50, heat, 0.50, 0)
            cell = np.zeros((145, 660, 3), dtype=np.uint8)
            cell[:96, :320] = base_bgr
            cell[:96, 340:660] = overlay
            cv2.putText(cell, f"{record['session_id']} #{record['frame_index']} GT={record['label']} pred={record['prediction']}", (4, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
            cv2.putText(cell, f"text fraction={text_fraction:.3f} enrichment={stats[-1]['text_enrichment']:.2f} border={border_fraction:.3f}", (4, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 220, 255), 1)
            cells.append(cell)
        if cells:
            sheet = np.vstack(cells)
            cv2.imwrite(str(SALIENCY_DIR / f"{category}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        output["categories"][category] = {
            "sample_count": len(stats),
            "mean_text_attention_fraction": float(np.mean([item["text_attention_fraction"] for item in stats])) if stats else 0.0,
            "mean_text_enrichment": float(np.mean([item["text_enrichment"] for item in stats])) if stats else 0.0,
            "mean_border_attention_fraction": float(np.mean([item["border_attention_fraction"] for item in stats])) if stats else 0.0,
            "representatives": stats,
            "contact_sheet": str(SALIENCY_DIR / f"{category}.jpg"),
        }
    return output


def _near_duplicate_summary(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(f"{item['label_a']}_vs_{item['label_b']}" for item in pairs)
    near_exact = [item for item in pairs if item["phash_distance"] <= 6 and item["dhash_distance"] <= 8]
    exact_counts = Counter(f"{item['label_a']}_vs_{item['label_b']}" for item in near_exact)
    identical = [item for item in pairs if item["phash_distance"] == 0 and item["dhash_distance"] == 0]
    identical_counts = Counter(f"{item['label_a']}_vs_{item['label_b']}" for item in identical)
    return {
        "reported_pairs": len(pairs),
        "pairs_by_label": dict(sorted(counts.items())),
        "near_exact_rule": "phash_distance <= 6 and dhash_distance <= 8",
        "near_exact_count": len(near_exact),
        "near_exact_by_label": dict(sorted(exact_counts.items())),
        "identical_hash_count_within_top_200": len(identical),
        "identical_hash_by_label": dict(sorted(identical_counts.items())),
        "top_pairs": list(pairs[:20]),
        "manual_review_required": True,
    }


def _diagnoses(
    roi: dict[str, Any], boundary: dict[str, Any], duplicates: dict[str, Any],
    saliency: dict[str, Any], validation_metrics: dict[str, Any], test_metrics: dict[str, Any],
) -> dict[str, Any]:
    sessions = roi["sessions"]
    test_ids = ("session_20260710_130308", "session_20260710_131254")
    test_trunc = sum(sessions[item]["suspected_truncation_count"] for item in test_ids)
    test_samples = sum(sessions[item]["sample_count"] for item in test_ids)
    out_of_frame = sum(item["crop_out_of_frame_count"] for item in sessions.values())
    if out_of_frame:
        roi_status = "confirmed"
    else:
        roi_status = "not_observed"
    ready_none_conflicts = duplicates["near_exact_by_label"].get("READY_vs_NONE", 0)
    label_status = "confirmed" if ready_none_conflicts >= 10 else ("likely" if ready_none_conflicts else "not_observed")
    boundary_rate = boundary["error_rates"]["boundary"]["rate"]
    interior_rate = boundary["error_rates"]["non_boundary"]["rate"]
    boundary_status = "confirmed" if boundary_rate > interior_rate * 1.5 and boundary["error_rates"]["boundary"]["errors"] >= 10 else ("likely" if boundary_rate > interior_rate * 1.1 else "not_observed")
    error_saliency = [saliency["categories"][name] for name in ("IDLE_to_READY", "NONE_to_READY")]
    mean_enrichment = float(np.mean([item["mean_text_enrichment"] for item in error_saliency]))
    mean_border = float(np.mean([item["mean_border_attention_fraction"] for item in error_saliency]))
    shortcut_status = "likely" if mean_enrichment < 2.0 or mean_border > 0.25 else "inconclusive"
    diversity_gap = float(validation_metrics["macro_f1"]) - float(test_metrics["macro_f1"])
    diversity_status = "likely" if diversity_gap > 0.10 else "not_observed"
    return {
        "ROI_misalignment": {
            "status": roi_status,
            "evidence": {
                "out_of_frame": out_of_frame,
                "automated_suspected_truncation": test_trunc,
                "test_samples": test_samples,
                "automated_mask_warning": "Bright-mask edge heuristic is background-sensitive and is not accepted as truncation proof.",
                "manual_review": {
                    "session_20260710_130308": "No visible prompt clipping or displacement; full text is inside ROI.",
                    "session_20260710_131254": "No visible prompt clipping or displacement; full text is inside ROI.",
                    "roi_too_broad": "likely: ROI includes substantial scenery and the bottom status strip.",
                },
                "contact_sheets": [str(SHEET_DIR / "session_roi_overview_130308.jpg"), str(SHEET_DIR / "session_roi_overview_131254.jpg")],
            },
            "manual_review_required": True,
        },
        "Prompt_label_conflict": {"status": label_status, "evidence": {"near_exact_READY_vs_NONE": ready_none_conflicts, "contact_sheet": str(REPORT_DIR / "cross_label_near_duplicates.jpg"), "representatives": duplicates["top_pairs"][:5]}},
        "Boundary_label_noise": {"status": boundary_status, "evidence": {"boundary_error_rate": boundary_rate, "non_boundary_error_rate": interior_rate, "NONE_to_READY": boundary["NONE_to_READY"], "IDLE_to_READY": boundary["IDLE_to_READY"]}},
        "Background_shortcut_learning": {"status": shortcut_status, "evidence": {"error_mean_text_enrichment": mean_enrichment, "error_mean_border_attention": mean_border, "saliency_dir": str(SALIENCY_DIR)}, "limitation": saliency["method"]},
        "Insufficient_session_diversity": {"status": diversity_status, "evidence": {"validation_macro_f1": validation_metrics["macro_f1"], "test_macro_f1": test_metrics["macro_f1"], "gap": diversity_gap, "test_sessions": list(test_ids)}},
    }


def _recommendations(diagnoses: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"recommendation_rank": 1, "option": "Fix prompt-specific label definition", "expected_benefit": "high", "implementation_cost": "medium", "evidence": diagnoses["Prompt_label_conflict"], "prerequisite": "Human review of READY/NONE near-duplicate sheets"},
        {"recommendation_rank": 2, "option": "Label NONE from actual prompt visibility", "expected_benefit": "high", "implementation_cost": "high", "evidence": diagnoses["Prompt_label_conflict"], "prerequisite": "Prompt-specific annotation policy such as IGNORE_PROMPT"},
        {"recommendation_rank": 3, "option": "Exclude prompt boundary frames from training", "expected_benefit": "medium-high", "implementation_cost": "low", "evidence": diagnoses["Boundary_label_noise"], "prerequisite": "Keep global ground truth unchanged; filter only prompt training selection"},
        {"recommendation_rank": 4, "option": "Record targeted IDLE/READY sessions", "expected_benefit": "medium-high", "implementation_cost": "medium", "evidence": diagnoses["Insufficient_session_diversity"], "prerequisite": "Resolve label policy first"},
        {"recommendation_rank": 5, "option": "Add more train sessions", "expected_benefit": "medium", "implementation_cost": "medium", "evidence": diagnoses["Insufficient_session_diversity"], "prerequisite": "Do not add test sessions; resolve label conflicts first"},
        {"recommendation_rank": 6, "option": "Use tighter text ROI", "expected_benefit": "medium", "implementation_cost": "medium", "evidence": diagnoses["ROI_misalignment"], "prerequisite": "Manual ROI completeness review across all session sheets"},
        {"recommendation_rank": 7, "option": "Use grayscale or text-mask preprocessing", "expected_benefit": "medium", "implementation_cost": "low-medium", "evidence": diagnoses["Background_shortcut_learning"], "prerequisite": "First fix label/ROI issues"},
        {"recommendation_rank": 8, "option": "RGB plus text-mask dual-path model", "expected_benefit": "uncertain", "implementation_cost": "high", "evidence": diagnoses["Background_shortcut_learning"], "prerequisite": "Offline ablation after clean labels"},
        {"recommendation_rank": 9, "option": "OCR", "expected_benefit": "uncertain", "implementation_cost": "high", "evidence": "Not evaluated; intentionally not the first response", "prerequisite": "Only reconsider after label/ROI repair"},
        {"recommendation_rank": 10, "option": "Abandon classifier for pure FSM", "expected_benefit": "uncertain", "implementation_cost": "high", "evidence": "Current failure does not establish that the task is impossible", "prerequisite": "Evaluate repaired prompt labels first"},
    ]


def _write_markdown(report: dict[str, Any]) -> None:
    boundary = report["boundary_analysis"]
    lines = [
        "# PromptClassifier v1 Failure Analysis",
        "",
        "This is a frozen-model diagnostic run. It did not train, tune thresholds, or change model/data labels.",
        "",
        "## Scope and immutability",
        "",
        f"- Analyzed splits: `{report['analyzed_splits']}`",
        f"- Fixed threshold: {report['threshold']}",
        f"- Diagnostic inference records: {report['diagnostic_record_count']}",
        f"- Protected inputs unchanged: {report['immutability']['unchanged']}",
        "",
        "## Boundary errors",
        "",
        f"- IDLE -> READY: `{boundary['IDLE_to_READY']}`",
        f"- NONE -> READY: `{boundary['NONE_to_READY']}`",
        f"- Boundary error rate: `{boundary['error_rates']['boundary']}`",
        f"- Stable-interior error rate: `{boundary['error_rates']['non_boundary']}`",
        "",
        "## NONE by original state",
        "",
    ]
    for state, values in report["none_breakdown"].items():
        lines.append(f"- {state}: `{values}`")
    lines.extend(["", "## ROI consistency", ""])
    for session_id, values in report["roi_analysis"]["sessions"].items():
        lines.append(f"- {session_id}: `{values}`")
    lines.extend([
        "",
        "Prompt completeness and suspected truncation remain `manual_review_required`; heuristic masks are not ground truth.",
        f"- Manual findings: `{report['manual_review_findings']}`",
        "",
        "## Cross-label near duplicates",
        "",
        f"- Summary: `{report['near_duplicates']['near_exact_by_label']}`",
        f"- CSV: `{report['near_duplicates']['csv_path']}`",
        f"- Contact sheet: `{report['near_duplicates']['contact_sheet']}`",
        "",
        "## Text-region statistics",
        "",
        f"- Separation: `{report['text_analysis']}`",
        "",
        "## Saliency",
        "",
        f"- Method/limitation: {report['saliency']['method']}",
    ])
    for category, values in report["saliency"]["categories"].items():
        lines.append(f"- {category}: `{values}`")
    lines.extend(["", "## Five-part diagnosis", ""])
    for name, values in report["diagnoses"].items():
        lines.append(f"- {name}: **{values['status']}** — `{values}`")
    lines.extend(["", "## Ranked recommendations", ""])
    for item in report["recommendations"]:
        lines.append(f"{item['recommendation_rank']}. {item['option']} — benefit={item['expected_benefit']}, cost={item['implementation_cost']}; prerequisite: {item['prerequisite']}")
    lines.extend([
        "",
        "## Human-review contact sheets",
        "",
        *[f"- `{path}`" for path in report["contact_sheets"]],
    ])
    (REPORT_DIR / "failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(splits: Sequence[str]) -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = read_manifest(MANIFEST_PATH)
    formal_rows = [row for row in all_rows if row.split in {"train", "validation", "test"}]
    analysis_rows = [row for row in all_rows if row.split in set(splits)]
    if not analysis_rows:
        raise ValueError(f"No prompt rows for splits: {splits}")
    split_config = yaml.safe_load(SPLIT_PATH.read_text(encoding="utf-8"))
    official_sessions = [*split_config["train"], *split_config["validation"], *split_config["test"]]
    protected = [
        MANIFEST_PATH, SPLIT_PATH, ROI_PATH, MODEL_PATH, THRESHOLD_PATH,
        REPORT_DIR / "final_test_errors.csv", REPORT_DIR / "final_test_metrics.json",
        REPORT_DIR / "validation_errors.csv", REPORT_DIR / "validation_metrics.json",
        *[SESSION_ROOT / session / "ground_truth.yaml" for session in official_sessions],
    ]
    before = protected_file_hashes(protected)
    threshold_data = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    threshold = float(threshold_data["selected_threshold"])
    diagnostic_records = _fixed_inference(analysis_rows, threshold)
    features = _feature_records(formal_rows)
    feature_by_key = {(item["session_id"], item["frame_index"], item["label"]): item for item in features}
    for record in diagnostic_records:
        record["stats"] = feature_by_key[(record["session_id"], record["frame_index"], record["label"])]["stats"]
    grouped = group_confusion_errors(diagnostic_records)
    contact_sheets: list[str] = []
    required_groups = {
        "test_IDLE_to_READY_130308.jpg": [item for item in diagnostic_records if item["session_id"].endswith("130308") and item["label"] == "IDLE" and item["prediction"] == "READY"],
        "test_IDLE_to_READY_131254.jpg": [item for item in diagnostic_records if item["session_id"].endswith("131254") and item["label"] == "IDLE" and item["prediction"] == "READY"],
        "test_NONE_to_READY_130308.jpg": [item for item in diagnostic_records if item["session_id"].endswith("130308") and item["label"] == "NONE" and item["prediction"] == "READY"],
        "test_NONE_to_READY_131254.jpg": [item for item in diagnostic_records if item["session_id"].endswith("131254") and item["label"] == "NONE" and item["prediction"] == "READY"],
        "validation_IDLE_errors.jpg": [item for item in diagnostic_records if item["split"] == "validation" and item["label"] == "IDLE" and item["prediction"] != "IDLE"],
        "validation_NONE_errors.jpg": [item for item in diagnostic_records if item["split"] == "validation" and item["label"] == "NONE" and item["prediction"] != "NONE"],
    }
    for filename, records in required_groups.items():
        path = SHEET_DIR / filename
        _write_failure_sheet(records, path)
        contact_sheets.append(str(path))
    roi_config = load_roi_config(ROI_PATH)
    roi_sessions: dict[str, Any] = {}
    for session_id in official_sessions:
        session_records = [item for item in features if item["session_id"] == session_id]
        roi_sessions[session_id] = summarize_session_roi(session_records, roi_config.rois["top_prompt"])
        path = SHEET_DIR / f"session_roi_overview_{session_id.split('_')[-1]}.jpg"
        _write_roi_overview(session_records, path)
        contact_sheets.append(str(path))
    pairs = find_cross_label_near_duplicates(features, maximum=200)
    _write_duplicate_csv(pairs)
    _write_duplicate_sheet(pairs)
    duplicates = _near_duplicate_summary(pairs)
    duplicates.update({"csv_path": str(REPORT_DIR / "cross_label_near_duplicates.csv"), "contact_sheet": str(REPORT_DIR / "cross_label_near_duplicates.jpg")})
    saliency = _saliency(diagnostic_records)
    validation_metrics = json.loads((REPORT_DIR / "validation_metrics.json").read_text(encoding="utf-8"))
    test_metrics = json.loads((REPORT_DIR / "final_test_metrics.json").read_text(encoding="utf-8"))
    boundary = boundary_analysis(diagnostic_records)
    diagnoses = _diagnoses({"sessions": roi_sessions}, boundary, duplicates, saliency, validation_metrics, test_metrics)
    report: dict[str, Any] = {
        "analysis_type": "frozen_model_diagnostic_not_model_evaluation",
        "analyzed_splits": list(splits),
        "threshold": threshold,
        "diagnostic_record_count": len(diagnostic_records),
        "error_group_counts": {name: len(items) for name, items in sorted(grouped.items())},
        "none_breakdown": none_breakdown(diagnostic_records),
        "none_breakdown_by_split": {
            split: none_breakdown([item for item in diagnostic_records if item["split"] == split])
            for split in splits
        },
        "boundary_analysis": boundary,
        "boundary_analysis_by_split": {
            split: boundary_analysis([item for item in diagnostic_records if item["split"] == split])
            for split in splits
        },
        "roi_analysis": {"normalized_top_prompt": list(roi_config.rois["top_prompt"]), "sessions": roi_sessions, "manual_review_required": True},
        "near_duplicates": duplicates,
        "text_analysis": _text_separation(features),
        "saliency": saliency,
        "diagnoses": diagnoses,
        "recommendations": _recommendations(diagnoses),
        "manual_review_findings": {
            "test_roi_alignment": {
                "conclusion": "No visible misalignment or clipping in either test session; both retain the full prompt line.",
                "roi_too_broad": "Likely: substantial scenery and the bottom status strip remain in the crop.",
                "evidence": [
                    str(SHEET_DIR / "session_roi_overview_130308.jpg"),
                    str(SHEET_DIR / "session_roi_overview_131254.jpg"),
                ],
            },
            "READY_vs_NONE": {
                "conclusion": "Confirmed visual label conflict: many HOOK frames mapped to NONE still show the preceding READY/Space prompt; PRESS frames contain a different visible prompt rather than no prompt.",
                "representative_frames": [
                    "session_20260710_130308:21 (HOOK)",
                    "session_20260710_130308:415 (PRESS)",
                    "session_20260710_131254:190 (HOOK)",
                    "session_20260710_131254:234 (PRESS)",
                ],
                "evidence": [
                    str(SHEET_DIR / "test_NONE_to_READY_130308.jpg"),
                    str(SHEET_DIR / "test_NONE_to_READY_131254.jpg"),
                    str(REPORT_DIR / "cross_label_near_duplicates.jpg"),
                ],
            },
            "IDLE_vs_READY": {
                "conclusion": "The ROI layout/background is nearly identical; the reliable semantic difference is primarily the prompt text.",
                "representative_frames": [
                    "session_20260710_130308:69 (IDLE)",
                    "session_20260710_130308:1 (READY)",
                    "session_20260710_131254:282 (IDLE)",
                    "session_20260710_131254:121 (READY)",
                ],
                "evidence": str(REPORT_DIR / "cross_label_near_duplicates.jpg"),
            },
            "saliency": {
                "conclusion": "Input gradients overlap text but are diffuse across scenery, the bottom UI strip, and ROI edges; text is not the dominant isolated cue.",
                "evidence": str(SALIENCY_DIR),
                "limitation": "Absolute input-gradient saliency is descriptive and not causal.",
            },
        },
        "contact_sheets": contact_sheets,
        "immutability": {"protected_file_count": len(before), "unchanged": True, "model_sha256": before[str(MODEL_PATH.resolve())], "manifest_sha256": before[str(MANIFEST_PATH.resolve())], "split_sha256": before[str(SPLIT_PATH.resolve())]},
        "constraints": {"training_started": False, "threshold_searched": False, "model_modified": False, "split_modified": False, "ground_truth_modified": False},
    }
    assert_hashes_unchanged(before)
    (REPORT_DIR / "failure_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze frozen PromptClassifier v1 failures.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--split", choices=("validation", "test"))
    group.add_argument("--all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits = [args.split] if args.split else ["validation", "test"]
    report = analyze(splits)
    print(json.dumps({
        "analyzed_splits": report["analyzed_splits"],
        "diagnostic_records": report["diagnostic_record_count"],
        "diagnoses": {name: value["status"] for name, value in report["diagnoses"].items()},
        "report": str(REPORT_DIR / "failure_analysis.md"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
