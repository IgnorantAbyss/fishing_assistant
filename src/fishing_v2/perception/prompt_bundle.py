"""Build and load the immutable deployment bundle for Prototype PromptObserver."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from src.fishing_v2.data.dataset_lineage import sha256_file
from src.fishing_v2.data.prompt_observation_dataset import (
    FORMAL_SESSION_IDS,
    TRIAL_SESSION_ID,
    read_prompt_observation_manifest,
)
from src.fishing_v2.data.prompt_roi import PromptROICandidate, load_approved_prompt_roi
from src.fishing_v2.evaluation.prompt_prototype_evaluation import load_or_build_features
from src.fishing_v2.perception.prototype_prompt_observer import (
    OBSERVER_VERSION,
    OPERATIONAL_LABELS,
    PromptPrototype,
    PrototypePromptModel,
    PrototypePromptObserver,
    build_prototypes,
    extract_prompt_feature,
)


BUNDLE_VERSION = "prototype_v1_final_2"
BUNDLE_JSON = "bundle.json"
PROTOTYPES_FILE = "prototypes.npz"
METADATA_JSON = "metadata.json"
PREPROCESSING = {
    "color": "BGR_to_grayscale_luminance",
    "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 4]},
    "background_normalization": {"method": "gaussian_subtraction", "sigma_x": 7.0, "sigma_y": 7.0},
    "bright_text_mask": {"gray_min": 145, "positive_local_min": 8},
    "edge_emphasis": {"method": "sobel_magnitude", "kernel_size": 3},
    "fixed_size": [170, 16],
    "feature_channels": ["bright_local", "gradient"],
    "normalization": "per_channel_mean_center_l2_then_joint_l2",
    "similarity": "cosine_max_over_class_prototypes",
}
LIVE_CANDIDATE_STATUS = "live_calibration_candidate"
LIVE_REVIEW_STATUS = "human_confirmed_live_candidate"


class PromptBundleError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedPromptBundle:
    path: Path
    bundle_version: str
    bundle_sha256: str
    observer: PrototypePromptObserver
    model: PrototypePromptModel
    roi: PromptROICandidate
    bundle: Mapping[str, Any]
    metadata: Mapping[str, Any]


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _bundle_digest(bundle_bytes: bytes, prototype_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"fishing-prompt-bundle-v1\0")
    digest.update(bundle_bytes)
    digest.update(b"\0prototypes\0")
    digest.update(prototype_bytes)
    return digest.hexdigest()


def _fold_calibration(summary_path: Path) -> tuple[dict[str, float], float, int, list[dict[str, Any]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    folds = summary.get("folds")
    if not isinstance(folds, list) or len(folds) != len(FORMAL_SESSION_IDS):
        raise PromptBundleError("Final bundle requires all seven LOSO fold calibrations")
    held_out = {str(item.get("held_out_session")) for item in folds}
    if held_out != set(FORMAL_SESSION_IDS) or TRIAL_SESSION_ID in held_out:
        raise PromptBundleError("LOSO calibration sessions do not match the formal session set")
    thresholds = {
        label: float(np.median([float(item["class_thresholds"][label]) for item in folds]))
        for label in OPERATIONAL_LABELS
    }
    margin = float(np.median([float(item["ambiguity_margin"]) for item in folds]))
    idle_stability = max(int(item["idle_stability_frames"]) for item in folds)
    return thresholds, margin, idle_stability, folds


def _load_live_calibration_candidates(
    manifest_path: Path | None,
    *,
    project_root: Path,
    expected_roi: tuple[int, int, int, int],
) -> tuple[list[PromptPrototype], list[dict[str, Any]], str | None]:
    if manifest_path is None:
        return [], [], None
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("status") != LIVE_CANDIDATE_STATUS:
        raise PromptBundleError("Live Prompt candidate manifest has invalid status")
    raw = payload.get("candidates")
    if not isinstance(raw, list) or not 1 <= len(raw) <= 2:
        raise PromptBundleError("Final bundle permits only one or two reviewed Live candidates")
    prototypes: list[PromptPrototype] = []
    metadata: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise PromptBundleError("Each Live Prompt candidate must be a mapping")
        candidate_id = str(item.get("candidate_id", ""))
        label = str(item.get("label", ""))
        if not candidate_id or candidate_id in candidate_ids:
            raise PromptBundleError("Live Prompt candidate ids must be non-empty and unique")
        if label not in OPERATIONAL_LABELS or item.get("review_status") != LIVE_REVIEW_STATUS:
            raise PromptBundleError("Live Prompt candidate must have an operational label and human review")
        if tuple(item.get("roi", ())) != expected_roi:
            raise PromptBundleError("Live Prompt candidate ROI differs from the approved ROI")
        image_path = (project_root / str(item.get("image_path", ""))).resolve()
        if not image_path.is_relative_to(project_root.resolve()):
            raise PromptBundleError("Live Prompt candidate image must remain inside the project")
        expected_hash = str(item.get("image_sha256", ""))
        if not image_path.is_file() or sha256_file(image_path) != expected_hash:
            raise PromptBundleError("Live Prompt candidate image hash verification failed")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.shape != (expected_roi[3] - expected_roi[1], expected_roi[2] - expected_roi[0], 3):
            raise PromptBundleError("Live Prompt candidate image has an invalid shape")
        feature = extract_prompt_feature(np.ascontiguousarray(image))
        source_session = str(item.get("source_live_session", ""))
        source_frame = int(item.get("source_frame", 0))
        prototype_id = f"{label}:live_calibration_candidate:{candidate_id}"
        prototypes.append(PromptPrototype(
            prototype_id, label, source_session, source_frame, feature
        ))
        metadata.append({
            "candidate_id": candidate_id,
            "prototype_id": prototype_id,
            "label": label,
            "review_status": LIVE_REVIEW_STATUS,
            "source_live_session": source_session,
            "source_frame": source_frame,
            "image_path": str(item["image_path"]),
            "image_sha256": expected_hash,
            "source_screenshot_sha256": str(item.get("source_screenshot_sha256", "")),
            "used_for_threshold_calibration": False,
            "used_as_ground_truth": False,
        })
        candidate_ids.add(candidate_id)
    return prototypes, metadata, sha256_file(manifest_path)


def build_final_prompt_bundle(
    *,
    config_path: str | Path,
    manifest_path: str | Path,
    dataset_root: str | Path,
    session_root: str | Path,
    loso_summary_path: str | Path,
    output_dir: str | Path,
    feature_cache: str | Path | None = None,
    live_candidate_manifest: str | Path | None = None,
) -> Path:
    config_path = Path(config_path)
    manifest_path = Path(manifest_path)
    dataset_root = Path(dataset_root)
    session_root = Path(session_root)
    loso_summary_path = Path(loso_summary_path)
    output_dir = Path(output_dir)
    candidate_manifest_path = Path(live_candidate_manifest) if live_candidate_manifest else None
    rows = read_prompt_observation_manifest(manifest_path)
    sessions = tuple(sorted({row["session_id"] for row in rows}))
    if sessions != tuple(sorted(FORMAL_SESSION_IDS)) or TRIAL_SESSION_ID in sessions:
        raise PromptBundleError("Final bundle training data must be exactly the seven formal sessions")
    roi = load_approved_prompt_roi(config_path)
    if roi is None or roi.pixel != (940, 36, 1620, 100):
        raise PromptBundleError("Final bundle requires approved Prompt ROI [940, 36, 1620, 100]")
    features = load_or_build_features(rows, dataset_root, cache_path=feature_cache)
    base_prototypes = build_prototypes(rows, features, FORMAL_SESSION_IDS)
    if len(base_prototypes) != len(OPERATIONAL_LABELS) * len(FORMAL_SESSION_IDS):
        raise PromptBundleError(f"Expected 35 formal-session prototypes, built {len(base_prototypes)}")
    live_prototypes, live_candidates, candidate_manifest_hash = _load_live_calibration_candidates(
        candidate_manifest_path,
        project_root=config_path.resolve().parents[1],
        expected_roi=roi.pixel,
    )
    prototypes = tuple(base_prototypes) + tuple(live_prototypes)
    thresholds, margin, idle_stability, folds = _fold_calibration(loso_summary_path)
    ground_truth_hashes = {
        session_id: sha256_file(session_root / session_id / "prompt_ground_truth.yaml")
        for session_id in FORMAL_SESSION_IDS
    }
    prototype_index = [
        {
            "row": index,
            "prototype_id": item.prototype_id,
            "label": item.label,
            "source_session": item.source_session,
            "source_frame": item.source_frame,
            "provenance": (
                "formal_session_medoid"
                if item.source_session in FORMAL_SESSION_IDS
                else LIVE_REVIEW_STATUS
            ),
        }
        for index, item in enumerate(prototypes)
    ]
    bundle: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "observer_version": OBSERVER_VERSION,
        "approved_roi": {
            "pixel": list(roi.pixel),
            "reference_resolution": [roi.reference_width, roi.reference_height],
            "source": "config/fishing_v2.yaml user-approved pixel ROI",
        },
        "preprocessing": PREPROCESSING,
        "base_prototype_count": len(base_prototypes),
        "live_calibration_candidate_count": len(live_prototypes),
        "prototype_count": len(prototypes),
        "prototype_index": prototype_index,
        "live_calibration_candidates": live_candidates,
        "class_thresholds": thresholds,
        "ambiguity_margin": margin,
        "idle_stability_frames": idle_stability,
        "threshold_aggregation": {
            "source": "seven LOSO folds with nested training-session calibration",
            "class_thresholds": "per-class median across seven held-out folds",
            "ambiguity_margin": "median across seven held-out folds",
            "idle_stability_frames": "maximum across seven held-out folds",
            "runtime_adaptation": False,
            "live_candidates_used_for_threshold_calibration": False,
        },
        "training_session_ids": list(FORMAL_SESSION_IDS),
        "excluded_session_ids": [TRIAL_SESSION_ID],
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_loso_summary_sha256": sha256_file(loso_summary_path),
        "live_candidate_manifest_sha256": candidate_manifest_hash,
        "prompt_ground_truth_sha256": ground_truth_hashes,
        "fold_calibration": [
            {
                "held_out_session": item["held_out_session"],
                "class_thresholds": item["class_thresholds"],
                "ambiguity_margin": item["ambiguity_margin"],
                "idle_stability_frames": item["idle_stability_frames"],
            }
            for item in folds
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    prototype_path = output_dir / PROTOTYPES_FILE
    np.savez_compressed(prototype_path, features=np.stack([item.feature for item in prototypes]))
    prototype_bytes = prototype_path.read_bytes()
    bundle["prototypes_sha256"] = hashlib.sha256(prototype_bytes).hexdigest()
    bundle_bytes = _json_bytes(bundle)
    (output_dir / BUNDLE_JSON).write_bytes(bundle_bytes)
    bundle_sha256 = _bundle_digest(bundle_bytes, prototype_bytes)
    metadata = {
        "bundle_version": BUNDLE_VERSION,
        "bundle_sha256": bundle_sha256,
        "hash_definition": "sha256(domain_separator + bundle.json bytes + separator + prototypes.npz bytes)",
        "bundle_json_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "prototypes_sha256": bundle["prototypes_sha256"],
        "immutable_runtime_calibration": True,
        "contains_neural_network_weights": False,
        "contains_fonts": False,
    }
    (output_dir / METADATA_JSON).write_bytes(_json_bytes(metadata))
    loaded = load_prompt_bundle(output_dir)
    if loaded.bundle_sha256 != bundle_sha256:
        raise PromptBundleError("Final bundle self-verification failed")
    return output_dir


def load_prompt_bundle(path: str | Path) -> LoadedPromptBundle:
    root = Path(path)
    bundle_path = root / BUNDLE_JSON
    prototype_path = root / PROTOTYPES_FILE
    metadata_path = root / METADATA_JSON
    try:
        bundle_bytes = bundle_path.read_bytes()
        prototype_bytes = prototype_path.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        bundle = json.loads(bundle_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PromptBundleError(f"Prompt bundle could not be loaded: {exc}") from exc
    if bundle.get("bundle_version") != BUNDLE_VERSION or metadata.get("bundle_version") != BUNDLE_VERSION:
        raise PromptBundleError("Unsupported Prompt bundle version")
    expected = str(metadata.get("bundle_sha256", ""))
    actual = _bundle_digest(bundle_bytes, prototype_bytes)
    if not expected or actual != expected:
        raise PromptBundleError("Prompt bundle SHA-256 verification failed")
    if hashlib.sha256(prototype_bytes).hexdigest() != bundle.get("prototypes_sha256"):
        raise PromptBundleError("Prompt prototype file SHA-256 verification failed")
    try:
        stored = np.load(prototype_path, allow_pickle=False)
        vectors = stored["features"].astype(np.float32, copy=False)
    except (OSError, ValueError, KeyError) as exc:
        raise PromptBundleError(f"Prompt prototypes could not be decoded: {exc}") from exc
    index = bundle.get("prototype_index")
    expected_count = int(bundle.get("prototype_count", 0))
    base_count = int(bundle.get("base_prototype_count", 0))
    candidate_count = int(bundle.get("live_calibration_candidate_count", 0))
    if (
        not isinstance(index, list)
        or base_count != 35
        or candidate_count not in {1, 2}
        or expected_count != base_count + candidate_count
        or len(index) != expected_count
        or vectors.shape[0] != len(index)
    ):
        raise PromptBundleError("Prompt bundle prototype index/count mismatch")
    training_sessions = tuple(str(item) for item in bundle.get("training_session_ids", []))
    if training_sessions != FORMAL_SESSION_IDS or TRIAL_SESSION_ID in training_sessions:
        raise PromptBundleError("Prompt bundle training session lineage is invalid")
    prototypes = tuple(
        PromptPrototype(
            str(item["prototype_id"]),
            str(item["label"]),
            str(item["source_session"]),
            int(item["source_frame"]),
            vectors[int(item["row"])],
        )
        for item in index
    )
    roi_data = bundle["approved_roi"]
    pixel = roi_data["pixel"]
    resolution = roi_data["reference_resolution"]
    roi = PromptROICandidate(
        "prompt_final_candidate",
        int(pixel[0]), int(pixel[1]), int(pixel[2]), int(pixel[3]),
        reference_width=int(resolution[0]), reference_height=int(resolution[1]),
    )
    model = PrototypePromptModel(
        prototypes=prototypes,
        class_thresholds={label: float(bundle["class_thresholds"][label]) for label in OPERATIONAL_LABELS},
        ambiguity_threshold=float(bundle["ambiguity_margin"]),
        training_sessions=training_sessions,
        calibration_sessions=training_sessions,
        idle_stability_frames=int(bundle["idle_stability_frames"]),
    )
    return LoadedPromptBundle(
        root,
        BUNDLE_VERSION,
        actual,
        PrototypePromptObserver(model, roi),
        model,
        roi,
        bundle,
        metadata,
    )
