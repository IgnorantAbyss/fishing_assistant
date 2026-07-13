import hashlib
from pathlib import Path

import numpy as np

from src.fishing_v2.data.prompt_observation_dataset import (
    FORMAL_SESSION_IDS,
    TRIAL_SESSION_ID,
    read_prompt_observation_manifest,
    validate_cross_label_hashes,
)
from src.fishing_v2.perception.prototype_prompt_observer import (
    OPERATIONAL_LABELS,
    PromptPrototype,
    PrototypePromptModel,
    build_prototypes,
    fit_prototype_model,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "prompt_observation_v1"


def test_manifest_contains_exactly_seven_formal_sessions_and_fixed_roi() -> None:
    rows = read_prompt_observation_manifest(DATASET / "manifest.csv")
    assert len(rows) == 4200
    assert {row["session_id"] for row in rows} == set(FORMAL_SESSION_IDS)
    assert TRIAL_SESSION_ID not in {row["session_id"] for row in rows}
    assert all(sum(row["session_id"] == session for row in rows) == 600 for session in FORMAL_SESSION_IDS)
    assert {
        (row["roi_x1"], row["roi_y1"], row["roi_x2"], row["roi_y2"])
        for row in rows
    } == {("940", "36", "1620", "100")}
    assert "IGNORE" in {row["label"] for row in rows}


def test_manifest_hash_matches_materialized_roi_when_available() -> None:
    rows = read_prompt_observation_manifest(DATASET / "manifest.csv")
    checked = 0
    for row in rows[::421]:
        crop = DATASET / row["roi_path"]
        if crop.is_file():
            assert hashlib.sha256(crop.read_bytes()).hexdigest() == row["image_sha256"]
            checked += 1
    assert checked in {0, 10}  # Bulk crops are intentionally optional/ignored.


def test_cross_non_ignore_label_hash_conflict_fails() -> None:
    rows = [
        {"session_id": "a", "frame_index": "1", "label": "IDLE_CAST", "image_sha256": "same"},
        {"session_id": "b", "frame_index": "2", "label": "READY_BITE", "image_sha256": "same"},
    ]
    try:
        validate_cross_label_hashes(rows)
    except ValueError as exc:
        assert "Cross-label duplicate" in str(exc)
    else:
        raise AssertionError("Cross-label duplicate must fail")


def _synthetic_training():
    sessions = ("train_a", "train_b", "held_out")
    rows = []
    features = []
    for session in sessions:
        for index, label in enumerate(OPERATIONAL_LABELS):
            rows.append({"session_id": session, "frame_index": str(index + 1), "label": label})
            vector = np.zeros(len(OPERATIONAL_LABELS) + 1, dtype=np.float32)
            vector[index] = 1.0
            features.append(vector)
        rows.append({"session_id": session, "frame_index": "6", "label": "IGNORE"})
        vector = np.zeros(len(OPERATIONAL_LABELS) + 1, dtype=np.float32)
        vector[-1] = 1.0
        features.append(vector)
    return rows, np.stack(features), sessions


def test_ignore_never_becomes_prototype_and_held_out_is_isolated() -> None:
    rows, features, sessions = _synthetic_training()
    prototypes = build_prototypes(rows, features, sessions[:2])
    assert {item.label for item in prototypes} == set(OPERATIONAL_LABELS)
    assert all(item.source_session != sessions[2] for item in prototypes)
    model = fit_prototype_model(rows, features, sessions[:2])
    assert sessions[2] not in model.training_sessions
    assert sessions[2] not in model.calibration_sessions
    assert set(model.calibration_sessions) == set(sessions[:2])


def _basis_model(*, threshold: float, margin: float) -> PrototypePromptModel:
    prototypes = tuple(
        PromptPrototype(f"p{index}", label, "train", index, np.eye(5, dtype=np.float32)[index])
        for index, label in enumerate(OPERATIONAL_LABELS)
    )
    return PrototypePromptModel(
        prototypes,
        {label: threshold for label in OPERATIONAL_LABELS},
        margin,
        ("train",),
        ("train",),
    )


def test_low_similarity_and_ambiguous_features_are_unknown() -> None:
    low = _basis_model(threshold=0.8, margin=0.0).predict_feature(np.ones(5, dtype=np.float32))
    assert low.predicted_label == "UNKNOWN"
    assert low.rejection_reason == "low_similarity"
    ambiguous_feature = np.asarray([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ambiguous = _basis_model(threshold=0.5, margin=0.1).predict_feature(ambiguous_feature)
    assert ambiguous.predicted_label == "UNKNOWN"
    assert ambiguous.rejection_reason == "ambiguous_top_two"
