from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ml.failure_analysis import (
    assert_hashes_unchanged,
    boundary_analysis,
    boundary_bucket,
    boundary_distance,
    difference_hash,
    find_cross_label_near_duplicates,
    group_confusion_errors,
    image_statistics,
    none_breakdown,
    perceptual_hash,
    protected_file_hashes,
    summarize_session_roi,
    transition_frames,
    uniform_sample,
)
from tools.analyze_prompt_classifier_failures import _write_failure_sheet


def _record(**overrides):
    record = {
        "session_id": "session_a",
        "frame_index": 10,
        "label": "IDLE",
        "original_state": "IDLE",
        "prediction": "READY",
        "confidence": 0.8,
        "probabilities": {"IDLE": 0.1, "WAITING": 0.05, "READY": 0.8, "NONE": 0.05},
        "is_boundary": False,
        "boundary_distance": 12,
    }
    record.update(overrides)
    return record


def test_errors_group_by_confusion_pair() -> None:
    grouped = group_confusion_errors([_record(), _record(label="READY", prediction="READY")])
    assert list(grouped) == ["IDLE_to_READY"]
    assert len(grouped["IDLE_to_READY"]) == 1


def test_none_breakdown_uses_original_global_state() -> None:
    records = [
        _record(label="NONE", original_state="HOOK", prediction="READY", is_boundary=True),
        _record(label="NONE", original_state="HOOK", prediction="NONE"),
        _record(label="NONE", original_state="PRESS", prediction="UNKNOWN"),
    ]
    result = none_breakdown(records)
    assert result["HOOK"]["total_samples"] == 2
    assert result["HOOK"]["predicted_READY"] == 1
    assert result["HOOK"]["boundary_errors"] == 1
    assert result["PRESS"]["predicted_UNKNOWN"] == 1
    assert result["GET"]["total_samples"] == 0


def test_transition_frames_and_boundary_distance_are_exact() -> None:
    segments = [
        {"start": 1, "end": 10, "state": "WAITING"},
        {"start": 11, "end": 20, "state": "READY"},
        {"start": 21, "end": 30, "state": "HOOK"},
    ]
    transitions = transition_frames(segments)
    assert transitions == [11, 21]
    assert boundary_distance(9, transitions) == 2
    assert boundary_distance(21, transitions) == 0


@pytest.mark.parametrize(("distance", "expected"), [(0, "0-3"), (3, "0-3"), (4, "4-5"), (6, "6-10"), (11, ">10")])
def test_boundary_distance_buckets(distance: int, expected: str) -> None:
    assert boundary_bucket(distance) == expected


def test_boundary_analysis_separates_boundary_and_interior() -> None:
    records = [
        _record(is_boundary=True, boundary_distance=1, nearest_transition="READY_to_IDLE"),
        _record(is_boundary=False, boundary_distance=12, nearest_transition="IDLE_to_WAITING"),
        _record(label="READY", prediction="READY", is_boundary=False),
    ]
    result = boundary_analysis(records)
    assert result["IDLE_to_READY"]["boundary"] == 1
    assert result["IDLE_to_READY"]["non_boundary"] == 1
    assert result["IDLE_to_READY"]["transition_counts_by_distance_bucket"]["0-3"] == {
        "READY_to_IDLE": 1
    }
    assert result["error_rates"]["boundary"]["rate"] == 1.0
    assert result["error_rates"]["non_boundary"]["rate"] == 0.5


def test_session_roi_statistics_are_computed() -> None:
    image = np.full((20, 40, 3), 80, dtype=np.uint8)
    item = {
        "stats": image_statistics(image),
        "source_width": 100,
        "source_height": 50,
        "crop_out_of_frame": False,
        "label": "IDLE",
    }
    result = summarize_session_roi([item], (0.3, 0.02, 0.7, 0.1))
    assert result["crop_widths"] == [40]
    assert result["source_widths"] == [100]
    assert result["crop_out_of_frame_count"] == 0
    assert result["manual_review_required"] is True


def test_perceptual_and_difference_hashes_are_reproducible() -> None:
    image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    assert perceptual_hash(bgr) == perceptual_hash(bgr.copy())
    assert difference_hash(bgr) == difference_hash(bgr.copy())


def test_near_duplicate_search_is_deterministic() -> None:
    common = {
        "original_state": "READY", "is_boundary": False,
        "crop_path": "crop.jpg", "dhash": 0,
    }
    records = [
        {**common, "session_id": "a", "frame_index": 1, "label": "READY", "phash": 0},
        {**common, "session_id": "b", "frame_index": 2, "label": "NONE", "original_state": "HOOK", "phash": 1},
        {**common, "session_id": "c", "frame_index": 3, "label": "IDLE", "original_state": "IDLE", "phash": 2},
    ]
    first = find_cross_label_near_duplicates(records, maximum=10)
    second = find_cross_label_near_duplicates(records, maximum=10)
    assert first == second
    assert first[0]["phash_distance"] == 1


def test_uniform_sample_spans_entire_session() -> None:
    selected = uniform_sample(list(range(100)), 5)
    assert selected[0] == 0
    assert selected[-1] == 99
    assert len(selected) == 5


def test_failure_contact_sheet_is_written(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    crop = tmp_path / "crop.jpg"
    cv2.imwrite(str(source), np.full((90, 160, 3), 70, dtype=np.uint8))
    cv2.imwrite(str(crop), np.full((30, 120, 3), 180, dtype=np.uint8))
    record = _record(source_frame=str(source), crop_file=str(crop))
    destination = tmp_path / "sheet.jpg"
    _write_failure_sheet([record], destination)
    assert destination.is_file()
    assert cv2.imread(str(destination)) is not None


def test_contact_sheet_does_not_modify_protected_inputs(tmp_path: Path) -> None:
    protected = tmp_path / "manifest.csv"
    protected.write_text("immutable", encoding="utf-8")
    before = protected_file_hashes([protected])
    destination = tmp_path / "empty.jpg"
    _write_failure_sheet([], destination)
    assert_hashes_unchanged(before)


def test_hash_guard_detects_changes(tmp_path: Path) -> None:
    protected = tmp_path / "model.pt"
    protected.write_bytes(b"before")
    before = protected_file_hashes([protected])
    protected.write_bytes(b"after")
    with pytest.raises(RuntimeError, match="Protected"):
        assert_hashes_unchanged(before)


def test_analysis_tool_does_not_import_training_pipeline() -> None:
    source = (Path(__file__).resolve().parents[1] / "tools" / "analyze_prompt_classifier_failures.py").read_text(encoding="utf-8")
    assert "train_prompt_classifier" not in source
    assert "optimizer" not in source
    assert 'torch.device("cpu")' in source
