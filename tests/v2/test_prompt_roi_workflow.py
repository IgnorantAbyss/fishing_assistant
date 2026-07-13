import json
from pathlib import Path

import numpy as np
import pytest
import yaml
import cv2

from src.fishing_v2.data.prompt_roi import PromptROICandidate, load_roi_candidates
from src.fishing_v2.data.runtime_constraints import (
    UnsupportedRuntimeEnvironmentError,
    load_runtime_constraints,
    require_supported_frame,
    validate_frame_environment,
)
from tools.review_prompt_roi import (
    _configured_session_ids,
    _load_session,
    select_stable_interior,
    select_transition_windows,
    with_metadata_margin,
    review_prompt_roi,
    review_final_candidate,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "fishing_v2.yaml"
CANDIDATES = ROOT / "config" / "prompt_roi_candidates.yaml"


def _review_session(root: Path, name: str, segments: list[dict]) -> Path:
    path = root / name
    (path / "frames").mkdir(parents=True)
    frame_count = int(segments[-1]["end"])
    (path / "manifest.json").write_text(json.dumps({
        "session_id": name,
        "frame_count": frame_count,
        "image_format": "jpg",
        "screen_size": [200, 100],
    }), encoding="utf-8")
    (path / "ground_truth.yaml").write_text(
        yaml.safe_dump({"segments": segments}, sort_keys=False), encoding="utf-8"
    )
    return path


def test_runtime_environment_is_fixed_to_supported_configuration() -> None:
    constraints = load_runtime_constraints(CONFIG)
    assert (constraints.width, constraints.height) == (2560, 1440)
    assert (constraints.ui_scale, constraints.language) == ("fixed", "zh-TW")
    assert (constraints.window_mode, constraints.prompt_position) == ("borderless", "fixed")
    assert constraints.supported_environment_only is True


def test_unsupported_resolution_is_rejected_before_future_runtime_use() -> None:
    constraints = load_runtime_constraints(CONFIG)
    result = validate_frame_environment(1920, 1080, constraints)
    assert result.supported is False
    assert result.violations
    with pytest.raises(UnsupportedRuntimeEnvironmentError, match="unsupported_resolution"):
        require_supported_frame(1920, 1080, constraints)


def test_prompt_roi_candidates_use_pixel_source_of_truth() -> None:
    raw = yaml.safe_load(CANDIDATES.read_text(encoding="utf-8"))
    assert raw["source_of_truth"] == "pixel"
    assert all(("pixel" in item) ^ ("pixel_roi" in item) for item in raw["candidates"])


def test_normalized_roi_is_derived_from_fixed_pixels() -> None:
    for candidate in load_roi_candidates(CANDIDATES):
        x1, y1, x2, y2 = candidate.normalized
        assert x1 == pytest.approx(candidate.x1 / 2560)
        assert y1 == pytest.approx(candidate.y1 / 1440)
        assert x2 == pytest.approx(candidate.x2 / 2560)
        assert y2 == pytest.approx(candidate.y2 / 1440)


def test_all_candidates_including_final_are_inside_fixed_frame() -> None:
    candidates = load_roi_candidates(CANDIDATES)
    assert [item.candidate_id for item in candidates] == [
        "legacy_reference", "tight_vertical", "icon_and_text_medium",
        "icon_and_text_tight", "text_only_medium", "text_only_tight",
        "prompt_final_candidate",
    ]
    assert all(0 <= item.x1 < item.x2 <= 2560 and 0 <= item.y1 < item.y2 <= 1440 for item in candidates)


def test_legacy_roi_is_preserved_as_reference() -> None:
    legacy = load_roi_candidates(CANDIDATES)[0]
    assert legacy.candidate_id == "legacy_reference"
    assert legacy.pixel == (768, 29, 1792, 144)
    assert (legacy.width, legacy.height, legacy.area) == (1024, 115, 117760)


def test_final_candidate_pixel_and_derived_geometry() -> None:
    candidate = next(item for item in load_roi_candidates(CANDIDATES) if item.candidate_id == "prompt_final_candidate")
    assert candidate.pixel == (940, 36, 1620, 100)
    assert (candidate.width, candidate.height, candidate.area) == (680, 64, 43520)
    assert candidate.normalized == pytest.approx((0.3671875, 0.025, 0.6328125, 0.0694444444))


def test_final_candidate_is_explicitly_approved_in_runtime_config() -> None:
    raw = yaml.safe_load(CANDIDATES.read_text(encoding="utf-8"))
    assert raw["manual_approval_required"] is True
    assert raw["roi_status"] == "approved"
    assert raw["approved_candidate_id"] == "prompt_final_candidate"
    prompt = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["prompt"]
    assert prompt["roi_status"] == "approved"
    assert prompt["approved_candidate_id"] == "prompt_final_candidate"
    assert prompt["roi"] == {"x1": 940, "y1": 36, "x2": 1620, "y2": 100}


def test_review_labels_are_appended_outside_prompt_content() -> None:
    content = np.full((24, 40, 3), 173, dtype=np.uint8)
    labelled = with_metadata_margin(content, ["metadata"])
    assert np.array_equal(labelled[:24], content)
    assert labelled.shape[0] > content.shape[0]


def test_stable_interior_sampling_is_stratified_per_session_and_state(tmp_path: Path) -> None:
    first = _review_session(tmp_path, "session_one", [
        {"start": 1, "end": 30, "state": "IDLE"},
        {"start": 31, "end": 60, "state": "WAITING"},
    ])
    second = _review_session(tmp_path, "session_two", [
        {"start": 1, "end": 30, "state": "READY"},
        {"start": 31, "end": 60, "state": "HOOK"},
    ])
    one = select_stable_interior(_load_session(first), samples_per_state=8)
    two = select_stable_interior(_load_session(second), samples_per_state=8)
    assert 0 < len(one["IDLE"]) <= 8 and 0 < len(one["WAITING"]) <= 8
    assert 0 < len(two["READY"]) <= 8 and 0 < len(two["HOOK"]) <= 8


def test_transition_windows_include_ten_frames_on_each_side(tmp_path: Path) -> None:
    session = _review_session(tmp_path, "session_transition", [
        {"start": 1, "end": 20, "state": "WAITING"},
        {"start": 21, "end": 40, "state": "READY"},
    ])
    windows = select_transition_windows(_load_session(session))
    assert len(windows) == 1
    assert windows[0].boundary_frame == 21
    assert [sample.frame_index for sample in windows[0].samples] == list(range(11, 31))


def test_transition_counts_are_accumulated_in_review_report(tmp_path: Path) -> None:
    session = _review_session(tmp_path, "session_transition_report", [
        {"start": 1, "end": 20, "state": "WAITING"},
        {"start": 21, "end": 40, "state": "READY"},
    ])
    for index in range(1, 41):
        cv2.imwrite(str(session / "frames" / f"{index:06d}.jpg"), np.full((100, 200, 3), index, dtype=np.uint8))
    candidate = PromptROICandidate("test", 20, 5, 180, 30, reference_width=200, reference_height=100)
    report = review_prompt_roi([session], [candidate], tmp_path / "review", samples_per_state=1)
    assert report["transition_windows"]["window_counts_by_transition"] == {"WAITING_to_READY": 1}
    assert report["transition_windows"]["sample_counts_by_transition"] == {"WAITING_to_READY": 20}


def test_final_candidate_writes_only_one_compact_review_sheet(tmp_path: Path) -> None:
    session = _review_session(tmp_path, "session_compact", [
        {"start": 1, "end": 12, "state": "IDLE"},
        {"start": 13, "end": 24, "state": "WAITING"},
        {"start": 25, "end": 36, "state": "READY"},
        {"start": 37, "end": 48, "state": "HOOK"},
        {"start": 49, "end": 60, "state": "PRESS"},
    ])
    for index in range(1, 61):
        cv2.imwrite(str(session / "frames" / f"{index:06d}.jpg"), np.full((100, 200, 3), index, dtype=np.uint8))
    candidate = PromptROICandidate("prompt_final_candidate", 20, 5, 180, 30, reference_width=200, reference_height=100)
    report = review_final_candidate([session], candidate, tmp_path / "review")
    assert (tmp_path / "review" / "prompt_final_candidate_review.jpg").is_file()
    assert (tmp_path / "review" / "prompt_final_candidate_review.md").is_file()
    assert list((tmp_path / "review").glob("*.jpg")) == [tmp_path / "review" / "prompt_final_candidate_review.jpg"]
    assert report["roi_status"] == "unapproved"


def test_trial_session_is_explicitly_excluded_from_all_review() -> None:
    included, excluded = _configured_session_ids(CONFIG)
    assert "session_20260709_192231" in excluded
    assert "session_20260709_192231" not in included
    assert len(included) == 7


def test_runtime_modules_do_not_consume_prompt_id() -> None:
    runtime_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src" / "fishing_v2" / "runtime").glob("*.py")
    )
    assert "prompt_id" not in runtime_source


def test_all_formal_prompt_ground_truth_exists() -> None:
    prompt_gt = list((ROOT / "assets" / "replay" / "sessions").glob("session_*/prompt_ground_truth.yaml"))
    actual_sessions = {path.parent.name for path in prompt_gt}
    expected_sessions = {
        "session_20260709_192315",
        "session_20260710_061220",
        "session_20260710_123210",
        "session_20260710_124419",
        "session_20260710_125441",
        "session_20260710_130308",
        "session_20260710_131254",
    }
    assert actual_sessions == expected_sessions
    assert "session_20260709_192231" not in actual_sessions


def test_prototype_phase_tracks_only_small_dataset_metadata() -> None:
    dataset = ROOT / "datasets" / "prompt_observation_v1"
    assert (dataset / "manifest.csv").is_file()
    assert (dataset / "dataset_summary.json").is_file()
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "datasets/prompt_observation_v1/*" in ignore


def test_review_tool_has_no_training_or_classifier_selection_path() -> None:
    source = (ROOT / "tools" / "review_prompt_roi.py").read_text(encoding="utf-8")
    assert "src.ml" not in source
    assert "PromptClassifier" not in source
    assert "bright_mask_used_for_approval" in source
    assert '"classifier_accuracy_used_for_selection": False' in source


def test_v2_has_no_keyboard_input_implementation() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src" / "fishing_v2").rglob("*.py")
    ).lower()
    assert "pyautogui" not in source
    assert "pydirectinput" not in source
    assert "keyboard.press" not in source
