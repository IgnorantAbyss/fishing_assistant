from pathlib import Path

import pytest

from src.fishing_v2.domain.observations import PromptObservationKind

from src.fishing_v2.data.prompt_annotation import (
    DEPRECATED_PROMPT_ANNOTATION_KINDS,
    FINAL_PROMPT_ANNOTATION_KINDS,
    PromptAnnotationKind,
    load_prompt_annotations,
    load_prompt_ground_truth,
    prompt_boundary_frames,
    validate_prompt_segments,
    write_prompt_ground_truth,
)


def _segments():
    return [
        {"start": 1, "end": 3, "observation": "WAITING_IN_PROGRESS"},
        {"start": 4, "end": 5, "observation": "READY_BITE"},
        {"start": 6, "end": 6, "observation": "IGNORE"},
        {"start": 7, "end": 8, "observation": "PRESS_INSTRUCTION"},
    ]


def test_prompt_specific_ranges_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "prompt_ground_truth.yaml"
    write_prompt_ground_truth(path, _segments(), 8)
    labels = load_prompt_ground_truth(path, 8)
    assert labels[1] == PromptAnnotationKind.WAITING_IN_PROGRESS
    assert labels[8] == PromptAnnotationKind.PRESS_INSTRUCTION


def test_prompt_boundary_can_differ_from_global_boundary() -> None:
    labels = {1: PromptAnnotationKind.READY_BITE, 2: PromptAnnotationKind.READY_BITE, 3: PromptAnnotationKind.HOOK_INSTRUCTION}
    assert prompt_boundary_frames(labels) == {2, 3}


def test_hook_can_be_annotated_ready_bite_from_visible_content() -> None:
    # No global-state argument exists: visible content alone is accepted.
    result = validate_prompt_segments([{"start": 1, "end": 1, "observation": "READY_BITE"}], 1)
    assert result[0]["observation"] == "READY_BITE"


def test_press_can_be_annotated_other_prompt() -> None:
    result = validate_prompt_segments([{"start": 1, "end": 1, "observation": "PRESS_INSTRUCTION"}], 1)
    assert result[0]["observation"] == "PRESS_INSTRUCTION"


def test_ignore_is_annotation_only_and_deprecated_targets_are_rejected_for_new_writes() -> None:
    assert not hasattr(PromptObservationKind, "IGNORE")
    with pytest.raises(ValueError, match="Deprecated"):
        validate_prompt_segments([{"start": 1, "end": 1, "observation": "NO_PROMPT"}], 1)


def test_other_and_no_prompt_are_not_final_classifier_targets() -> None:
    assert PromptAnnotationKind.OTHER_PROMPT in DEPRECATED_PROMPT_ANNOTATION_KINDS
    assert PromptAnnotationKind.NO_PROMPT in DEPRECATED_PROMPT_ANNOTATION_KINDS
    assert PromptAnnotationKind.OTHER_PROMPT not in FINAL_PROMPT_ANNOTATION_KINDS
    assert PromptAnnotationKind.NO_PROMPT not in FINAL_PROMPT_ANNOTATION_KINDS


def test_ignore_can_cover_transition() -> None:
    result = validate_prompt_segments(_segments(), 8)
    assert result[2] == {"start": 6, "end": 6, "observation": "IGNORE"}


def test_prompt_annotation_rejects_gap() -> None:
    with pytest.raises(ValueError, match="gap"):
        validate_prompt_segments([{"start": 1, "end": 2, "observation": "WAITING_IN_PROGRESS"}, {"start": 4, "end": 4, "observation": "IDLE_CAST"}], 4)


def test_prompt_annotation_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_prompt_segments([{"start": 1, "end": 3, "observation": "WAITING_IN_PROGRESS"}, {"start": 3, "end": 4, "observation": "IDLE_CAST"}], 4)


def test_write_does_not_modify_global_ground_truth(tmp_path: Path) -> None:
    global_path = tmp_path / "ground_truth.yaml"
    global_path.write_text("segments:\n- start: 1\n  end: 8\n  state: HOOK\n", encoding="utf-8")
    before = global_path.read_bytes()
    write_prompt_ground_truth(tmp_path / "prompt_ground_truth.yaml", _segments(), 8)
    assert global_path.read_bytes() == before


def test_existing_prompt_annotation_requires_force(tmp_path: Path) -> None:
    path = tmp_path / "prompt_ground_truth.yaml"
    write_prompt_ground_truth(path, _segments(), 8)
    with pytest.raises(FileExistsError):
        write_prompt_ground_truth(path, _segments(), 8)
    write_prompt_ground_truth(path, _segments(), 8, force=True)


def test_prompt_id_and_notes_are_optional_dataset_metadata(tmp_path: Path) -> None:
    path = tmp_path / "prompt_ground_truth.yaml"
    segments = [
        {"start": 1, "end": 1, "observation": "WAITING_IN_PROGRESS", "prompt_id": "WAITING_IN_PROGRESS", "notes": "manual"},
        {"start": 2, "end": 2, "observation": "HOOK_INSTRUCTION", "prompt_id": None},
        {"start": 3, "end": 3, "observation": "IGNORE"},
    ]
    write_prompt_ground_truth(path, segments, 3)
    annotations = load_prompt_annotations(path, 3)
    assert annotations[1].prompt_id == "WAITING_IN_PROGRESS"
    assert annotations[1].notes == "manual"
    assert annotations[2].prompt_id is None
    assert annotations[3].prompt_id is None


def test_old_prompt_yaml_without_prompt_id_remains_readable(tmp_path: Path) -> None:
    path = tmp_path / "prompt_ground_truth.yaml"
    path.write_text("version: 1\nsegments:\n- start: 1\n  end: 1\n  observation: IDLE_PROMPT\n", encoding="utf-8")
    assert load_prompt_annotations(path, 1)[1].prompt_id is None
