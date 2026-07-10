from pathlib import Path

import pytest

from src.fishing_v2.data.prompt_annotation import (
    PromptAnnotationKind,
    load_prompt_ground_truth,
    prompt_boundary_frames,
    validate_prompt_segments,
    write_prompt_ground_truth,
)


def _segments():
    return [
        {"start": 1, "end": 3, "observation": "WAITING_PROMPT"},
        {"start": 4, "end": 5, "observation": "READY_PROMPT"},
        {"start": 6, "end": 6, "observation": "IGNORE"},
        {"start": 7, "end": 8, "observation": "OTHER_PROMPT"},
    ]


def test_prompt_specific_ranges_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "prompt_ground_truth.yaml"
    write_prompt_ground_truth(path, _segments(), 8)
    labels = load_prompt_ground_truth(path, 8)
    assert labels[1] == PromptAnnotationKind.WAITING_PROMPT
    assert labels[8] == PromptAnnotationKind.OTHER_PROMPT


def test_prompt_boundary_can_differ_from_global_boundary() -> None:
    labels = {1: PromptAnnotationKind.READY_PROMPT, 2: PromptAnnotationKind.READY_PROMPT, 3: PromptAnnotationKind.NO_PROMPT}
    assert prompt_boundary_frames(labels) == {2, 3}


def test_hook_can_be_annotated_ready_prompt() -> None:
    # No global-state argument exists: visible content alone is accepted.
    result = validate_prompt_segments([{"start": 1, "end": 1, "observation": "READY_PROMPT"}], 1)
    assert result[0]["observation"] == "READY_PROMPT"


def test_press_can_be_annotated_other_prompt() -> None:
    result = validate_prompt_segments([{"start": 1, "end": 1, "observation": "OTHER_PROMPT"}], 1)
    assert result[0]["observation"] == "OTHER_PROMPT"


def test_no_prompt_is_independent_annotation_class() -> None:
    assert PromptAnnotationKind.NO_PROMPT != PromptAnnotationKind.OTHER_PROMPT


def test_ignore_can_cover_transition() -> None:
    result = validate_prompt_segments(_segments(), 8)
    assert result[2] == {"start": 6, "end": 6, "observation": "IGNORE"}


def test_prompt_annotation_rejects_gap() -> None:
    with pytest.raises(ValueError, match="gap"):
        validate_prompt_segments([{"start": 1, "end": 2, "observation": "NO_PROMPT"}, {"start": 4, "end": 4, "observation": "IDLE_PROMPT"}], 4)


def test_prompt_annotation_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_prompt_segments([{"start": 1, "end": 3, "observation": "NO_PROMPT"}, {"start": 3, "end": 4, "observation": "IDLE_PROMPT"}], 4)


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
