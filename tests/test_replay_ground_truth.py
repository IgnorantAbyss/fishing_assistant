from pathlib import Path

import pytest
import yaml

from src.replay_ground_truth import load_annotations, validate_segments
from tools.create_ground_truth import _parse_range


COMPLETE_SEGMENTS = [
    {"start": 1, "end": 451, "state": "WAITING"},
    {"start": 452, "end": 479, "state": "READY"},
    {"start": 480, "end": 506, "state": "HOOK"},
    {"start": 507, "end": 517, "state": "PRESS"},
    {"start": 518, "end": 529, "state": "IGNORE"},
    {"start": 530, "end": 558, "state": "IDLE"},
    {"start": 559, "end": 600, "state": "WAITING"},
]


def test_ignore_is_legal_and_complete_coverage_without_get_succeeds() -> None:
    assert _parse_range("518-529:ignore") == {"start": 518, "end": 529, "state": "IGNORE"}
    assert validate_segments(COMPLETE_SEGMENTS, 600) == COMPLETE_SEGMENTS
    assert all(segment["state"] != "GET" for segment in COMPLETE_SEGMENTS)


@pytest.mark.parametrize(
    "segments, message",
    [
        ([{"start": 1, "end": 4, "state": "WAITING"}, {"start": 6, "end": 10, "state": "IDLE"}], "gap"),
        ([{"start": 1, "end": 6, "state": "WAITING"}, {"start": 6, "end": 10, "state": "IDLE"}], "overlap"),
    ],
)
def test_incomplete_or_overlapping_coverage_fails(
    segments: list[dict[str, int | str]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_segments(segments, 10)


def test_hook_bar_annotation_may_start_after_top_level_hook(tmp_path: Path) -> None:
    path = tmp_path / "annotations.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "events": {
                    "hook_state": {"start": 480, "end": 506},
                    "hook_bar_visible": {"start": 486, "end": 506},
                },
                "notes": {"get_skipped": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    annotations = load_annotations(path, 600)

    assert annotations["events"]["hook_state"]["start"] == 480
    assert annotations["events"]["hook_bar_visible"]["start"] == 486
