import pytest

from src.replay_ground_truth import validate_segments


@pytest.mark.parametrize(
    "segments, message",
    [
        ([{"start": 2, "end": 5, "state": "WAITING"}], "gap at frame 1"),
        ([{"start": 1, "end": 4, "state": "WAITING"}], "gap at frame 5"),
        ([{"start": 1, "end": 6, "state": "WAITING"}], "exceeds"),
        ([{"start": 1, "end": 5, "state": "MISSING"}], "Invalid"),
        ([{"start": 3, "end": 2, "state": "WAITING"}], "Invalid"),
    ],
)
def test_invalid_range_shapes_are_rejected(
    segments: list[dict[str, int | str]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_segments(segments, 5)


def test_adjacent_ranges_cover_every_frame_exactly_once() -> None:
    segments = [
        {"start": 1, "end": 2, "state": "WAITING"},
        {"start": 3, "end": 4, "state": "IGNORE"},
        {"start": 5, "end": 5, "state": "READY"},
    ]

    assert validate_segments(segments, 5) == segments
