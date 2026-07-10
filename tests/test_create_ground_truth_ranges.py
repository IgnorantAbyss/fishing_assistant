from pathlib import Path

import numpy as np
import pytest

from src.replay_ground_truth import load_ground_truth
from src.replay_session import ReplaySession
from tools.create_ground_truth import create_ground_truth_file, resolve_session_path


def _session(root: Path, frame_count: int = 5) -> ReplaySession:
    session = ReplaySession.create(
        root,
        interval_sec=0.2,
        duration_sec=1.0,
        image_format="png",
        screen_size=(8, 8),
        monitor_index=1,
    )
    for _ in range(frame_count):
        session.save_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    return session


def test_complete_ranges_without_get_can_save_ignore(tmp_path: Path) -> None:
    session = _session(tmp_path)
    ranges = [
        {"start": 1, "end": 2, "state": "WAITING"},
        {"start": 3, "end": 3, "state": "IGNORE"},
        {"start": 4, "end": 5, "state": "READY"},
    ]

    destination, segments = create_ground_truth_file(session.path, ranges)

    assert destination.is_file()
    assert segments == ranges
    assert load_ground_truth(destination, 5) == {
        1: "WAITING",
        2: "WAITING",
        3: "IGNORE",
        4: "READY",
        5: "READY",
    }


def test_existing_ground_truth_requires_force_and_force_replaces_it(tmp_path: Path) -> None:
    session = _session(tmp_path)
    first = [{"start": 1, "end": 5, "state": "WAITING"}]
    second = [{"start": 1, "end": 5, "state": "IDLE"}]
    destination, _ = create_ground_truth_file(session.path, first)
    original = destination.read_bytes()

    with pytest.raises(FileExistsError, match="use --force"):
        create_ground_truth_file(session.path, second)
    assert destination.read_bytes() == original

    create_ground_truth_file(session.path, second, force=True)
    assert set(load_ground_truth(destination, 5).values()) == {"IDLE"}


def test_invalid_force_attempt_does_not_damage_existing_file(tmp_path: Path) -> None:
    session = _session(tmp_path)
    destination, _ = create_ground_truth_file(
        session.path, [{"start": 1, "end": 5, "state": "WAITING"}]
    )
    original = destination.read_bytes()

    with pytest.raises(ValueError, match="gap"):
        create_ground_truth_file(
            session.path,
            [
                {"start": 1, "end": 2, "state": "WAITING"},
                {"start": 4, "end": 5, "state": "READY"},
            ],
            force=True,
        )

    assert destination.read_bytes() == original


def test_session_id_resolves_beneath_session_root(tmp_path: Path) -> None:
    assert resolve_session_path("session_example", tmp_path) == tmp_path / "session_example"
