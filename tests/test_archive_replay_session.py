from pathlib import Path
import zipfile

import cv2
import pytest

from src.replay_ground_truth import write_ground_truth
from src.replay_session import ReplaySession
from tools.archive_replay_session import archive_replay_session


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "assets" / "reference"


def _completed_session(root: Path) -> ReplaySession:
    session = ReplaySession.create(
        root,
        interval_sec=0.2,
        duration_sec=1.0,
        image_format="png",
        screen_size=(2048, 1151),
        monitor_index=1,
    )
    frame = cv2.imread(str(REFERENCE_DIR / "waiting.png"), cv2.IMREAD_COLOR)
    assert frame is not None
    session.save_frame(frame)
    write_ground_truth(
        session.path / "ground_truth.yaml",
        [{"start": 1, "end": 1, "state": "WAITING"}],
        1,
    )
    return session


def test_archive_validation_failure_never_deletes_raw_frames(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    session = _completed_session(root)
    ReplaySession.create(
        root,
        interval_sec=0.2,
        duration_sec=1.0,
        image_format="png",
        screen_size=(8, 8),
        monitor_index=1,
    )
    frame_path = session.frame_paths()[0]

    with pytest.raises(ValueError, match="Dataset validation failed"):
        archive_replay_session(
            session.path,
            dataset_output_dir=tmp_path / "missing_dataset",
            session_root=root,
            dry_run=False,
            delete_frames=True,
        )

    assert frame_path.is_file()
    assert not (session.path / "session_archive_manifest.json").exists()


def test_archive_dry_run_does_not_create_zip_or_manifest(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    session = _completed_session(root)
    archive_root = tmp_path / "archives"

    result = archive_replay_session(
        session.path,
        dataset_output_dir=tmp_path / "dataset",
        session_root=root,
        archive_root=archive_root,
        dry_run=True,
        create_zip=True,
    )

    assert result.dry_run is True
    assert result.frames_deleted == 0
    assert session.frame_paths()[0].is_file()
    assert not archive_root.exists()
    assert not (session.path / "session_archive_manifest.json").exists()


def test_zip_archive_is_standard_and_reopenable(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    session = _completed_session(root)
    archive_root = tmp_path / "archives"

    result = archive_replay_session(
        session.path,
        dataset_output_dir=tmp_path / "dataset",
        session_root=root,
        archive_root=archive_root,
        dry_run=False,
        create_zip=True,
    )

    assert result.zip_path is not None and result.zip_path.is_file()
    with zipfile.ZipFile(result.zip_path, "r") as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == sum(path.is_file() for path in session.path.rglob("*"))
