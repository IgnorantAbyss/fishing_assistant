import json

import numpy as np

from src.config_loader import ReplayRetentionSettings
from src.replay_session import ReplaySession, apply_retention


def test_replay_session_saves_synthetic_frame_and_manifest(tmp_path) -> None:
    session = ReplaySession.create(
        tmp_path,
        interval_sec=0.2,
        duration_sec=1.0,
        image_format="png",
        screen_size=(32, 24),
        monitor_index=1,
        notes="synthetic test",
    )
    frame_path = session.save_frame(np.full((24, 32, 3), 127, dtype=np.uint8))
    session.add_marker("manual-note", 0.2)

    manifest = json.loads((session.path / "manifest.json").read_text(encoding="utf-8"))
    assert frame_path.is_file()
    assert manifest["frame_count"] == 1
    assert manifest["screen_size"] == [32, 24]
    assert manifest["markers"] == [{"label": "manual-note", "timestamp_sec": 0.2}]
    assert ReplaySession.load(session.path).frame_paths() == [frame_path]


def test_retention_never_deletes_current_session(tmp_path) -> None:
    old_session = ReplaySession.create(
        tmp_path, interval_sec=0.2, duration_sec=1.0, image_format="png", screen_size=(8, 8), monitor_index=1
    )
    current_session = ReplaySession.create(
        tmp_path, interval_sec=0.2, duration_sec=1.0, image_format="png", screen_size=(8, 8), monitor_index=1
    )
    deleted = apply_retention(
        tmp_path,
        ReplayRetentionSettings(max_sessions=1, max_total_size_mb=10, max_age_days=7),
        keep_session=current_session.path,
    )

    assert current_session.path.is_dir()
    assert old_session.path in deleted
    assert not old_session.path.exists()
