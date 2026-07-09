from pathlib import Path

import numpy as np

from src.config_loader import DebugSettings
from src.debug_saver import DebugSaver


def test_debug_saver_suppresses_repeated_events_and_enforces_count(tmp_path: Path) -> None:
    settings = DebugSettings(True, True, True, True, 2, 10, 2)
    saver = DebugSaver(tmp_path, settings)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    assert saver.save_if_needed(frame, state="WAITING") is None
    assert saver.save_if_needed(frame, state="UNKNOWN") is not None
    assert saver.save_if_needed(frame, state="UNKNOWN") is None
    assert saver.save_if_needed(frame, state="IDLE", save_all=True) is not None
    assert saver.save_if_needed(frame, state="READY", save_all=True) is not None

    assert len(list(tmp_path.glob("*.png"))) == 2
