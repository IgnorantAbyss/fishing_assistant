import csv
from pathlib import Path

import cv2

from src.replay_session import ReplaySession
from src.state_detector import expected_reference_images
from tools.replay_detector import run_replay_detector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "assets" / "reference"


def test_replay_detector_writes_results_for_reference_session(tmp_path) -> None:
    session = ReplaySession.create(
        tmp_path,
        interval_sec=0.2,
        duration_sec=3.0,
        image_format="png",
        screen_size=(2559, 1439),
        monitor_index=1,
        notes="reference replay",
    )
    reference_images = list(expected_reference_images(REFERENCE_DIR))
    expected_states = {state for _, state in reference_images}
    for image_path, _ in reference_images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        assert image is not None
        session.save_frame(image)

    run = run_replay_detector(session.path)
    with run.results_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert run.results_path.is_file()
    assert run.report_path.is_file()
    assert len(rows) == len(reference_images)
    assert expected_states <= {row["state"] for row in rows}
    assert any(row["press_sequence_text"] == "DSDSASSD" for row in rows)
    assert any(row["hook_fill_ratio"] for row in rows)
