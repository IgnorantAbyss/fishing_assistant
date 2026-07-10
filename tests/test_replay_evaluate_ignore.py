import csv
from pathlib import Path

import cv2
import yaml

from src.replay_session import ReplaySession
from tools.replay_evaluate import evaluate_session


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "assets" / "reference"


def test_ignore_is_excluded_from_metrics_and_confusion_without_get_support(tmp_path: Path) -> None:
    session = ReplaySession.create(
        tmp_path,
        interval_sec=0.2,
        duration_sec=0.6,
        image_format="png",
        screen_size=(2048, 1151),
        monitor_index=1,
        notes="IGNORE evaluator test",
    )
    for filename in ("waiting.png", "get.png", "idle.png"):
        image = cv2.imread(str(REFERENCE_DIR / filename), cv2.IMREAD_COLOR)
        assert image is not None
        session.save_frame(image)
    (session.path / "ground_truth.yaml").write_text(
        yaml.safe_dump(
            {
                "segments": [
                    {"start": 1, "end": 1, "state": "WAITING"},
                    {"start": 2, "end": 2, "state": "IGNORE"},
                    {"start": 3, "end": 3, "state": "IDLE"},
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    summary = evaluate_session(session.path, export_debug=False)
    with summary.results_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    confusion = (session.path / "confusion_matrix.csv").read_text(encoding="utf-8")
    report = summary.report_path.read_text(encoding="utf-8")

    assert (summary.total_frames, summary.evaluated_frames, summary.ignored_frames) == (3, 2, 1)
    assert summary.raw_accuracy == 1.0
    assert summary.smoothed_accuracy == 1.0
    assert rows[1]["expected_state"] == "IGNORE"
    assert rows[1]["is_evaluated"] == "False"
    assert rows[1]["is_correct_raw"] == ""
    assert "IGNORE" not in confusion
    assert "| GET | 0 | N/A | N/A | N/A | N/A | N/A | N/A |" in report
