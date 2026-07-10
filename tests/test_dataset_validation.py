from pathlib import Path

import cv2
import pytest
import yaml

from src.dataset.roi_exporter import PROMPT_LABEL_MAP, SPECIAL_LABEL_MAP
from src.dataset.session_scanner import SessionScanError, scan_session
from src.dataset.validation import validate_dataset
from src.replay_ground_truth import write_ground_truth
from src.replay_session import ReplaySession
from tools.annotate_ground_truth import markers_to_segments
from tools.build_training_dataset import build_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "assets" / "reference"


def _session(root: Path, frame_count: int = 3) -> ReplaySession:
    session = ReplaySession.create(
        root,
        interval_sec=0.2,
        duration_sec=1.0,
        image_format="png",
        screen_size=(2048, 1151),
        monitor_index=1,
    )
    image = cv2.imread(str(REFERENCE_DIR / "waiting.png"), cv2.IMREAD_COLOR)
    assert image is not None
    for _ in range(frame_count):
        session.save_frame(image)
    return session


@pytest.mark.parametrize(
    "segments, message",
    [
        (
            [
                {"start": 1, "end": 1, "state": "WAITING"},
                {"start": 3, "end": 3, "state": "READY"},
            ],
            "gap",
        ),
        (
            [
                {"start": 1, "end": 2, "state": "WAITING"},
                {"start": 2, "end": 3, "state": "READY"},
            ],
            "overlap",
        ),
    ],
)
def test_scanner_rejects_ground_truth_gap_or_overlap(
    tmp_path: Path, segments: list[dict[str, int | str]], message: str
) -> None:
    session = _session(tmp_path)
    (session.path / "ground_truth.yaml").write_text(
        yaml.safe_dump({"segments": segments}, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(SessionScanError, match=message):
        scan_session(session.path)


def test_prompt_and_special_label_mappings_are_explicit() -> None:
    assert PROMPT_LABEL_MAP == {
        "IDLE": "IDLE",
        "WAITING": "WAITING",
        "READY": "READY",
        "HOOK": "NONE",
        "PRESS": "NONE",
        "GET": "NONE",
        "IGNORE": None,
    }
    assert SPECIAL_LABEL_MAP == {
        "IDLE": "NONE",
        "WAITING": "NONE",
        "READY": "NONE",
        "HOOK": "HOOK",
        "PRESS": "PRESS",
        "GET": "GET",
        "IGNORE": None,
    }


def test_marker_transitions_create_complete_non_overlapping_segments() -> None:
    assert markers_to_segments([(1, "WAITING"), (5, "HOOK"), (8, "IGNORE")], 10) == [
        {"start": 1, "end": 4, "state": "WAITING"},
        {"start": 5, "end": 7, "state": "HOOK"},
        {"start": 8, "end": 10, "state": "IGNORE"},
    ]


def test_dataset_dry_run_does_not_modify_files(tmp_path: Path) -> None:
    session_root = tmp_path / "sessions"
    session = _session(session_root)
    write_ground_truth(
        session.path / "ground_truth.yaml",
        [{"start": 1, "end": 3, "state": "WAITING"}],
        3,
    )
    config = yaml.safe_load((PROJECT_ROOT / "config" / "dataset.yaml").read_text(encoding="utf-8"))
    config["dataset"]["output_dir"] = str(tmp_path / "dataset_output")
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    result = build_dataset(
        config_path=config_path, session_root=session_root, dry_run=True
    )
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    assert result.report["session_count"] == 1
    assert before == after
    assert not (tmp_path / "dataset_output").exists()


def test_incremental_builder_materializes_unique_crops_and_valid_manifest(tmp_path: Path) -> None:
    session_root = tmp_path / "sessions"
    session = _session(session_root)
    write_ground_truth(
        session.path / "ground_truth.yaml",
        [{"start": 1, "end": 3, "state": "WAITING"}],
        3,
    )
    config = yaml.safe_load((PROJECT_ROOT / "config" / "dataset.yaml").read_text(encoding="utf-8"))
    output_dir = tmp_path / "dataset_output"
    config["dataset"]["output_dir"] = str(output_dir)
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    build_dataset(config_path=config_path, session_root=session_root)
    first_files = {path.relative_to(output_dir) for path in output_dir.rglob("*") if path.is_file()}
    build_dataset(config_path=config_path, session_root=session_root)
    second_files = {path.relative_to(output_dir) for path in output_dir.rglob("*") if path.is_file()}
    validation = validate_dataset(output_dir, required_session=session.path.name)

    assert first_files == second_files
    assert validation.valid is True
    assert len(validation.prompt_rows) == 1
    assert len(validation.special_rows) == 1
