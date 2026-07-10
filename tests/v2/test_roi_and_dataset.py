import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.fishing_v2.data.prompt_annotation import write_prompt_ground_truth
from src.fishing_v2.data.prompt_dataset_builder import build_prompt_dataset
from src.fishing_v2.data.prompt_roi import PromptROICandidate, load_approved_prompt_roi
from tools.review_prompt_roi import review_prompt_roi


def _session(root: Path, name: str = "session_test") -> Path:
    session = root / name
    frames = session / "frames"
    frames.mkdir(parents=True)
    for index in range(1, 5):
        image = np.full((100, 200, 3), 30 + index * 20, dtype=np.uint8)
        cv2.putText(image, f"prompt {index}", (50, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imwrite(str(frames / f"{index:06d}.jpg"), image)
    (session / "manifest.json").write_text(json.dumps({
        "session_id": name, "frame_count": 4, "image_format": "jpg",
        "screen_size": [200, 100], "interval_sec": 0.2,
    }), encoding="utf-8")
    (session / "ground_truth.yaml").write_text(
        "segments:\n  - start: 1\n    end: 4\n    state: HOOK\n", encoding="utf-8"
    )
    write_prompt_ground_truth(session / "prompt_ground_truth.yaml", [
        {"start": 1, "end": 2, "observation": "READY_PROMPT"},
        {"start": 3, "end": 3, "observation": "IGNORE"},
        {"start": 4, "end": 4, "observation": "OTHER_PROMPT"},
    ], 4)
    return session


def _config(path: Path, legacy: Path, output: Path, *, approved: bool) -> Path:
    prompt = {"observer": "unimplemented", "roi_status": "unapproved", "roi": None}
    if approved:
        prompt = {"observer": "unimplemented", "roi_status": "approved", "roi": {"x1": 0.2, "y1": 0.0, "x2": 0.8, "y2": 0.3}}
    data = {
        "data": {
            "session_root": str(path.parent / "sessions"),
            "legacy_dataset_root": str(legacy),
            "prompt_dataset_root": str(output),
            "dataset_version": "prompt_observation_v1",
            "development_sessions": ["session_test"],
            "final_test": {"status": "not_collected", "sessions": []},
        },
        "prompt": prompt,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _split(path: Path) -> Path:
    path.write_text(yaml.safe_dump({"train": [], "validation": [], "test": ["session_test"]}), encoding="utf-8")
    return path


def test_multiple_roi_candidate_contact_sheets_are_created(tmp_path: Path) -> None:
    session = _session(tmp_path / "sessions")
    candidates = [
        PromptROICandidate("a", 0.1, 0.0, 0.9, 0.3),
        PromptROICandidate("b", 0.2, 0.0, 0.8, 0.2),
    ]
    report = review_prompt_roi([session], candidates, tmp_path / "reports", samples_per_state=1)
    assert (tmp_path / "reports" / "a.jpg").is_file()
    assert (tmp_path / "reports" / "b.jpg").is_file()
    assert report["automatic_selection_performed"] is False


def test_unapproved_roi_blocks_materialized_dataset(tmp_path: Path) -> None:
    _session(tmp_path / "sessions")
    config = _config(tmp_path / "config.yaml", tmp_path / "legacy", tmp_path / "new", approved=False)
    with pytest.raises(PermissionError, match="unapproved"):
        build_prompt_dataset(config_path=config, split_path=_split(tmp_path / "split.yaml"), session_root=tmp_path / "sessions", output_root=tmp_path / "new", dry_run=False)


def test_unapproved_roi_allows_dry_run_with_candidate(tmp_path: Path) -> None:
    _session(tmp_path / "sessions")
    config = _config(tmp_path / "config.yaml", tmp_path / "legacy", tmp_path / "new", approved=False)
    result = build_prompt_dataset(
        config_path=config, split_path=_split(tmp_path / "split.yaml"),
        session_root=tmp_path / "sessions", output_root=tmp_path / "new", dry_run=True,
        dry_run_candidate=PromptROICandidate("candidate", 0.2, 0.0, 0.8, 0.3),
    )
    assert result.row_count == 3  # IGNORE excluded, explicit boundary labels retained.
    assert not (tmp_path / "new").exists()


def test_review_never_approves_from_bright_mask(tmp_path: Path) -> None:
    session = _session(tmp_path / "sessions")
    report = review_prompt_roi([session], [PromptROICandidate("a", 0.1, 0.0, 0.9, 0.3)], tmp_path / "reports")
    assert report["roi_status"] == "unapproved"
    assert report["bright_mask_used_for_approval"] is False


def test_roi_review_does_not_overwrite_legacy_roi_config(tmp_path: Path) -> None:
    legacy = tmp_path / "roi.yaml"
    legacy.write_text("legacy: true\n", encoding="utf-8")
    before = legacy.read_bytes()
    review_prompt_roi([_session(tmp_path / "sessions")], [PromptROICandidate("a", 0.1, 0.0, 0.9, 0.3)], tmp_path / "reports")
    assert legacy.read_bytes() == before


def test_prompt_label_is_not_derived_from_global_hook_state(tmp_path: Path) -> None:
    _session(tmp_path / "sessions")
    config = _config(tmp_path / "config.yaml", tmp_path / "legacy", tmp_path / "new", approved=False)
    result = build_prompt_dataset(
        config_path=config, split_path=_split(tmp_path / "split.yaml"), session_root=tmp_path / "sessions",
        output_root=tmp_path / "new", dry_run=True,
        dry_run_candidate=PromptROICandidate("candidate", 0.2, 0.0, 0.8, 0.3),
    )
    assert result.rows[0]["global_state"] == "HOOK"
    assert result.rows[0]["prompt_observation"] == "READY_PROMPT"


def test_old_test_session_is_marked_development_diagnostic(tmp_path: Path) -> None:
    _session(tmp_path / "sessions")
    config = _config(tmp_path / "config.yaml", tmp_path / "legacy", tmp_path / "new", approved=False)
    result = build_prompt_dataset(
        config_path=config, split_path=_split(tmp_path / "split.yaml"), session_root=tmp_path / "sessions",
        output_root=tmp_path / "new", dry_run=True,
        dry_run_candidate=PromptROICandidate("candidate", 0.2, 0.0, 0.8, 0.3),
    )
    assert {row["usage"] for row in result.rows} == {"development_diagnostic"}


def test_final_test_remains_not_collected() -> None:
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "config" / "fishing_v2.yaml").read_text(encoding="utf-8"))
    assert config["data"]["final_test"] == {"status": "not_collected", "sessions": []}


def test_materialized_dataset_has_lineage_and_does_not_touch_legacy(tmp_path: Path) -> None:
    _session(tmp_path / "sessions")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    sentinel = legacy / "sentinel.txt"
    sentinel.write_text("frozen", encoding="utf-8")
    config = _config(tmp_path / "config.yaml", legacy, tmp_path / "new", approved=True)
    result = build_prompt_dataset(
        config_path=config, split_path=_split(tmp_path / "split.yaml"), session_root=tmp_path / "sessions",
        output_root=tmp_path / "new", dry_run=False,
    )
    assert result.row_count == 3
    assert (tmp_path / "new" / "manifest.csv").is_file()
    lineage = json.loads((tmp_path / "new" / "dataset_lineage.json").read_text(encoding="utf-8"))
    assert lineage["label_source"] == "prompt_ground_truth.yaml only"
    assert lineage["final_test_eligibility"]["status"] == "not_collected"
    assert sentinel.read_text(encoding="utf-8") == "frozen"


def test_builder_refuses_legacy_output_path(tmp_path: Path) -> None:
    _session(tmp_path / "sessions")
    legacy = tmp_path / "legacy"
    config = _config(tmp_path / "config.yaml", legacy, tmp_path / "new", approved=True)
    with pytest.raises(ValueError, match="frozen"):
        build_prompt_dataset(config_path=config, split_path=_split(tmp_path / "split.yaml"), session_root=tmp_path / "sessions", output_root=legacy, dry_run=True)
