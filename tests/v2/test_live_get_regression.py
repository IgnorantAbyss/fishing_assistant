from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from src.config_loader import ROIConfig, load_roi_config
from src.detectors.get_detector import detect_get_window
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.legacy_adapters.get_detector_adapter import LegacyGetDetectorAdapter
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode
from src.fishing_v2.runtime.detector_evidence import (
    DetectorEvidenceQualifier,
    EvidenceQualificationConfig,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "assets" / "reference" / "get"
MANIFEST = REFERENCE / "live_get_regression.yaml"
ROI_AS_FRAME = ROIConfig(
    screen_reference=(486, 418),
    rois={"get_window": (0.0, 0.0, 1.0, 1.0), "get_search": (0.0, 0.0, 1.0, 1.0)},
    source=None,
)


def _items() -> list[dict[str, object]]:
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return payload["items"]


def _path(item: dict[str, object]) -> Path:
    return REFERENCE / str(item["image"])


def _result(item: dict[str, object]) -> dict[str, object]:
    return detect_get_window(_path(item), roi_config=ROI_AS_FRAME)


def test_live_get_manifest_preserves_review_and_source_hashes() -> None:
    items = _items()
    assert len(items) == 16
    for item in items:
        path = _path(item)
        assert item["source_session"] == "session_20260718_082600"
        assert item["review_status"] == "human_reviewed_live_audit"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["source_image_sha256"]


@pytest.mark.parametrize("frame", [2428, 2433, 2538, 6528, 6529, 6569, 6573])
def test_reviewed_live_collect_ready_panels_are_raw_positive(frame: int) -> None:
    item = next(item for item in _items() if item["source_frame"] == frame)
    result = _result(item)
    assert result["detected"] is True
    assert result["confidence"] >= 0.85
    assert result["debug"]["localization_source"] == "vertical_sliding_strong_grid"
    assert result["debug"]["vertical_sliding"]["selected"]["structure_debug"]["grid_cell_candidates"] >= 8


@pytest.mark.parametrize("frame", [4482, 4402, 4407, 4546, 4421, 4556, 4477])
def test_reviewed_live_non_get_frames_remain_raw_negative(frame: int) -> None:
    item = next(item for item in _items() if item["source_frame"] == frame)
    result = _result(item)
    assert result["detected"] is False
    assert result["debug"]["panel_bbox"] is None


@pytest.mark.parametrize("frames", [(2428, 2433), (6528, 6529)])
def test_stable_live_panels_temporally_qualify(frames: tuple[int, int]) -> None:
    qualifier = DetectorEvidenceQualifier(EvidenceQualificationConfig(
        get_panel_confirmation_frames=2,
    ))
    outcomes = []
    for frame in frames:
        item = next(item for item in _items() if item["source_frame"] == frame)
        observation = LegacyGetDetectorAdapter(
            lambda _image, item=item: _result(item)
        ).observe(object(), FrameContext(frame, frame / 20.0))
        _, outcome = qualifier.qualify_get(observation, DetectorActivationMode.BURST)
        outcomes.append(outcome)
    assert outcomes[0].qualified_detected is False
    assert outcomes[1].qualified_detected is True
    assert outcomes[1].used_by_fusion is True


def test_isolated_cycle2_candidates_never_qualify() -> None:
    qualifier = DetectorEvidenceQualifier(EvidenceQualificationConfig(
        get_panel_confirmation_frames=2,
    ))
    for frame in (4402, 4407, 4546):
        item = next(item for item in _items() if item["source_frame"] == frame)
        result = _result(item)
        observation = LegacyGetDetectorAdapter(
            lambda _image, result=result: result
        ).observe(object(), FrameContext(frame, frame / 20.0))
        _, outcome = qualifier.qualify_get(observation, DetectorActivationMode.BURST)
        assert outcome.qualified_detected is False


def test_get_search_roi_covers_legacy_and_live_vertical_positions() -> None:
    config = load_roi_config()
    search = config.pixel_roi("get_search", 2560, 1440)
    legacy = config.pixel_roi("get_window", 2560, 1440)
    assert search[0] == legacy[0] and search[2:] == legacy[2:]
    assert search[1] == 677
    assert legacy[1] == 792
    assert legacy[1] - search[1] == 115


def test_panel_geometry_is_relative_to_detected_vertical_anchor() -> None:
    items = _items()
    live = _result(next(item for item in items if item["source_frame"] == 2428))
    replay = detect_get_window(
        ROOT / "assets" / "replay" / "sessions" / "session_20260710_130308" / "frames" / "000440.jpg"
    )
    assert live["debug"]["panel_bbox"][1] == 0
    assert replay["debug"]["panel_bbox"][1] > 0
    assert live["debug"]["structure_debug"]["grid_cell_candidates"] >= 8
    assert replay["debug"]["structure_debug"]["grid_cell_candidates"] >= 8
