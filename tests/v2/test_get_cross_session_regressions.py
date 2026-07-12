from pathlib import Path

import pytest
import yaml

from src.detectors.get_detector import detect_get_window
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import GetObservation
from src.fishing_v2.legacy_adapters.get_detector_adapter import LegacyGetDetectorAdapter
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode
from src.fishing_v2.runtime.detector_evidence import (
    DetectorEvidenceQualifier,
    EvidenceQualificationConfig,
)


ROOT = Path(__file__).resolve().parents[2]
SESSIONS = ROOT / "assets" / "replay" / "sessions"


def _frame(session: str, frame: int) -> Path:
    return SESSIONS / session / "frames" / f"{frame:06d}.jpg"


def test_get_geometry_and_temporal_rules_are_configured() -> None:
    config = yaml.safe_load((ROOT / "config" / "fishing_v2.yaml").read_text(encoding="utf-8"))
    get = config["get_detector"]
    assert get["panel_confirmation_frames"] == 2
    assert get["dark_scene_fallback"]["min_grid_cells"] == 8
    assert get["dark_scene_fallback"]["min_dark_ratio"] == 0.80


@pytest.mark.parametrize("frame", [440, 441, 458])
def test_pilot_true_get_panel_remains_detected(frame: int) -> None:
    result = detect_get_window(_frame("session_20260710_130308", frame))
    assert result["detected"] is True
    assert result["debug"]["localization_source"] == "geometry_valid_dark_contour"
    assert "item_grid" in result["matched_features"]


@pytest.mark.parametrize("frame", [485, 490, 499])
def test_dark_scene_true_get_uses_strong_grid_fallback(frame: int) -> None:
    result = detect_get_window(_frame("session_20260709_192315", frame))
    assert result["detected"] is True
    assert result["debug"]["localization_source"] == "fixed_geometry_strong_grid_fallback"
    assert result["debug"]["structure_debug"]["grid_cell_candidates"] >= 8


def test_dark_scene_get_fade_in_is_explicitly_not_yet_visible() -> None:
    result = detect_get_window(_frame("session_20260709_192315", 484))
    assert result["detected"] is False
    assert result["debug"]["localization_source"] == "none"


@pytest.mark.parametrize(("session", "frame"), [
    ("session_20260710_124419", 316),
    ("session_20260710_125441", 9),
    ("session_20260710_125441", 394),
    ("session_20260710_131254", 1),
])
def test_known_non_get_frames_are_rejected_by_panel_geometry(session: str, frame: int) -> None:
    result = detect_get_window(_frame(session, frame))
    assert result["detected"] is False
    assert result["debug"]["panel_bbox"] is None
    assert result["debug"]["localization_source"] == "none"


def _get(frame: int, *, detected: bool = True) -> GetObservation:
    return GetObservation(
        detected, 1.0 if detected else 0.0, frame, frame * 0.2,
        evidence={"matched_features": ["inventory_title", "item_grid", "collect_button"]},
    )


def _actual_observation(session: str, frame: int) -> GetObservation:
    return LegacyGetDetectorAdapter().observe(
        _frame(session, frame), FrameContext(frame, frame * 0.2)
    )


@pytest.mark.parametrize(("session", "first", "second"), [
    ("session_20260710_130308", 440, 441),
    ("session_20260709_192315", 485, 486),
])
def test_real_get_frames_become_qualified_after_confirmation(
    session: str, first: int, second: int
) -> None:
    qualifier = DetectorEvidenceQualifier(EvidenceQualificationConfig(get_panel_confirmation_frames=2))
    _, pending = qualifier.qualify_get(_actual_observation(session, first), DetectorActivationMode.ARMED)
    _, confirmed = qualifier.qualify_get(_actual_observation(session, second), DetectorActivationMode.ARMED)
    assert pending.raw_detected is True and pending.qualified_detected is False
    assert confirmed.qualified_detected is True and confirmed.used_by_fusion is True


def test_real_pilot_panel_disappearance_clears_qualified_get() -> None:
    qualifier = DetectorEvidenceQualifier(EvidenceQualificationConfig(get_panel_confirmation_frames=2))
    qualifier.qualify_get(_actual_observation("session_20260710_130308", 457), DetectorActivationMode.ARMED)
    _, present = qualifier.qualify_get(
        _actual_observation("session_20260710_130308", 458), DetectorActivationMode.ARMED
    )
    cleared, missing = qualifier.qualify_get(
        _actual_observation("session_20260710_130308", 459), DetectorActivationMode.ACTIVE
    )
    assert present.qualified_detected is True
    assert missing.qualified_detected is False and missing.used_by_fusion is False
    assert cleared is not None and cleared.detected is False


def test_single_raw_get_candidate_does_not_enter_fusion() -> None:
    qualifier = DetectorEvidenceQualifier(EvidenceQualificationConfig(get_panel_confirmation_frames=2))
    observation, result = qualifier.qualify_get(_get(10), DetectorActivationMode.ARMED)
    assert result.raw_detected is True
    assert result.qualified_detected is False
    assert result.used_by_fusion is False
    assert observation is not None and observation.detected is False
    assert result.qualification_reason == "get_panel_temporal_confirmation_pending"


def test_get_confirmation_requires_distinct_frames_and_disappears_immediately() -> None:
    qualifier = DetectorEvidenceQualifier(EvidenceQualificationConfig(get_panel_confirmation_frames=2))
    qualifier.qualify_get(_get(20), DetectorActivationMode.ARMED)
    _, duplicate = qualifier.qualify_get(_get(20), DetectorActivationMode.ARMED)
    assert duplicate.qualified_detected is False
    _, confirmed = qualifier.qualify_get(_get(21), DetectorActivationMode.ARMED)
    assert confirmed.qualified_detected is True
    cleared, missing = qualifier.qualify_get(_get(22, detected=False), DetectorActivationMode.ACTIVE)
    assert missing.qualified_detected is False
    assert missing.used_by_fusion is False
    assert cleared is not None and cleared.detected is False
