from pathlib import Path

import pytest

from src.detectors.get_detector import detect_get_window
from src.detectors.hook_detector import detect_hook_bar
from src.detectors.press_detector import detect_press_sequence
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.legacy_adapters.get_detector_adapter import LegacyGetDetectorAdapter
from src.fishing_v2.legacy_adapters.hook_detector_adapter import LegacyHookDetectorAdapter
from src.fishing_v2.legacy_adapters.press_detector_adapter import LegacyPressDetectorAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = PROJECT_ROOT / "assets" / "reference"
CONTEXT = FrameContext(7, 1.2)


def test_hook_adapter_matches_legacy_semantics() -> None:
    frame = REFERENCE / "hook.png"
    legacy = detect_hook_bar(frame, save_debug=False)
    observation = LegacyHookDetectorAdapter().observe(frame, CONTEXT)
    assert observation.detected == legacy["detected"]
    assert observation.confidence == legacy["confidence"]
    assert observation.fill_ratio == legacy["fill_ratio"]
    assert observation.evidence["legacy_debug"] == legacy["debug"]


def test_press_adapter_matches_legacy_semantics() -> None:
    frame = REFERENCE / "press.png"
    legacy = detect_press_sequence(frame, save_debug=False)
    observation = LegacyPressDetectorAdapter().observe(frame, CONTEXT)
    assert observation.detected == legacy["detected"]
    assert observation.confidence == legacy["confidence"]
    assert observation.sequence == ()
    assert "".join(observation.sequence_candidate) == legacy["sequence_text"]
    assert observation.panel_present is True
    assert observation.sequence_ready is False


def test_get_adapter_matches_legacy_semantics() -> None:
    frame = REFERENCE / "get.png"
    legacy = detect_get_window(frame)
    observation = LegacyGetDetectorAdapter().observe(frame, CONTEXT)
    assert observation.detected == legacy["detected"]
    assert observation.confidence == legacy["confidence"]
    assert observation.evidence["matched_features"] == legacy["matched_features"]


@pytest.mark.parametrize("adapter", [
    LegacyHookDetectorAdapter(lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("hook failed"))),
    LegacyPressDetectorAdapter(lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("press failed"))),
    LegacyGetDetectorAdapter(lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("get failed"))),
])
def test_adapter_exception_never_crashes_runtime(adapter) -> None:
    observation = adapter.observe(REFERENCE / "idle.png", CONTEXT)
    assert observation.detected is False
    assert observation.confidence == 0.0
    assert "exception" in observation.evidence


def test_press_adapter_preserves_debug_evidence() -> None:
    result = {
        "detected": True, "confidence": 0.91, "sequence": list("WASD"),
        "sequence_text": "WASD", "key_boxes": [{"key": "W"}],
        "matched_features": ["press_panel"], "debug": {"colour_mode": "teal"},
    }
    observation = LegacyPressDetectorAdapter(lambda *args, **kwargs: result).observe(None, CONTEXT)
    assert observation.evidence["sequence_text"] == "WASD"
    assert observation.evidence["legacy_debug"]["colour_mode"] == "teal"


def test_adapter_preserves_frame_context() -> None:
    observation = LegacyGetDetectorAdapter(lambda frame: {"detected": False, "confidence": 0.2}).observe(None, CONTEXT)
    assert observation.frame_index == 7
    assert observation.timestamp == 1.2
