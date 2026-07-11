from pathlib import Path

from src.detectors.get_detector import detect_get_window
from src.detectors.press_detector import detect_press_sequence


ROOT = Path(__file__).resolve().parents[2]
FRAMES = ROOT / "assets" / "replay" / "sessions" / "session_20260710_130308" / "frames"


def test_pilot_press_frame_428_parses_mixed_progress_colours() -> None:
    result = detect_press_sequence(FRAMES / "000428.jpg", save_debug=False)
    assert len(result["key_boxes"]) == 8
    assert all(key in "WASD" for key in result["sequence_candidate"])
    # Panel presence is structural and independent of glyph confidence. A
    # single frame never exposes an action-ready sequence.
    assert result["panel_present"] is True
    assert result["detected"] is True
    assert result["sequence_ready"] is False
    assert result["sequence"] == []
    assert result["sequence_confidence"] < 0.68


def test_pilot_get_panel_is_detected_from_frame_440() -> None:
    result = detect_get_window(FRAMES / "000440.jpg")
    assert result["detected"] is True
    assert "item_grid" in result["matched_features"] or "collect_button" in result["matched_features"]


def test_pilot_get_panel_disappearance_is_visible_at_frame_459() -> None:
    assert detect_get_window(FRAMES / "000458.jpg")["detected"] is True
    assert detect_get_window(FRAMES / "000459.jpg")["detected"] is False
