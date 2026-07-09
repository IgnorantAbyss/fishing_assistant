from pathlib import Path

import pytest

from src.detectors.hook_detector import detect_hook_bar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "assets" / "reference"


@pytest.mark.parametrize("filename", ["hook.png", "hook2.png"])
def test_hook_bar_is_detected_from_static_references(filename: str) -> None:
    result = detect_hook_bar(REFERENCE_DIR / filename)

    assert result["detected"] is True
    assert result["confidence"] >= 0.60
    assert result["bar_bbox"] is not None
    assert 0.0 <= result["fill_ratio"] <= 1.0
    assert "hook_bar_rect" in result["matched_features"]
    debug_path = result["debug"]["debug_image_path"]
    assert debug_path is None or Path(debug_path).is_file()
    assert result["should_press_space"] is False
