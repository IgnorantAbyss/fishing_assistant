from pathlib import Path

import pytest

from src.detectors.press_detector import detect_press_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "assets" / "reference"


@pytest.mark.parametrize("filename", ["press.png", "press2.png"])
def test_press_sequence_is_parsed_from_static_references(filename: str) -> None:
    result = detect_press_sequence(REFERENCE_DIR / filename)

    assert result["detected"] is True
    assert result["confidence"] >= 0.60
    assert result["sequence_text"] == "DSDSASSD"
    assert set(result["sequence_text"]) <= set("WASD")
    assert len(result["sequence_text"]) >= 4
    assert len(result["key_boxes"]) == len(result["sequence_text"])
    assert {"press_panel", "key_cells", "letter_templates"} <= set(result["matched_features"])
    debug_path = result["debug"]["debug_image_path"]
    assert debug_path is None or Path(debug_path).is_file()
