from pathlib import Path

import numpy as np
import pytest

from src.state_detector import StateDetector, expected_reference_images


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "assets" / "reference"


@pytest.fixture(scope="module")
def detector() -> StateDetector:
    return StateDetector(REFERENCE_DIR)


@pytest.mark.parametrize("path, expected_state", list(expected_reference_images(REFERENCE_DIR)))
def test_reference_images_are_classified_to_the_expected_state(
    detector: StateDetector, path: Path, expected_state: str
) -> None:
    result = detector.detect(path)

    assert result.state == expected_state
    assert 0.0 <= result.confidence <= 1.0
    assert result.matched_features
    assert result.debug["best_reference"]
    assert set(result.debug["raw_scores"]) == {"IDLE", "WAITING", "READY", "HOOK", "PRESS", "GET"}


def test_black_synthetic_image_is_unknown(detector: StateDetector) -> None:
    result = detector.detect(np.zeros((1151, 2048, 3), dtype=np.uint8))

    assert result.state == "UNKNOWN"
    assert result.matched_features == ["blank_frame"]
    assert result.debug["raw_scores"] == {
        "IDLE": 0.0,
        "WAITING": 0.0,
        "READY": 0.0,
        "HOOK": 0.0,
        "PRESS": 0.0,
        "GET": 0.0,
    }
