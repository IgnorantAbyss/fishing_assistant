from pathlib import Path

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
