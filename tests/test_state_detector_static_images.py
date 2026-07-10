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
    assert set(
        (
            "top_prompt_text_score",
            "top_prompt_image_score",
            "prompt_margin",
            "selected_prompt_state",
            "decision_reason",
        )
    ).issubset(result.debug)


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
    assert result.debug["decision_reason"] == "blank_frame"


def test_hook_fusion_requires_divider_or_prompt_for_single_frame_override() -> None:
    strong = StateDetector._hook_fusion_gate(
        {"detected": True, "confidence": 0.80, "matched_features": ["hook_bar_rect", "bar_fill", "divider_line"]},
        0.74,
    )
    weak = StateDetector._hook_fusion_gate(
        {"detected": True, "confidence": 0.90, "matched_features": ["hook_bar_rect", "bar_fill"]},
        0.80,
    )
    prompt_wins = StateDetector._hook_fusion_gate(
        {"detected": True, "confidence": 0.90, "matched_features": ["hook_bar_rect", "bar_fill"]},
        0.85,
    )

    assert strong["strong"] is True
    assert weak == {
        "decision": "weak_hook_requires_temporal_support",
        "strong": False,
        "weak_candidate": True,
        "hook_minus_prompt": 0.1,
    }
    assert prompt_wins["decision"] == "hook_evidence_rejected"
