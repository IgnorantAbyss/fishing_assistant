import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from src.fishing_v2.data.prompt_observation_dataset import FORMAL_SESSION_IDS, TRIAL_SESSION_ID
from src.fishing_v2.perception.prompt_bundle import PromptBundleError, load_prompt_bundle
from src.fishing_v2.perception.prototype_prompt_observer import OPERATIONAL_LABELS


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts" / "prompt_observer" / "prototype_v1"
LOSO = ROOT / "reports" / "fishing_v2" / "prompt_observer_prototype_summary.json"
VALIDATION = ROOT / "reports" / "fishing_v2" / "final_prompt_bundle_validation.json"


def test_final_bundle_has_exact_formal_lineage_and_35_session_medoids() -> None:
    loaded = load_prompt_bundle(BUNDLE)
    payload = loaded.bundle
    assert tuple(payload["training_session_ids"]) == FORMAL_SESSION_IDS
    assert payload["excluded_session_ids"] == [TRIAL_SESSION_ID]
    assert payload["prototype_count"] == 35
    assert len(loaded.model.prototypes) == 35
    assert "IGNORE" not in {item.label for item in loaded.model.prototypes}
    counts = {
        label: sum(item.label == label for item in loaded.model.prototypes)
        for label in OPERATIONAL_LABELS
    }
    assert counts == {label: 7 for label in OPERATIONAL_LABELS}
    assert set(payload["prompt_ground_truth_sha256"]) == set(FORMAL_SESSION_IDS)
    assert all(len(value) == 64 for value in payload["prompt_ground_truth_sha256"].values())


def test_final_calibration_is_conservative_aggregation_of_seven_loso_folds() -> None:
    loaded = load_prompt_bundle(BUNDLE)
    folds = json.loads(LOSO.read_text(encoding="utf-8"))["folds"]
    assert len(folds) == 7
    for label in OPERATIONAL_LABELS:
        expected = float(np.median([fold["class_thresholds"][label] for fold in folds]))
        assert loaded.bundle["class_thresholds"][label] == pytest.approx(expected)
    assert loaded.bundle["ambiguity_margin"] == pytest.approx(
        float(np.median([fold["ambiguity_margin"] for fold in folds]))
    )
    assert loaded.bundle["idle_stability_frames"] == max(
        fold["idle_stability_frames"] for fold in folds
    )
    assert loaded.bundle["threshold_aggregation"]["runtime_adaptation"] is False


def test_bundle_hash_verifies_and_tampering_is_rejected(tmp_path: Path) -> None:
    loaded = load_prompt_bundle(BUNDLE)
    assert loaded.bundle_sha256 == loaded.metadata["bundle_sha256"]
    assert hashlib.sha256((BUNDLE / "prototypes.npz").read_bytes()).hexdigest() == (
        loaded.bundle["prototypes_sha256"]
    )
    altered = tmp_path / "bundle"
    shutil.copytree(BUNDLE, altered)
    with (altered / "bundle.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(PromptBundleError, match="SHA-256"):
        load_prompt_bundle(altered)


def test_final_bundle_predicted_replay_is_a_clean_deployment_regression() -> None:
    report = json.loads(VALIDATION.read_text(encoding="utf-8"))
    assert report["deployment_regression_only"] is True
    assert report["deployment_regression_passed"] is True
    assert report["formal_sessions"] == list(FORMAL_SESSION_IDS)
    assert report["replay_complete_count"] == 7
    assert report["result_counts"] == {"PASS": 7}
    assert report["false_intents"] == 0
    assert report["missed_expected_intents"] == 0
    assert report["sync_required_frames"] == 0
    assert report["actions_applied"] == 0
