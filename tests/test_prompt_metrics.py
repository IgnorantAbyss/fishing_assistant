import pytest

from src.ml.prompt_metrics import (
    classification_metrics,
    confusion_matrix,
    metrics_by_session,
    probabilities_to_predictions,
    select_confidence_threshold,
)


def test_confusion_matrix_uses_fixed_class_axes() -> None:
    matrix = confusion_matrix(["IDLE", "WAITING"], ["WAITING", "WAITING"])
    assert matrix[0][1] == 1
    assert matrix[1][1] == 1
    assert len(matrix) == 4


def test_perfect_predictions_have_macro_f1_one() -> None:
    labels = ["IDLE", "WAITING", "READY", "NONE"]
    metrics = classification_metrics(labels, labels)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)


def test_idle_waiting_mutual_confusion_is_counted() -> None:
    metrics = classification_metrics(
        ["IDLE", "IDLE", "WAITING", "WAITING"],
        ["WAITING", "IDLE", "IDLE", "WAITING"],
    )
    assert metrics["idle_to_waiting"] == 1
    assert metrics["waiting_to_idle"] == 1
    assert metrics["idle_waiting_mutual_confusion_rate"] == pytest.approx(0.5)


def test_unknown_ratio_and_recall_include_rejected_predictions() -> None:
    metrics = classification_metrics(
        ["IDLE", "WAITING", "READY", "NONE"],
        ["UNKNOWN", "WAITING", "READY", "NONE"],
    )
    assert metrics["unknown_ratio"] == pytest.approx(0.25)
    assert metrics["per_class"]["IDLE"]["recall"] == 0.0


def test_metrics_are_grouped_per_session() -> None:
    grouped = metrics_by_session(
        ["IDLE", "WAITING", "READY", "NONE"],
        ["IDLE", "WAITING", "READY", "NONE"],
        ["a", "a", "b", "b"],
    )
    assert set(grouped) == {"a", "b"}
    assert grouped["a"]["sample_count"] == 2


def test_threshold_search_selects_highest_candidate_under_unknown_limit() -> None:
    probabilities = [[0.95, 0.02, 0.02, 0.01]] * 9 + [[0.55, 0.2, 0.15, 0.1]]
    result = select_confidence_threshold(
        probabilities,
        ["IDLE"] * 10,
        [0.5, 0.55, 0.9],
        0.1,
        source_split="validation",
    )
    assert result["selected_threshold"] == 0.9
    assert result["validation_unknown_ratio"] == pytest.approx(0.1)


def test_threshold_search_warns_when_no_candidate_meets_limit() -> None:
    result = select_confidence_threshold(
        [[0.4, 0.3, 0.2, 0.1]],
        ["IDLE"],
        [0.5, 0.9],
        0.1,
        source_split="validation",
    )
    assert result["confidence_gate_warning"] is True
    assert result["selection_rule"] == "closest_validation_unknown_ratio_to_limit"


def test_threshold_search_rejects_test_data() -> None:
    with pytest.raises(ValueError, match="validation only"):
        select_confidence_threshold(
            [[0.9, 0.05, 0.03, 0.02]],
            ["IDLE"],
            [0.5],
            0.1,
            source_split="test",
        )


def test_probability_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="shape"):
        probabilities_to_predictions([[0.5, 0.5]], 0.5)
