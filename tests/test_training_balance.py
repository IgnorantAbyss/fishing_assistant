from collections import Counter

from src.dataset.sampling import BalanceCandidate
from tools.prepare_training_dataset import _selection_fields, balance_training_candidates


def _candidates(
    counts: dict[str, int], *, dataset: str, split: str = "train"
) -> list[BalanceCandidate]:
    result: list[BalanceCandidate] = []
    frame = 1
    for label, count in counts.items():
        for offset in range(count):
            session_id = f"session_{offset % 4}"
            if dataset == "special" and label == "NONE":
                original_state = ("IDLE", "WAITING", "READY")[offset % 3]
            else:
                original_state = label if label != "NONE" else "HOOK"
            result.append(
                BalanceCandidate(
                    session_id=session_id,
                    frame_index=frame,
                    original_state=original_state,
                    label=label,
                    split=split,
                    is_boundary=offset < 3,
                )
            )
            frame += 1
    return result


def test_only_train_majorities_are_balanced_and_rare_labels_are_kept() -> None:
    prompt = _candidates(
        {"IDLE": 120, "WAITING": 500, "READY": 150, "NONE": 400},
        dataset="prompt",
    )
    special = _candidates(
        {"HOOK": 40, "PRESS": 30, "GET": 20, "NONE": 600},
        dataset="special",
    )

    selections, details = balance_training_candidates(prompt, special)
    prompt_after = Counter(
        candidate.label for candidate in prompt if candidate.key in selections["prompt"]
    )
    special_after = Counter(
        candidate.label for candidate in special if candidate.key in selections["special"]
    )

    assert details["prompt_majority_cap"] == 240
    assert prompt_after == {"IDLE": 120, "WAITING": 240, "READY": 150, "NONE": 240}
    assert details["special_none_cap"] == 250
    assert special_after == {"HOOK": 40, "PRESS": 30, "GET": 20, "NONE": 250}


def test_special_none_selection_covers_states_sessions_and_boundaries() -> None:
    prompt = _candidates(
        {"IDLE": 10, "WAITING": 10, "READY": 10, "NONE": 10}, dataset="prompt"
    )
    special = _candidates(
        {"HOOK": 40, "PRESS": 30, "GET": 20, "NONE": 600}, dataset="special"
    )

    selections, _ = balance_training_candidates(prompt, special)
    selected_none = [
        candidate
        for candidate in special
        if candidate.label == "NONE" and candidate.key in selections["special"]
    ]

    assert {candidate.original_state for candidate in selected_none} == {"IDLE", "WAITING", "READY"}
    assert {candidate.session_id for candidate in selected_none} == {
        "session_0",
        "session_1",
        "session_2",
        "session_3",
    }
    assert all(
        candidate.key in selections["special"]
        for candidate in special
        if candidate.label == "NONE" and candidate.is_boundary
    )


def test_evaluation_rows_are_never_majority_downsampled_or_selected_for_training() -> None:
    candidate = BalanceCandidate(
        "session_validation", 1, "WAITING", "WAITING", "validation", False
    )

    assert _selection_fields(candidate, frozenset()) == (False, "evaluation_only")


def test_balance_is_deterministic_when_input_order_changes() -> None:
    prompt = _candidates(
        {"IDLE": 120, "WAITING": 500, "READY": 150, "NONE": 400},
        dataset="prompt",
    )
    special = _candidates(
        {"HOOK": 40, "PRESS": 30, "GET": 20, "NONE": 600},
        dataset="special",
    )

    first, _ = balance_training_candidates(prompt, special)
    second, _ = balance_training_candidates(list(reversed(prompt)), list(reversed(special)))

    assert first == second
