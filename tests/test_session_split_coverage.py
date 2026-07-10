from pathlib import Path

from src.dataset.validation import load_session_split, validate_session_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_ROOT / "config" / "session_split.yaml"

FORMAL_SESSIONS = {
    "session_20260709_192315",
    "session_20260710_061220",
    "session_20260710_123210",
    "session_20260710_124419",
    "session_20260710_125441",
    "session_20260710_130308",
    "session_20260710_131254",
}


def test_fixed_split_contains_seven_formal_sessions_exactly_once() -> None:
    all_sessions = ["session_20260709_192231", *sorted(FORMAL_SESSIONS)]
    split = load_session_split(SPLIT_PATH, all_sessions)
    validate_session_split(split)

    assigned = [*split.train, *split.validation, *split.test]
    assert len(assigned) == 7
    assert set(assigned) == FORMAL_SESSIONS
    assert len(assigned) == len(set(assigned))
    assert len(split.train) == 4
    assert len(split.validation) == 1
    assert len(split.test) == 2


def test_trial_is_excluded_and_seed_does_not_change_fixed_assignment() -> None:
    forward = ["session_20260709_192231", *sorted(FORMAL_SESSIONS)]
    reverse = list(reversed(forward))

    first = load_session_split(SPLIT_PATH, forward)
    second = load_session_split(SPLIT_PATH, reverse)

    assert first.seed == 42
    assert first.train == second.train
    assert first.validation == second.validation
    assert first.test == second.test
    assert first.split_for("session_20260709_192231") == "excluded"
    assert first.excluded[0].reason == "trial_session"
