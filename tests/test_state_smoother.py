from src.state_smoother import StateSmoother


def test_smoother_fills_short_ready_to_hook_transition() -> None:
    states = ["READY", "UNKNOWN", "IDLE", "UNKNOWN", "HOOK", "HOOK"]

    assert StateSmoother().smooth(states, [0.9, 0.7, 0.8, 0.7, 0.9, 0.9]) == [
        "READY", "HOOK", "HOOK", "HOOK", "HOOK", "HOOK"
    ]


def test_smoother_preserves_long_unknown_and_fills_press_to_get_gap() -> None:
    smoother = StateSmoother()

    assert smoother.smooth(["WAITING", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "READY"]) == [
        "WAITING", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "READY"
    ]
    assert smoother.smooth(["PRESS", "UNKNOWN", "UNKNOWN", "GET"]) == ["PRESS", "PRESS", "PRESS", "GET"]
