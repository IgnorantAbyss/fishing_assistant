from src.dataset import BoundarySampling, SamplingRule
from src.dataset.sampling import sample_session_frames


def _rules(**overrides: SamplingRule) -> dict[str, SamplingRule]:
    rules = {
        "IDLE": SamplingRule(120, "uniform"),
        "WAITING": SamplingRule(150, "uniform"),
        "READY": SamplingRule(None, "all"),
        "HOOK": SamplingRule(None, "all"),
        "PRESS": SamplingRule(None, "all"),
        "GET": SamplingRule(None, "all"),
        "IGNORE": SamplingRule(0, "exclude"),
    }
    rules.update(overrides)
    return rules


def test_waiting_uniform_sampling_is_capped_and_spread_across_interval() -> None:
    labels = {frame: "WAITING" for frame in range(1, 1001)}

    samples = sample_session_frames(
        labels, _rules(), BoundarySampling(False, 0, 0), sample_every_n_frames=2
    )
    indexes = [sample.frame_index for sample in samples]

    assert len(indexes) == 150
    assert indexes[0] == 1
    assert indexes[-1] >= 995
    assert len([index for index in indexes if index <= 500]) in {75, 76}


def test_short_states_are_all_kept_and_ignore_is_excluded() -> None:
    labels = {
        **{frame: "READY" for frame in range(1, 4)},
        **{frame: "HOOK" for frame in range(4, 7)},
        **{frame: "PRESS" for frame in range(7, 10)},
        **{frame: "GET" for frame in range(10, 13)},
        **{frame: "IGNORE" for frame in range(13, 16)},
    }

    samples = sample_session_frames(
        labels, _rules(), BoundarySampling(False, 0, 0), sample_every_n_frames=2
    )

    assert [sample.frame_index for sample in samples] == list(range(1, 13))
    assert all(sample.original_state != "IGNORE" for sample in samples)


def test_boundary_frames_are_prioritized_within_uniform_cap() -> None:
    labels = {
        **{frame: "WAITING" for frame in range(1, 21)},
        **{frame: "READY" for frame in range(21, 26)},
    }
    rules = _rules(WAITING=SamplingRule(5, "uniform"))

    samples = sample_session_frames(
        labels, rules, BoundarySampling(True, 3, 3), sample_every_n_frames=2
    )
    waiting = {sample.frame_index: sample for sample in samples if sample.original_state == "WAITING"}

    assert len(waiting) == 5
    assert {18, 19, 20} <= set(waiting)
    assert all(waiting[index].is_boundary for index in (18, 19, 20))
