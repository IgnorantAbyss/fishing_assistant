"""Deterministic, boundary-aware per-session frame sampling."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Hashable, Mapping, Sequence

from src.dataset import BoundarySampling, SamplingRule
from src.replay_ground_truth import IGNORE_STATE


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    original_state: str
    is_boundary: bool


@dataclass(frozen=True)
class BalanceCandidate:
    session_id: str
    frame_index: int
    original_state: str
    label: str
    split: str
    is_boundary: bool

    @property
    def key(self) -> tuple[str, int, str]:
        return self.session_id, self.frame_index, self.label


@dataclass(frozen=True)
class BalanceSelection:
    selected_keys: frozenset[tuple[str, int, str]]
    requested_cap: int
    effective_cap: int
    boundary_kept: int
    boundary_overflow: int


def boundary_frame_indexes(
    labels: Mapping[int, str], settings: BoundarySampling
) -> set[int]:
    if not settings.enabled or not labels:
        return set()
    last_frame = max(labels)
    boundaries: set[int] = set()
    for frame_index in range(2, last_frame + 1):
        if labels[frame_index] == labels[frame_index - 1]:
            continue
        boundaries.update(
            range(max(1, frame_index - settings.frames_before), frame_index)
        )
        boundaries.update(
            range(frame_index, min(last_frame + 1, frame_index + settings.frames_after))
        )
    return boundaries


def uniform_indexes(indexes: Sequence[int], count: int) -> list[int]:
    """Choose ``count`` indexes spread across the complete ordered sequence."""
    ordered = sorted(set(indexes))
    if count <= 0 or not ordered:
        return []
    if count >= len(ordered):
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    positions = [round(position * (len(ordered) - 1) / (count - 1)) for position in range(count)]
    return [ordered[position] for position in positions]


def _uniform_candidates(
    candidates: Sequence[BalanceCandidate], count: int
) -> list[BalanceCandidate]:
    ordered = sorted(candidates, key=lambda item: (item.session_id, item.frame_index, item.label))
    if count <= 0 or not ordered:
        return []
    if count >= len(ordered):
        return ordered
    positions = uniform_indexes(list(range(len(ordered))), count)
    return [ordered[position] for position in positions]


def select_balanced_candidates(
    candidates: Sequence[BalanceCandidate],
    cap: int,
    *,
    group_key: Callable[[BalanceCandidate], Hashable],
) -> BalanceSelection:
    """Select a deterministic, boundary-first sample distributed across groups."""
    unique = {
        candidate.key: candidate for candidate in candidates
    }
    ordered = sorted(unique.values(), key=lambda item: (item.session_id, item.frame_index, item.label))
    boundary = [candidate for candidate in ordered if candidate.is_boundary]
    effective_cap = max(cap, len(boundary))
    if len(ordered) <= effective_cap:
        return BalanceSelection(
            frozenset(candidate.key for candidate in ordered),
            cap,
            effective_cap,
            len(boundary),
            max(0, len(boundary) - cap),
        )

    selected = {candidate.key: candidate for candidate in boundary}
    groups: dict[Hashable, list[BalanceCandidate]] = defaultdict(list)
    for candidate in ordered:
        if candidate.key not in selected:
            groups[group_key(candidate)].append(candidate)
    group_names = sorted(groups, key=str)
    remaining_slots = effective_cap - len(selected)
    if group_names and remaining_slots > 0:
        quota, extra = divmod(remaining_slots, len(group_names))
        for index, name in enumerate(group_names):
            amount = quota + (1 if index < extra else 0)
            for candidate in _uniform_candidates(groups[name], amount):
                selected[candidate.key] = candidate

    remaining_slots = effective_cap - len(selected)
    if remaining_slots > 0:
        pool = [candidate for candidate in ordered if candidate.key not in selected]
        for candidate in _uniform_candidates(pool, remaining_slots):
            selected[candidate.key] = candidate
    return BalanceSelection(
        frozenset(selected),
        cap,
        effective_cap,
        len(boundary),
        max(0, len(boundary) - cap),
    )


def _sample_state(
    indexes: list[int],
    boundary_indexes: set[int],
    rule: SamplingRule,
    sample_every_n_frames: int,
) -> list[int]:
    if rule.strategy == "exclude" or rule.max_samples == 0:
        return []
    if rule.strategy == "all":
        selected = sorted(indexes)
        return selected[: rule.max_samples] if rule.max_samples is not None else selected

    base_candidates = [
        frame_index
        for frame_index in indexes
        if (frame_index - 1) % sample_every_n_frames == 0
    ]
    priority = sorted(boundary_indexes.intersection(indexes))
    maximum = rule.max_samples
    if maximum is None:
        return sorted(set(base_candidates).union(priority))
    if len(priority) >= maximum:
        return uniform_indexes(priority, maximum)
    remaining = maximum - len(priority)
    pool = [frame_index for frame_index in base_candidates if frame_index not in boundary_indexes]
    return sorted(priority + uniform_indexes(pool, min(remaining, len(pool))))


def sample_session_frames(
    labels: Mapping[int, str],
    rules: Mapping[str, SamplingRule],
    boundary: BoundarySampling,
    sample_every_n_frames: int,
) -> list[SampledFrame]:
    boundaries = boundary_frame_indexes(labels, boundary)
    sampled: list[SampledFrame] = []
    states = sorted(set(labels.values()))
    for state in states:
        if state == IGNORE_STATE:
            continue
        try:
            rule = rules[state]
        except KeyError as exc:
            raise ValueError(f"No dataset sampling rule for state {state}") from exc
        indexes = [frame_index for frame_index, label in labels.items() if label == state]
        for frame_index in _sample_state(
            indexes, boundaries, rule, sample_every_n_frames
        ):
            sampled.append(SampledFrame(frame_index, state, frame_index in boundaries))
    return sorted(sampled, key=lambda item: item.frame_index)
