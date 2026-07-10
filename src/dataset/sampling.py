"""Deterministic, boundary-aware per-session frame sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from src.dataset import BoundarySampling, SamplingRule
from src.replay_ground_truth import IGNORE_STATE


@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    original_state: str
    is_boundary: bool


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
