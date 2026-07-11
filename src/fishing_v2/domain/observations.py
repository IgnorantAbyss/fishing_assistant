"""Frame-scoped perception output; never runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping


class PromptObservationKind(str, Enum):
    IDLE_CAST = "IDLE_CAST"
    WAITING_IN_PROGRESS = "WAITING_IN_PROGRESS"
    READY_BITE = "READY_BITE"
    HOOK_INSTRUCTION = "HOOK_INSTRUCTION"
    PRESS_INSTRUCTION = "PRESS_INSTRUCTION"
    UNKNOWN = "UNKNOWN"


RUNTIME_PROMPT_KINDS = frozenset(PromptObservationKind)


@dataclass(frozen=True)
class PromptObservation:
    kind: PromptObservationKind
    confidence: float
    probabilities: Mapping[str, float]
    source: str
    frame_index: int
    timestamp: float
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HookObservation:
    detected: bool
    confidence: float
    frame_index: int
    timestamp: float
    source: str = "hook_detector"
    fill_ratio: float | None = None
    divider_ratio: float | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PressObservation:
    detected: bool
    confidence: float
    frame_index: int
    timestamp: float
    source: str = "press_detector"
    sequence: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GetObservation:
    detected: bool
    confidence: float
    frame_index: int
    timestamp: float
    source: str = "get_detector"
    evidence: Mapping[str, Any] = field(default_factory=dict)
