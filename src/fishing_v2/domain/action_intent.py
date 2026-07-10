from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ActionIntent(str, Enum):
    NONE = "NONE"
    CAST = "CAST"
    START_HOOK = "START_HOOK"
    PRESS_SEQUENCE = "PRESS_SEQUENCE"
    COLLECT = "COLLECT"
    PAUSE = "PAUSE"
    STOP = "STOP"


@dataclass(frozen=True)
class ActionRequest:
    intent: ActionIntent
    confidence: float
    reason: str
    payload: Mapping[str, Any] = field(default_factory=dict)
