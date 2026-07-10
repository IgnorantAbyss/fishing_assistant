from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class RuntimeEventKind(str, Enum):
    OBSERVATION = "OBSERVATION"
    ACTION_PROPOSED = "ACTION_PROPOSED"
    STATE_CHANGED = "STATE_CHANGED"
    TIMEOUT = "TIMEOUT"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class RuntimeEvent:
    kind: RuntimeEventKind
    timestamp: float
    details: Mapping[str, Any] = field(default_factory=dict)
