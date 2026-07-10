from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class FrameContext:
    frame_index: int
    timestamp: float
    source_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
