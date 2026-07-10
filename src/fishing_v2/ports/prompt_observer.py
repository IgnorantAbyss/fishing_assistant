from typing import Any, Protocol

from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.observations import PromptObservation


class PromptObserver(Protocol):
    def observe(self, frame: Any, context: FrameContext) -> PromptObservation: ...
