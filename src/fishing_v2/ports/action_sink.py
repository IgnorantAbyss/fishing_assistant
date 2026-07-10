from typing import Protocol

from src.fishing_v2.domain.action_intent import ActionRequest


class ActionSink(Protocol):
    def emit(self, request: ActionRequest) -> None: ...
