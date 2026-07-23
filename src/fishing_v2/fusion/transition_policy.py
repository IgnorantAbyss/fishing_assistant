from __future__ import annotations

from src.fishing_v2.domain.runtime_state import RuntimeState


LEGAL_TRANSITIONS: dict[RuntimeState, frozenset[RuntimeState]] = {
    RuntimeState.SYNCING: frozenset({
        RuntimeState.IDLE, RuntimeState.WAITING, RuntimeState.READY,
        RuntimeState.HOOK, RuntimeState.PRESS, RuntimeState.GET,
        RuntimeState.SYNC_REQUIRED,
    }),
    RuntimeState.IDLE: frozenset({RuntimeState.CAST_PENDING, RuntimeState.WAITING, RuntimeState.GET}),
    RuntimeState.CAST_PENDING: frozenset({RuntimeState.WAITING, RuntimeState.READY, RuntimeState.GET, RuntimeState.SYNC_REQUIRED}),
    RuntimeState.WAITING: frozenset({
        RuntimeState.READY,
        RuntimeState.HOOK_PENDING,
        RuntimeState.SYNC_REQUIRED,
    }),
    RuntimeState.READY: frozenset({RuntimeState.HOOK_PENDING, RuntimeState.HOOK, RuntimeState.GET, RuntimeState.SYNC_REQUIRED}),
    RuntimeState.HOOK_PENDING: frozenset({RuntimeState.HOOK, RuntimeState.GET, RuntimeState.SYNC_REQUIRED}),
    RuntimeState.HOOK: frozenset({RuntimeState.RESULT_PENDING, RuntimeState.PRESS, RuntimeState.GET, RuntimeState.SYNC_REQUIRED}),
    RuntimeState.RESULT_PENDING: frozenset({RuntimeState.PRESS, RuntimeState.GET, RuntimeState.IDLE, RuntimeState.SYNC_REQUIRED}),
    RuntimeState.PRESS: frozenset({RuntimeState.RESULT_PENDING, RuntimeState.GET, RuntimeState.SYNC_REQUIRED}),
    RuntimeState.GET: frozenset({RuntimeState.COLLECT_PENDING, RuntimeState.SYNC_REQUIRED}),
    RuntimeState.COLLECT_PENDING: frozenset({RuntimeState.GET, RuntimeState.IDLE, RuntimeState.SYNC_REQUIRED}),
    RuntimeState.SYNC_REQUIRED: frozenset({RuntimeState.SYNCING}),
}


class TransitionPolicy:
    def is_legal(self, current: RuntimeState, target: RuntimeState) -> bool:
        return target == current or target in LEGAL_TRANSITIONS.get(current, frozenset())

    def require_legal(self, current: RuntimeState, target: RuntimeState) -> None:
        if not self.is_legal(current, target):
            raise ValueError(f"Illegal v2 runtime transition: {current.value} -> {target.value}")
