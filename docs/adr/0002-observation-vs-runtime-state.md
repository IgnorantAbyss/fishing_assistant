# ADR 0002: Observation Is Not Runtime State

Status: accepted

Observation, RuntimeState, and ActionIntent are distinct enums/dataclasses. A visible READY prompt may coexist with runtime HOOK, and a proposed START_HOOK action is neither of those facts. Direct enum conversion or equality-based overwrites are prohibited and tested.
