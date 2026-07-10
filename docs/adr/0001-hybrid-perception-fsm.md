# ADR 0001: Hybrid Perception and FSM

Status: accepted

Per-frame perception supplies uncertain evidence; a legal-transition FSM supplies temporal/action context. Neither may replace the other. Fusion produces `StateEvidence`, and only the FSM updates runtime state. This prevents one noisy image from controlling actions while still allowing startup in a specialized UI state.
