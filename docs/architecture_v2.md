# Fishing Assistant Hybrid Runtime v2

## Purpose

Hybrid Runtime v2 separates three concepts that v1 mixed together:

1. **Observation** describes what a detector can currently see. It is uncertain, source-labelled, frame-scoped evidence.
2. **RuntimeState** is the controller's time-aware belief about the fishing workflow. It changes only through a transition policy and FSM.
3. **ActionIntent** describes a proposed side effect. It is not an input event and must pass the safety policy.

`READY_PROMPT` therefore does not mean that runtime state is `READY`, and runtime `HOOK` does not imply `NO_PROMPT`. A prompt can remain visible after the game has entered HOOK. This distinction is the core response to the v1 label conflict.

## Why neither a classifier nor an FSM is sufficient alone

A single-frame classifier has no action history, timeout context, stability window, or legal-transition knowledge. It cannot safely overwrite runtime state. Conversely, an FSM cannot see whether the game started in PRESS/GET, whether an action took effect, or whether the UI disagrees with its internal state. The fusion layer converts observations into scored `StateEvidence`; the FSM combines that evidence with history and legal transitions.

## Dependency direction

```text
domain <- ports <- perception <- fusion <- runtime
   ^          ^                    ^         ^
   |          +--- legacy_adapters +---------+
   +--- data / replay orchestration (composition only)
```

The domain, fusion, and runtime packages must not import legacy detectors, `state_detector`, `state_smoother`, template prompt logic, or `src.ml`. Only `legacy_adapters` may import specialized legacy detector implementations. Runtime consumes port contracts and new observations, never legacy dictionaries.

## Startup synchronization

Runtime begins in `SYNCING` and observes a bounded window (default 10 frames/5 seconds). Strong specialized GET/PRESS/HOOK evidence can synchronize immediately. High-confidence prompt consensus can synchronize IDLE/WAITING/READY. `OTHER_PROMPT`, `NO_PROMPT`, missing prompt observers, or insufficient consensus cannot invent a global state. Timeout produces `SYNC_REQUIRED`. A caller may explicitly supply a manual start-state override.

## Normal flow

```text
IDLE -> CAST_PENDING -> WAITING -> READY -> HOOK_PENDING -> HOOK
HOOK -> PRESS -> POST_CATCH -> IDLE
HOOK -> GET -> COLLECT_PENDING -> IDLE
PRESS -> GET or POST_CATCH
```

The no-GET route is legal. Pending states remember emitted intents and wait for evidence/timeout rather than repeatedly emitting actions.

## Evidence conflicts and recovery

Fusion scores every candidate and records both supporting and conflicting observations. Strong specialized evidence plus a legal current flow can outweigh a residual prompt (for example HOOK + READY_PROMPT). One weak IDLE prompt cannot pull WAITING directly to IDLE. `NO_PROMPT` and `UNKNOWN` never fall back to IDLE. Persistent conflict or illegal/time-expired transitions enter `SYNC_REQUIRED`, which emits no game action.

## Safety policy

Every `ActionIntent` passes an independent policy checking sync status, legal runtime state, evidence confidence/conflicts, per-state deduplication, cooldown, foreground-window status, and the global `emit_actions` switch. This phase fixes `emit_actions: false`; the action sink is a recording port only and no keyboard sink exists.

## Desynchronization handling

The controller tracks how long evidence conflicts with runtime state. Before the timeout it waits; after the timeout it enters `SYNC_REQUIRED`. Recovery requires a new synchronization window or explicit override. It never guesses IDLE merely because all detectors are quiet.

## Legacy reuse through adapters

Capture/replay readers and global ground-truth validation remain infrastructure/reference tools. HOOK, PRESS, and GET specialized detectors are wrapped by thin adapters that preserve confidence/debug evidence and convert exceptions into non-detected observations. Legacy prompt fusion, `StateDetector`, `StateSmoother`, v1 PromptClassifier output/threshold, and global-state-to-prompt mappings are excluded from the v2 core.

## Data boundaries

Global `ground_truth.yaml` remains the runtime-state annotation. Human-authored `prompt_ground_truth.yaml` describes only visible prompt content and may have different boundaries. New crops go only to `datasets/prompt_observation_v1`; `datasets/fishing_v2` is a frozen lineage reference. Sessions previously inspected as v1 test data are development/diagnostic data in v2. A new v2 final test is `not_collected` and is never auto-selected.
