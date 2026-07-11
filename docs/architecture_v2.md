# Fishing Assistant Hybrid Runtime v2

## Contract boundaries

Hybrid Runtime v2 keeps three concepts separate:

1. A frame-scoped **observation** reports visible evidence.
2. A time-aware **RuntimeState** records the legal fishing flow and action history.
3. An **ActionIntent** is a proposal that must pass SafetyPolicy; it is not keyboard input.

Prompt is a hint. HookBar, PressPanel, and GetPanel are specialized active confirmations. Runtime never derives Prompt from global ground truth, and a Prompt hint alone cannot activate HOOK or PRESS.

## Final Prompt contract

The runtime-facing Prompt observer may emit only:

- `IDLE_CAST`
- `WAITING_IN_PROGRESS`
- `READY_BITE`
- `HOOK_INSTRUCTION`
- `PRESS_INSTRUCTION`
- `UNKNOWN`

`IGNORE` is annotation-only. Deprecated version-1 annotation names remain readable for lineage, but the new writer rejects them and replay maps them to `UNKNOWN`. `OTHER_PROMPT` and `NO_PROMPT` are not v2 classifier targets. Central result notifications do not change the Prompt label when the top Prompt ROI remains visible. Future presence/rejection logic belongs inside PromptObserver; it is not a RuntimeState.

## Detector activation

| Mode | Meaning |
| --- | --- |
| `OFF` | The UI is not possible in the current flow; do not run the detector. |
| `ARMED` | Action history/runtime state makes the UI possible; run at low frequency without acting. |
| `BURST` | A matching Prompt hint accelerates detection; Prompt still cannot authorize action. |
| `ACTIVE` | The specialized detector confirms the real panel/bar and may confirm the corresponding state. |

After `START_HOOK`, Hook is ARMED; `HOOK_INSTRUCTION` makes it BURST; only HookBar makes it ACTIVE and confirms HOOK. After the safe-zone hook intent, Press and Get are ARMED; `PRESS_INSTRUCTION` makes Press BURST; only PressPanel makes PRESS ACTIVE. Initial FPS values live in config and require replay/live detect-only validation.

## Final runtime flow

```text
SYNCING
  -> IDLE -> CAST_PENDING -> WAITING -> READY -> HOOK_PENDING -> HOOK
  -> RESULT_PENDING -> PRESS -> RESULT_PENDING
  -> RESULT_PENDING -> GET -> COLLECT_PENDING -> IDLE
  -> SYNC_REQUIRED on bounded timeout/conflict/retry exhaustion
```

The complete state set is `SYNCING`, `IDLE`, `CAST_PENDING`, `WAITING`, `READY`, `HOOK_PENDING`, `HOOK`, `RESULT_PENDING`, `PRESS`, `GET`, `COLLECT_PENDING`, and `SYNC_REQUIRED`. There is no separate Failure Recovery state; `FAILED` is telemetry while RESULT_PENDING continues waiting.

Startup prioritizes strong GetPanel, then PressPanel, then HookBar. Stable `IDLE_CAST`, `WAITING_IN_PROGRESS`, and `READY_BITE` may synchronize their matching states. Other evidence continues synchronization and never guesses IDLE.

In IDLE, a stable `IDLE_CAST` is necessary but not sufficient for CAST. A current GetPanel observation is the guard: present cancels CAST and enters GET; confirmed absent permits one CAST intent. CAST_PENDING accepts WAITING or READY and otherwise times out to SYNC_REQUIRED. WAITING uses configurable low-frequency Prompt polling; WAITING/UNKNOWN holds state, and an IDLE hint cannot pull it backward.

READY proposes `START_HOOK` once and enters HOOK_PENDING. A hook instruction only selects BURST. HookBar confirms HOOK. HOOK uses the configured `fill_ratio` safe zone, emits one `HOOK_ACTION`, and moves to RESULT_PENDING. It deliberately does not chase the detector's 0.95 Perfect zone.

RESULT_PENDING arms Press/Get and uses medium Prompt polling. PressPanel confirms PRESS; a parsed sequence is proposed once, then action history prevents residual-panel duplication. GetPanel has priority over `IDLE_CAST` and confirms GET.

GET keeps proposing COLLECT while GetPanel remains visible and each attempt passes SafetyPolicy. Defaults are 0.4 seconds, 12 attempts, and 5.0 seconds; the reviewed interval range is 0.3–0.5 seconds. Panel disappearance stops COLLECT immediately and enters COLLECT_PENDING. Attempt/duration exhaustion enters SYNC_REQUIRED. COLLECT_PENDING returns to GET if the panel reappears, or reaches IDLE only after the panel is absent and `IDLE_CAST` is stable.

## Safety

`emit_actions` remains false and no keyboard sink exists. SafetyPolicy rejects non-foreground execution, unsupported resolution, duplicate HOOK/PRESS intents, CAST without an absent-Get guard, COLLECT without a visible GetPanel, retries outside cooldown/attempt/duration limits, and every action in SYNC_REQUIRED. The supported environment is exactly 2560x1440, fixed UI scale, zh-TW, borderless, and fixed Prompt position.

## ROI and data boundaries

Prompt ROI uses fixed pixels as the source of truth; normalized coordinates are derived display metadata. `prompt_final_candidate` is `[940, 36, 1620, 100]` but remains unapproved. The compact review is manual evidence, not classifier evaluation or approval.

Global `ground_truth.yaml` remains evaluation-only runtime-state annotation. Human Prompt annotation is independent. This phase creates no official `prompt_ground_truth.yaml`, Prompt dataset, model, or raw replay modification. Legacy specialized detector algorithms remain unchanged behind adapters; v1 PromptClassifier and global StateDetector remain outside v2 core.
