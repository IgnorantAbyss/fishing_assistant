# Fishing Assistant Hybrid Runtime v2

## Contract boundaries

Hybrid Runtime v2 keeps three concepts separate:

1. A frame-scoped **observation** reports visible evidence.
2. A time-aware **RuntimeState** records the legal fishing flow and action history.
3. An **ActionIntent** is a proposal that must pass SafetyPolicy; it is not keyboard input.

Action handling is a two-phase commit. `action_proposed` is observable but does not change action-dependent state or action history. Only an allowed request emitted through a real sink becomes `action_applied`; that commit records the action, starts cooldown, and enters its pending/result state. `emit_actions=false`, Safety WAIT/DENY, or a missing sink always leaves `action_applied=false`.

Prompt is a hint. HookBar, PressPanel, and GetPanel are specialized active confirmations. Runtime never derives Prompt from global ground truth, and a Prompt hint alone cannot activate HOOK or PRESS.

Raw detector output is diagnostic data. OFF evidence is marked diagnostic-only, and ARMED/BURST evidence must pass detector-specific qualification before Fusion. Hook rectangle-only candidates and non-positive fill never become active Hook evidence; `bar_fill`, positive `fill_ratio`, and strong confidence are required, while `divider_line` is optional.

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

RESULT_PENDING arms Press/Get and uses medium Prompt polling. PRESS is a three-layer contract: grid/timer geometry creates `PRESS_PANEL_CANDIDATE`, consecutive structural confirmation creates `PRESS_PANEL_PRESENT` and may confirm runtime PRESS, and an eligible pre-input clean frame creates `PRESS_SEQUENCE_READY`. Slot occupancy determines variable sequence length; arrow direction is primary evidence (`LEFT=A`, `DOWN=S`, `RIGHT=D`, `UP=W`) and letter templates are auxiliary only. The earliest clean sequence is frozen for the episode and may propose one `PRESS_SEQUENCE` intent after panel confirmation. Frames after the first input effect cannot replace it. Panel disappearance moves PRESS to RESULT_PENDING. GetPanel has priority over `IDLE_CAST` and confirms GET.

GET keeps proposing COLLECT while GetPanel remains visible and each attempt passes SafetyPolicy. Defaults are 0.4 seconds, 12 attempts, and 5.0 seconds; the reviewed interval range is 0.3–0.5 seconds. Panel disappearance stops COLLECT immediately and enters COLLECT_PENDING. Attempt/duration exhaustion enters SYNC_REQUIRED. COLLECT_PENDING returns to GET if the panel reappears, or reaches IDLE only after the panel is absent and `IDLE_CAST` is stable.

## Safety

`emit_actions` remains false and no keyboard sink exists. SafetyPolicy rejects non-foreground execution, unsupported resolution, duplicate HOOK/PRESS intents, CAST without an absent-Get guard, COLLECT without a visible GetPanel, retries outside cooldown/attempt/duration limits, and every action in SYNC_REQUIRED. The supported environment is exactly 2560x1440, fixed UI scale, zh-TW, borderless, and fixed Prompt position.

`recorded_observation` replay never claims action execution. It reports proposed intent with `action_applied=false`, then follows qualified visual acknowledgements already present in the fixed recording. This validates perception, qualification, Fusion, and state flow without an action timeline or counterfactual screen changes.

## ROI and data boundaries

Prompt ROI uses fixed pixels as the source of truth; normalized coordinates are derived display metadata. `prompt_final_candidate` `[940, 36, 1620, 100]` is approved for the fixed environment after explicit manual review. Other candidates remain historical comparisons.

Global `ground_truth.yaml` remains evaluation-only runtime-state annotation. Human Prompt annotation is independent. This phase creates no official `prompt_ground_truth.yaml`, Prompt dataset, model, or raw replay modification. Legacy specialized detector algorithms remain unchanged behind adapters; v1 PromptClassifier and global StateDetector remain outside v2 core.
