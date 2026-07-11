# Hybrid Runtime v2 Open Questions

## Fixed decisions

- Supported environment: 2560x1440, fixed UI scale, zh-TW, borderless, fixed Prompt position.
- Runtime Prompt hints: `IDLE_CAST`, `WAITING_IN_PROGRESS`, `READY_BITE`, `HOOK_INSTRUCTION`, `PRESS_INSTRUCTION`, or `UNKNOWN`.
- `IGNORE` is annotation-only.
- Prompt accelerates detector scheduling; specialized panel/bar evidence confirms active states.
- GET panel outranks IDLE Prompt, and CAST requires an absent-Get guard.
- HOOK uses a configurable safe zone and does not target Perfect.
- There is no independent Failure Recovery state.

## Still requiring evidence or implementation

- Explicit user approval of `prompt_final_candidate`; ROI status remains `unapproved`.
- Replay/live detect-only validation of polling frequencies, burst FPS, and the initial 0.65–0.85 HOOK safe zone.
- PromptObserver presence/rejection implementation and confidence calibration.
- Foreground executable/process identity and platform-specific foreground confirmation.
- New untouched sessions for final v2 testing.
- Future action acknowledgement semantics if a real sink is ever separately authorized.

Until resolved, PromptObserver remains unimplemented, action emission remains disabled, and final test status remains `not_collected`.
