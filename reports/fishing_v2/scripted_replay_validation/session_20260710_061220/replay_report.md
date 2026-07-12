# Scripted Prompt Replay: session_20260710_061220

- Result: **PASS_WITH_WARNINGS**
- Annotation: **PASS**
- Frames/final: 600 / 600 `WAITING`
- Replay completed: **True**
- Actions applied: **0**
- SYNC_REQUIRED frames: **0**
- Proposed intents: `{'START_HOOK': 26, 'HOOK_ACTION': 12, 'PRESS_SEQUENCE': 1, 'CAST': 27}`
- HOOK qualified/support: 20/27
- PRESS panel/support: 8/11
- PRESS sequence: ready=7, clean=507
- GET qualified/support: 0/0
- False action intents: `{'CAST': [], 'START_HOOK': [], 'HOOK_ACTION': [], 'PRESS_SEQUENCE': []}`

## Warnings

- HOOK: qualified miss 7/27 HOOK frames
- PRESS: panel present 8/11 PRESS frames
- PRESS: panel candidate on 12 non-PRESS frames

## Runtime transitions

- frame 10: SYNCING -> WAITING (prompt_consensus)
- frame 453: WAITING -> READY (stable_highest_supported_candidate)
- frame 480: READY -> HOOK_PENDING (recorded_hook_instruction_acknowledgement)
- frame 487: HOOK_PENDING -> HOOK (stable_strong_hook_evidence)
- frame 507: HOOK -> RESULT_PENDING (recorded_hook_result_prompt_acknowledgement)
- frame 509: RESULT_PENDING -> PRESS (stable_strong_press_evidence)
- frame 515: PRESS -> RESULT_PENDING (press_panel_disappeared)
- frame 531: RESULT_PENDING -> IDLE (stable_highest_supported_candidate)
- frame 560: IDLE -> WAITING (stable_highest_supported_candidate)
