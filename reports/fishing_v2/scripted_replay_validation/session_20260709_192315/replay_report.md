# Scripted Prompt Replay: session_20260709_192315

- Result: **PASS_WITH_WARNINGS**
- Annotation: **PASS**
- Frames/final: 600 / 600 `WAITING`
- Replay completed: **True**
- Actions applied: **0**
- SYNC_REQUIRED frames: **0**
- Proposed intents: `{'CAST': 41, 'START_HOOK': 9, 'HOOK_ACTION': 8, 'PRESS_SEQUENCE': 1}`
- HOOK qualified/support: 13/17
- PRESS panel/support: 11/12
- PRESS sequence: ready=10, clean=472
- GET qualified/support: 0/16
- False action intents: `{'CAST': [], 'START_HOOK': [], 'HOOK_ACTION': [], 'PRESS_SEQUENCE': []}`

## Warnings

- HOOK: qualified miss 4/17 HOOK frames
- PRESS: panel present 11/12 PRESS frames
- PRESS: panel candidate on 588 non-PRESS frames
- GET: qualified miss 16/16 GET frames

## Runtime transitions

- frame 10: SYNCING -> IDLE (prompt_consensus)
- frame 27: IDLE -> WAITING (stable_highest_supported_candidate)
- frame 444: WAITING -> READY (stable_highest_supported_candidate)
- frame 454: READY -> HOOK_PENDING (recorded_hook_instruction_acknowledgement)
- frame 460: HOOK_PENDING -> HOOK (stable_strong_hook_evidence)
- frame 472: HOOK -> RESULT_PENDING (recorded_hook_result_prompt_acknowledgement)
- frame 474: RESULT_PENDING -> PRESS (stable_strong_press_evidence)
- frame 499: PRESS -> RESULT_PENDING (recorded_press_result_prompt_acknowledgement)
- frame 501: RESULT_PENDING -> IDLE (stable_highest_supported_candidate)
- frame 528: IDLE -> WAITING (stable_highest_supported_candidate)
