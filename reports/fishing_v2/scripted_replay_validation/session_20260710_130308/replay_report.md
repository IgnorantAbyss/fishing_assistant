# Scripted Prompt Replay: session_20260710_130308

- Result: **PASS_WITH_WARNINGS**
- Annotation: **PASS**
- Frames/final: 600 / 600 `WAITING`
- Replay completed: **True**
- Actions applied: **0**
- SYNC_REQUIRED frames: **0**
- Proposed intents: `{'START_HOOK': 68, 'HOOK_ACTION': 3, 'CAST': 85, 'PRESS_SEQUENCE': 1, 'COLLECT': 18}`
- HOOK qualified/support: 50/76
- PRESS panel/support: 24/24
- PRESS sequence: ready=23, clean=415
- GET qualified/support: 19/19
- False action intents: `{'CAST': [], 'START_HOOK': [], 'HOOK_ACTION': [], 'PRESS_SEQUENCE': []}`

## Warnings

- HOOK: qualified miss 26/76 HOOK frames
- PRESS: panel candidate on 483 non-PRESS frames

## Runtime transitions

- frame 10: SYNCING -> READY (prompt_consensus)
- frame 21: READY -> HOOK_PENDING (recorded_hook_instruction_acknowledgement)
- frame 27: HOOK_PENDING -> HOOK (stable_strong_hook_evidence)
- frame 63: HOOK -> RESULT_PENDING (recorded_hook_result_prompt_acknowledgement)
- frame 65: RESULT_PENDING -> IDLE (stable_highest_supported_candidate)
- frame 101: IDLE -> WAITING (stable_highest_supported_candidate)
- frame 322: WAITING -> READY (stable_highest_supported_candidate)
- frame 380: READY -> HOOK_PENDING (recorded_hook_instruction_acknowledgement)
- frame 388: HOOK_PENDING -> HOOK (stable_strong_hook_evidence)
- frame 415: HOOK -> RESULT_PENDING (recorded_hook_result_prompt_acknowledgement)
- frame 417: RESULT_PENDING -> PRESS (stable_strong_press_evidence)
- frame 439: PRESS -> RESULT_PENDING (press_panel_disappeared)
- frame 440: RESULT_PENDING -> GET (get_panel_priority)
- frame 459: GET -> COLLECT_PENDING (get_panel_disappeared)
- frame 461: COLLECT_PENDING -> IDLE (stable_highest_supported_candidate)
- frame 514: IDLE -> WAITING (stable_highest_supported_candidate)
