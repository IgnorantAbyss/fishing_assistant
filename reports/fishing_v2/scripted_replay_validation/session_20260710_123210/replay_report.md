# Scripted Prompt Replay: session_20260710_123210

- Result: **FAIL**
- Annotation: **PASS**
- Frames/final: 600 / 600 `PRESS`
- Replay completed: **True**
- Actions applied: **0**
- SYNC_REQUIRED frames: **0**
- Proposed intents: `{'CAST': 39, 'START_HOOK': 53, 'HOOK_ACTION': 4, 'PRESS_SEQUENCE': 1}`
- HOOK qualified/support: 6/17
- PRESS panel/support: 10/10
- PRESS sequence: ready=9, clean=574
- GET qualified/support: 0/0
- False action intents: `{'CAST': [], 'START_HOOK': [], 'HOOK_ACTION': [], 'PRESS_SEQUENCE': []}`

## Warnings

- HOOK: qualified miss 11/17 HOOK frames
- PRESS: panel candidate on 489 non-PRESS frames

## Failures

- unexplained_runtime_transition
- frame 600: terminal_runtime_state_mismatch (PRESS -> IDLE, replay_ended_in_state_inconsistent_with_global_ground_truth)

## Runtime transitions

- frame 10: SYNCING -> IDLE (prompt_consensus)
- frame 50: IDLE -> WAITING (stable_highest_supported_candidate)
- frame 503: WAITING -> READY (stable_highest_supported_candidate)
- frame 557: READY -> HOOK_PENDING (recorded_hook_instruction_acknowledgement)
- frame 563: HOOK_PENDING -> HOOK (stable_strong_hook_evidence)
- frame 574: HOOK -> RESULT_PENDING (recorded_hook_result_prompt_acknowledgement)
- frame 576: RESULT_PENDING -> PRESS (stable_strong_press_evidence)
