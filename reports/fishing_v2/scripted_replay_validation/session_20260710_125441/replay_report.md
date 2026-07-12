# Scripted Prompt Replay: session_20260710_125441

- Result: **FAIL**
- Annotation: **PASS**
- Frames/final: 600 / 600 `SYNC_REQUIRED`
- Replay completed: **True**
- Actions applied: **0**
- SYNC_REQUIRED frames: **178**
- Proposed intents: `{'COLLECT': 10, 'CAST': 45, 'START_HOOK': 85}`
- HOOK qualified/support: 17/35
- PRESS panel/support: 24/24
- PRESS sequence: ready=0, clean=none
- GET qualified/support: 0/18
- False action intents: `{'CAST': [], 'START_HOOK': [], 'HOOK_ACTION': [], 'PRESS_SEQUENCE': []}`

## Warnings

- HOOK: qualified miss 18/35 HOOK frames
- PRESS: no clean pre-input sequence frame
- PRESS: panel candidate on 550 non-PRESS frames
- GET: qualified miss 18/18 GET frames
- GET: false qualified on 11 non-GET frames

## Failures

- sync_required
- unexplained_runtime_transition
- frame 9: specialized_state_entered_outside_global_support (SYNCING -> GET, strong_get_startup_evidence)
- frame 394: specialized_state_entered_outside_global_support (HOOK -> GET, get_panel_priority)
- frame 600: terminal_runtime_state_mismatch (SYNC_REQUIRED -> WAITING, replay_ended_in_state_inconsistent_with_global_ground_truth)

## Runtime transitions

- frame 9: SYNCING -> GET (strong_get_startup_evidence)
- frame 11: GET -> COLLECT_PENDING (get_panel_disappeared)
- frame 13: COLLECT_PENDING -> IDLE (stable_highest_supported_candidate)
- frame 60: IDLE -> WAITING (stable_highest_supported_candidate)
- frame 275: WAITING -> READY (stable_highest_supported_candidate)
- frame 361: READY -> HOOK_PENDING (recorded_hook_instruction_acknowledgement)
- frame 367: HOOK_PENDING -> HOOK (stable_strong_hook_evidence)
- frame 394: HOOK -> GET (get_panel_priority)
- frame 403: GET -> COLLECT_PENDING (get_panel_disappeared)
- frame 423: COLLECT_PENDING -> SYNC_REQUIRED (collect_pending_timeout)
