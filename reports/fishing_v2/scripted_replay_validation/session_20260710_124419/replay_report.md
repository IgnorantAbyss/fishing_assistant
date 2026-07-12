# Scripted Prompt Replay: session_20260710_124419

- Result: **FAIL**
- Annotation: **PASS**
- Frames/final: 600 / 600 `SYNC_REQUIRED`
- Replay completed: **True**
- Actions applied: **0**
- SYNC_REQUIRED frames: **232**
- Proposed intents: `{'CAST': 35, 'START_HOOK': 50, 'COLLECT': 32}`
- HOOK qualified/support: 2/25
- PRESS panel/support: 22/22
- PRESS sequence: ready=0, clean=none
- GET qualified/support: 0/23
- False action intents: `{'CAST': [], 'START_HOOK': [], 'HOOK_ACTION': [], 'PRESS_SEQUENCE': []}`

## Warnings

- HOOK: qualified miss 23/25 HOOK frames
- PRESS: no clean pre-input sequence frame
- PRESS: panel candidate on 540 non-PRESS frames
- PRESS: panel present on 1 non-PRESS frames
- GET: qualified miss 23/23 GET frames
- GET: false qualified on 33 non-GET frames

## Failures

- sync_required
- unexplained_runtime_transition
- frame 316: specialized_state_entered_outside_global_support (HOOK -> GET, get_panel_priority)
- frame 600: terminal_runtime_state_mismatch (SYNC_REQUIRED -> WAITING, replay_ended_in_state_inconsistent_with_global_ground_truth)

## Runtime transitions

- frame 10: SYNCING -> IDLE (prompt_consensus)
- frame 46: IDLE -> WAITING (stable_highest_supported_candidate)
- frame 258: WAITING -> READY (stable_highest_supported_candidate)
- frame 309: READY -> HOOK_PENDING (recorded_hook_instruction_acknowledgement)
- frame 315: HOOK_PENDING -> HOOK (stable_strong_hook_evidence)
- frame 316: HOOK -> GET (get_panel_priority)
- frame 349: GET -> COLLECT_PENDING (get_panel_disappeared)
- frame 369: COLLECT_PENDING -> SYNC_REQUIRED (collect_pending_timeout)
