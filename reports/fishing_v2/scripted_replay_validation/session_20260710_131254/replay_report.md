# Scripted Prompt Replay: session_20260710_131254

- Result: **FAIL**
- Annotation: **PASS**
- Frames/final: 600 / 600 `SYNC_REQUIRED`
- Replay completed: **True**
- Actions applied: **0**
- SYNC_REQUIRED frames: **577**
- Proposed intents: `{'COLLECT': 12}`
- HOOK qualified/support: 0/62
- PRESS panel/support: 48/48
- PRESS sequence: ready=0, clean=none
- GET qualified/support: 0/23
- False action intents: `{'CAST': [], 'START_HOOK': [], 'HOOK_ACTION': [], 'PRESS_SEQUENCE': []}`

## Warnings

- HOOK: qualified miss 62/62 HOOK frames
- PRESS: no clean pre-input sequence frame
- PRESS: panel candidate on 546 non-PRESS frames
- GET: qualified miss 23/23 GET frames
- GET: false qualified on 12 non-GET frames

## Failures

- sync_required
- unexplained_runtime_transition
- frame 1: specialized_state_entered_outside_global_support (SYNCING -> GET, strong_get_startup_evidence)
- frame 600: terminal_runtime_state_mismatch (SYNC_REQUIRED -> IDLE, replay_ended_in_state_inconsistent_with_global_ground_truth)

## Runtime transitions

- frame 1: SYNCING -> GET (strong_get_startup_evidence)
- frame 13: GET -> COLLECT_PENDING (get_panel_disappeared)
- frame 24: COLLECT_PENDING -> SYNC_REQUIRED (persistent_conflicting_or_illegal_evidence)
