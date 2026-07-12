# Scripted Prompt Replay Validation

| session | annotation | replay | final state | sync frames | hook | press panel | press sequence | get | result |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| `session_20260709_192315` | PASS | PASS | `WAITING` | 0 | 13/17 | 11/12 | ready=10, clean=472 | 0/16 | **PASS_WITH_WARNINGS** |
| `session_20260710_061220` | PASS | PASS | `WAITING` | 0 | 20/27 | 8/11 | ready=7, clean=507 | 0/0 | **PASS_WITH_WARNINGS** |
| `session_20260710_123210` | PASS | PASS | `PRESS` | 0 | 6/17 | 10/10 | ready=9, clean=574 | 0/0 | **FAIL** |
| `session_20260710_124419` | PASS | PASS | `SYNC_REQUIRED` | 232 | 2/25 | 22/22 | ready=0, clean=none | 0/23 | **FAIL** |
| `session_20260710_125441` | PASS | PASS | `SYNC_REQUIRED` | 178 | 17/35 | 24/24 | ready=0, clean=none | 0/18 | **FAIL** |
| `session_20260710_130308` | PASS | PASS | `WAITING` | 0 | 50/76 | 24/24 | ready=23, clean=415 | 19/19 | **PASS_WITH_WARNINGS** |
| `session_20260710_131254` | PASS | PASS | `SYNC_REQUIRED` | 577 | 0/62 | 48/48 | ready=0, clean=none | 0/23 | **FAIL** |

- Annotation PASS: **7/7**
- Replay complete: **7/7**
- PASS / PASS_WITH_WARNINGS / FAIL: **0 / 3 / 4**
- Actions applied: **0**
- SYNC_REQUIRED frames: **987**
- HOOK raw / qualified / used / support: **177 / 108 / 108 / 259**
- PRESS candidate / present / sequence-ready / intents / support: **148 / 147 / 49 / 4 / 151**
- GET raw / qualified / used / support: **42 / 19 / 19 / 99**
- False action intents present: **False**
- Unexplained Runtime transitions: **8**
- Trial excluded: `session_20260709_192231`
- Specialized warning impact: Warnings remain relevant to Runtime Fusion, but do not invalidate future PromptObserver prediction scoring against independent human Prompt annotations.
- `ready_for_prompt_observer = false`

## Detector warnings

- `session_20260709_192315` HOOK: qualified miss 4/17 HOOK frames
- `session_20260709_192315` PRESS: panel present 11/12 PRESS frames
- `session_20260709_192315` PRESS: panel candidate on 588 non-PRESS frames
- `session_20260709_192315` GET: qualified miss 16/16 GET frames
- `session_20260710_061220` HOOK: qualified miss 7/27 HOOK frames
- `session_20260710_061220` PRESS: panel present 8/11 PRESS frames
- `session_20260710_061220` PRESS: panel candidate on 12 non-PRESS frames
- `session_20260710_123210` HOOK: qualified miss 11/17 HOOK frames
- `session_20260710_123210` PRESS: panel candidate on 489 non-PRESS frames
- `session_20260710_124419` HOOK: qualified miss 23/25 HOOK frames
- `session_20260710_124419` PRESS: no clean pre-input sequence frame
- `session_20260710_124419` PRESS: panel candidate on 540 non-PRESS frames
- `session_20260710_124419` PRESS: panel present on 1 non-PRESS frames
- `session_20260710_124419` GET: qualified miss 23/23 GET frames
- `session_20260710_124419` GET: false qualified on 33 non-GET frames
- `session_20260710_125441` HOOK: qualified miss 18/35 HOOK frames
- `session_20260710_125441` PRESS: no clean pre-input sequence frame
- `session_20260710_125441` PRESS: panel candidate on 550 non-PRESS frames
- `session_20260710_125441` GET: qualified miss 18/18 GET frames
- `session_20260710_125441` GET: false qualified on 11 non-GET frames
- `session_20260710_130308` HOOK: qualified miss 26/76 HOOK frames
- `session_20260710_130308` PRESS: panel candidate on 483 non-PRESS frames
- `session_20260710_131254` HOOK: qualified miss 62/62 HOOK frames
- `session_20260710_131254` PRESS: no clean pre-input sequence frame
- `session_20260710_131254` PRESS: panel candidate on 546 non-PRESS frames
- `session_20260710_131254` GET: qualified miss 23/23 GET frames
- `session_20260710_131254` GET: false qualified on 12 non-GET frames

## SYNC_REQUIRED and Runtime issues

- `session_20260710_123210` frame 600: terminal_runtime_state_mismatch (PRESS -> IDLE, replay_ended_in_state_inconsistent_with_global_ground_truth)
- `session_20260710_124419` frame 369: collect_pending_timeout
- `session_20260710_124419` frame 316: specialized_state_entered_outside_global_support (HOOK -> GET, get_panel_priority)
- `session_20260710_124419` frame 600: terminal_runtime_state_mismatch (SYNC_REQUIRED -> WAITING, replay_ended_in_state_inconsistent_with_global_ground_truth)
- `session_20260710_125441` frame 423: collect_pending_timeout
- `session_20260710_125441` frame 9: specialized_state_entered_outside_global_support (SYNCING -> GET, strong_get_startup_evidence)
- `session_20260710_125441` frame 394: specialized_state_entered_outside_global_support (HOOK -> GET, get_panel_priority)
- `session_20260710_125441` frame 600: terminal_runtime_state_mismatch (SYNC_REQUIRED -> WAITING, replay_ended_in_state_inconsistent_with_global_ground_truth)
- `session_20260710_131254` frame 24: persistent_conflicting_or_illegal_evidence
- `session_20260710_131254` frame 1: specialized_state_entered_outside_global_support (SYNCING -> GET, strong_get_startup_evidence)
- `session_20260710_131254` frame 600: terminal_runtime_state_mismatch (SYNC_REQUIRED -> IDLE, replay_ended_in_state_inconsistent_with_global_ground_truth)
