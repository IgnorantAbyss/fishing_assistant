# Scripted Prompt Replay Validation

| session | annotation | replay | final state | sync frames | hook | press panel | press sequence | get | result |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| `session_20260709_192315` | PASS | PASS | `WAITING` | 0 | 13/17 | 11/12 | ready=10, clean=472 | 14/16 | **PASS_WITH_WARNINGS** |
| `session_20260710_061220` | PASS | PASS | `WAITING` | 0 | 20/27 | 8/11 | ready=7, clean=507 | 0/0 | **PASS_WITH_WARNINGS** |
| `session_20260710_123210` | PASS | PASS | `IDLE` | 0 | 6/17 | 10/10 | ready=9, clean=574 | 0/0 | **PASS_WITH_WARNINGS** |
| `session_20260710_124419` | PASS | PASS | `WAITING` | 0 | 9/25 | 22/22 | ready=0, clean=none | 21/23 | **PASS_WITH_WARNINGS** |
| `session_20260710_125441` | PASS | PASS | `WAITING` | 0 | 17/35 | 24/24 | ready=0, clean=none | 16/18 | **PASS_WITH_WARNINGS** |
| `session_20260710_130308` | PASS | PASS | `WAITING` | 0 | 50/76 | 24/24 | ready=23, clean=415 | 18/19 | **PASS_WITH_WARNINGS** |
| `session_20260710_131254` | PASS | PASS | `IDLE` | 0 | 49/62 | 48/48 | ready=24, clean=225 | 21/23 | **PASS_WITH_WARNINGS** |

- Annotation PASS: **7/7**
- Replay complete: **7/7**
- PASS / PASS_WITH_WARNINGS / FAIL: **0 / 7 / 0**
- Actions applied: **0**
- SYNC_REQUIRED frames: **0**
- HOOK raw / qualified / used / support: **177 / 164 / 164 / 259**
- PRESS candidate / present / sequence-ready / intents / support: **148 / 147 / 73 / 5 / 151**
- GET raw / qualified / used / support: **95 / 90 / 90 / 99**
- False action intents present: **False**
- Unexplained Runtime transitions: **0**
- Trial excluded: `session_20260709_192231`
- Specialized warning impact: Warnings remain relevant to Runtime Fusion, but do not invalidate future PromptObserver prediction scoring against independent human Prompt annotations.
- `ready_for_prompt_observer = true`

## Detector warnings

- `session_20260709_192315` HOOK: qualified miss 4/17 HOOK frames
- `session_20260709_192315` PRESS: panel present 11/12 PRESS frames
- `session_20260709_192315` PRESS: panel candidate on 588 non-PRESS frames
- `session_20260709_192315` GET: qualified miss 2/16 GET frames
- `session_20260710_061220` HOOK: qualified miss 7/27 HOOK frames
- `session_20260710_061220` PRESS: panel present 8/11 PRESS frames
- `session_20260710_061220` PRESS: panel candidate on 12 non-PRESS frames
- `session_20260710_123210` HOOK: qualified miss 11/17 HOOK frames
- `session_20260710_123210` PRESS: panel candidate on 489 non-PRESS frames
- `session_20260710_124419` HOOK: qualified miss 16/25 HOOK frames
- `session_20260710_124419` PRESS: no clean pre-input sequence frame
- `session_20260710_124419` PRESS: panel candidate on 540 non-PRESS frames
- `session_20260710_124419` PRESS: panel present on 1 non-PRESS frames
- `session_20260710_124419` GET: qualified miss 2/23 GET frames
- `session_20260710_125441` HOOK: qualified miss 18/35 HOOK frames
- `session_20260710_125441` PRESS: no clean pre-input sequence frame
- `session_20260710_125441` PRESS: panel candidate on 550 non-PRESS frames
- `session_20260710_125441` GET: qualified miss 2/18 GET frames
- `session_20260710_130308` HOOK: qualified miss 26/76 HOOK frames
- `session_20260710_130308` PRESS: panel candidate on 483 non-PRESS frames
- `session_20260710_130308` GET: qualified miss 1/19 GET frames
- `session_20260710_131254` HOOK: qualified miss 13/62 HOOK frames
- `session_20260710_131254` PRESS: panel candidate on 546 non-PRESS frames
- `session_20260710_131254` GET: qualified miss 2/23 GET frames

## SYNC_REQUIRED and Runtime issues

- None
