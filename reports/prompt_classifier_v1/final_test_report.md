# PromptClassifier v1 Test Report

- Samples: 1182
- Threshold: 0.55
- Accuracy: 0.670897
- Balanced accuracy: 0.520451
- Macro precision / recall / F1: 0.573675 / 0.520451 / 0.432721
- IDLE -> WAITING: 0
- WAITING -> IDLE: 0
- IDLE/WAITING mutual confusion: 0.000000
- UNKNOWN ratio: 0.011844
- Mean / p50 / p95 latency (ms): 0.9165 / 0.8753 / 1.6007
- Error count: 389

## Per class

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| IDLE | 0.000000 | 0.000000 | 0.000000 | 156 |
| WAITING | 0.981605 | 0.981605 | 0.981605 | 598 |
| READY | 0.313093 | 0.937500 | 0.469417 | 176 |
| NONE | 1.000000 | 0.162698 | 0.279863 | 252 |

## Per session

- session_20260710_130308: macro-F1=0.451583, IDLE=0.000000, WAITING=0.974110, READY=0.987342, UNKNOWN=0.006678
- session_20260710_131254: macro-F1=0.414201, IDLE=0.000000, WAITING=0.989619, READY=0.896907, UNKNOWN=0.017153

## Integration gate

- candidate_for_integration: false
- Failure: macro_f1 0.432721 does not satisfy >= 0.85
- Failure: IDLE recall 0.000000 does not satisfy >= 0.85
- Failure: session_20260710_130308 macro-F1 is below 0.75
- Failure: session_20260710_131254 macro-F1 is below 0.75
- Dominant confusions: `NONE -> READY (197)`, `IDLE -> READY (155)`, `NONE -> WAITING (11)`, `WAITING -> READY (10)`, `READY -> IDLE (2)`
- Data type: Natural held-out prompt crops; all test rows including evaluation_only samples.
- Affected sessions: `session_20260710_130308`, `session_20260710_131254`
- Recommendation: Investigate cross-session prompt appearance shift with future training-only data, prioritizing IDLE/NONE versus READY hard examples. Do not tune on this final test.
- No automatic retraining or production integration was performed.
