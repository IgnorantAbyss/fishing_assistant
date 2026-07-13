# Final Prompt Bundle Deployment Validation

- Bundle version: `prototype_v1_final_1`
- Bundle SHA-256: `426dbdd152ce51521604be4893d9864dd3c6a4e4b4495bae08b7e4e1d914d851`
- Prototypes: **35**
- Threshold aggregation: per-class LOSO median; ambiguity LOSO median; IDLE stability LOSO maximum.
- This all-session replay is deployment regression only; it does not replace LOSO generalization results.

| session | final scripted | final bundle | state agreement | false intents | missed events | sync | result |
|---|---|---|---:|---:|---:|---:|---|
| session_20260709_192315 | WAITING | WAITING | 0.9983 | 0 | 0 | 0 | PASS |
| session_20260710_061220 | WAITING | WAITING | 1.0000 | 0 | 0 | 0 | PASS |
| session_20260710_123210 | IDLE | IDLE | 0.9967 | 0 | 0 | 0 | PASS |
| session_20260710_124419 | WAITING | WAITING | 1.0000 | 0 | 0 | 0 | PASS |
| session_20260710_125441 | WAITING | WAITING | 0.9967 | 0 | 0 | 0 | PASS |
| session_20260710_130308 | WAITING | WAITING | 0.9917 | 0 | 0 | 0 | PASS |
| session_20260710_131254 | IDLE | IDLE | 0.9967 | 0 | 0 | 0 | PASS |

- Complete: **7/7**
- False intents: **0**
- Missed expected events: **0**
- SYNC_REQUIRED frames: **0**
- Actions applied: **0**
- Deployment regression passed: **true**
