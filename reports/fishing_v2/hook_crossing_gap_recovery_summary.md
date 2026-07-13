# Hook Crossing Gap Recovery

- Policy: **level-triggered one-shot**; crossing edge is diagnostic only.
- Action requires current usable geometry; stale geometry is never cached.
- Legacy fill_ratio is untrusted and cannot trigger fallback, including `1.0`.
- 125441: first usable post-cross evidence/intention = **369 / 369**.

| episode | cross | margin | first usable | intent | usable latency | intents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `session_20260709_192315#1` | 461 | 461 | 461 | 461 | 0 | 1 |
| `session_20260710_061220#1` | 488 | 488 | 488 | 488 | 0 | 1 |
| `session_20260710_123210#1` | 564 | 564 | 564 | 564 | 0 | 1 |
| `session_20260710_124419#1` | 316 | 316 | 316 | 316 | 0 | 1 |
| `session_20260710_125441#1` | 368 | 369 | 369 | 369 | 0 | 1 |
| `session_20260710_130308#1` | 28 | 28 | 28 | 28 | 0 | 1 |
| `session_20260710_130308#2` | 387 | 387 | 396 | 396 | 0 | 1 |
| `session_20260710_131254#1` | 198 | 198 | 198 | 198 | 0 | 1 |
| `session_20260710_131254#2` | 535 | 535 | 535 | 535 | 0 | 1 |

## Gates

- Total episode intents: **9/9**
- Pre-threshold intents: **0**
- Duplicate intents: **0**
- Outside-visible intents: **0**
- Actions applied: **0**
- `hook_detector_operationally_ready = true`

## Live detect-only remaining

- Confirm fresh geometry remains stable at the real capture cadence.
- Confirm disappearance/timeout latch clearing without applying input.
- Validate ratio fallback only after a future source explicitly marks its ratio calibrated.
