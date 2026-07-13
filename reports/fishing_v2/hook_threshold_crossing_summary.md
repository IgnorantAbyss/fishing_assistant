# Hook Threshold Crossing Validation

- Prior gate: `0.65 <= fill_ratio <= 0.85` (incorrect hard upper bound).
- Current primary gate: qualified bar + cyan endpoint past white divider + 10 px.
- Fallback: `fill_ratio >= 0.70` only when divider is unavailable; no upper bound.
- One-shot: proposal guard per Hook episode.

| session / episode | visible | qualified | valid fill | cross | margin | intent | latency | intents |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `session_20260709_192315#1` | 459 | 459 | 459 | 461 | 461 | 461 | 0 | 1 |
| `session_20260710_061220#1` | 486 | 486 | 486 | 488 | 488 | 488 | 0 | 1 |
| `session_20260710_123210#1` | 562 | 562 | 562 | 564 | 564 | 568 | 4 | 1 |
| `session_20260710_124419#1` | 314 | 314 | 314 | 316 | 316 | 323 | 7 | 1 |
| `session_20260710_125441#1` | 366 | 366 | 366 | 368 | 369 | None | None | 0 |
| `session_20260710_130308#1` | 26 | 26 | 26 | 28 | 28 | 29 | 1 | 1 |
| `session_20260710_130308#2` | 385 | 385 | 385 | 387 | 387 | 396 | 9 | 1 |
| `session_20260710_131254#1` | 195 | 195 | 195 | 198 | 198 | 198 | 0 | 1 |
| `session_20260710_131254#2` | 532 | 532 | 532 | 535 | 535 | 535 | 0 | 1 |

## Missing historical observation band

- 124419: `fill_ratio_calibration_error`.
- 125441: `detector_missing_band_frames` (secondary: `fill_ratio_calibration_error`).

## Outside-visible operational risk

- Qualified: 26
- Used by Fusion: 0
- Action-ready: 0
- HOOK_ACTION intents: 0

- Pre-threshold intents: **0**
- Duplicate intents: **0**
- Actions applied: **0**
- `hook_detector_operationally_ready = false`
- Historical replay not fully evaluable: `['session_20260710_125441#1']`
- Live detect-only remaining: Confirm crossing timing and raw Hook qualification continuity, especially the session_20260710_125441 pattern, without applying input.
