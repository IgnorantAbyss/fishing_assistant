# Live Prompt Domain-Gap Diagnosis

## Outcome

- Cause: **prototype coverage / new live variant not covered**.
- The approved ROI `[940, 36, 1620, 100]` is correct and visibly contains the full IDLE prompt.
- Replay and Live share the canonical `uint8 H×W×3 BGR contiguous 0..255` contract.
- MSS input is BGRA; the capture boundary drops alpha and preserves BGR. No RGB swap was found.
- The fix adds one bounded `human_confirmed_live_candidate`; it does not alter Ground Truth or calibration thresholds.

The saved frame is visually the IDLE_CAST instruction `好像釣到什麼，請按下 'Space' 開始吧。`.
Existing medoids show a different instruction such as `在岸邊按下 'Space' 開始進行釣魚吧。`.
Across all 451 existing annotated IDLE frames, the nearest similarity was only `0.52371025`, confirming that the exact visual variant was absent.

## Runtime versus offline reproduction

| path | result | top class | top similarity | second class | second similarity | margin |
|---|---|---|---:|---|---:|---:|
| Original Live event | UNKNOWN | HOOK_INSTRUCTION | 0.526628 | PRESS_INSTRUCTION | 0.515988 | 0.010640 |
| Saved JPEG offline | UNKNOWN | HOOK_INSTRUCTION | 0.527797 | PRESS_INSTRUCTION | 0.519986 | 0.007811 |

The categorical signature, prototype id, and rejection reason match. The small numeric difference is expected because the recorded event and saved diagnostic screenshot are different frames and the screenshot is JPEG. The old event did not log a feature hash; the saved-frame feature hash is `56f2f4b487e0fb8805b3f2a0ccf62e678935961b1610b170d1908a151d8c226e`. Future events now include it.

## Baseline nearest prototypes

| rank | class | session / frame | similarity |
|---:|---|---|---:|
| 1 | HOOK_INSTRUCTION | session_20260709_192315 / 465 | 0.527797 |
| 2 | HOOK_INSTRUCTION | session_20260710_123210 / 565 | 0.525318 |
| 3 | HOOK_INSTRUCTION | session_20260710_061220 / 500 | 0.523899 |
| 4 | HOOK_INSTRUCTION | session_20260710_130308 / 45 | 0.523021 |
| 5 | HOOK_INSTRUCTION | session_20260710_131254 / 216 | 0.522131 |
| 6 | HOOK_INSTRUCTION | session_20260710_125441 / 371 | 0.520995 |
| 7 | PRESS_INSTRUCTION | session_20260710_123210 / 577 | 0.519986 |
| 8 | HOOK_INSTRUCTION | session_20260710_124419 / 322 | 0.518878 |
| 9 | PRESS_INSTRUCTION | session_20260709_192315 / 479 | 0.518145 |
| 10 | WAITING_IN_PROGRESS | session_20260709_192315 / 144 | 0.516911 |

The best formal IDLE medoid ranked 29th at `0.499259`. HOOK/PRESS won because all classes were weak matches, not because Live used a different color format.

## Ambiguity calibration

The margin is the difference between the best and second-best class cosine similarities. Seven LOSO fold values are:

`0.018878, 0.281386, 0.022681, 0.283565, 0.020192, 0.279120, 0.280238`

Their median is correctly `0.2791204953`; the builder and loader preserve it. Lowering it to approximately `0.01` would accept the wrong HOOK top class and therefore cannot fix this failure. No threshold or margin was relaxed.

## Minimal coverage fix

- Candidate: `assets/reference/prompt/live_idle_cast_20260713_000035.png`
- Manifest status: `live_calibration_candidate`
- Review status: `human_confirmed_live_candidate`
- Ground Truth: **false**
- Used for threshold calibration: **false**
- Existing formal medoids preserved: **35/35**
- Candidate prototypes added: **1**
- Bundle: `prototype_v1_final_2`, 36 prototypes
- Bundle SHA-256: `7a6eaff6fc96ea695f55f94daa77a3478f060d62b304e568692a12fbd985964e`

Corrected saved-frame result: `IDLE_CAST`, similarity `0.99999994`, second HOOK `0.52779675`, margin `0.47220320`. Over 30 repeated frames, the existing IDLE temporal guard intentionally returns UNKNOWN for frames 1–3, then IDLE_CAST for frames 4–30; HOOK and PRESS counts remain zero.

## Regression

- LOSO macro F1: **0.992316**
- IGNORE rejection: **1.000000**
- Final bundle predicted Replay: **7/7 PASS**
- False intents / missed events / SYNC_REQUIRED / actions applied: **0 / 0 / 0 / 0**

Diagnostic crops and preprocessing views are in `reports/fishing_v2/live_prompt_diagnostics/`. The original full Live session remains ignored and is not part of the change.
