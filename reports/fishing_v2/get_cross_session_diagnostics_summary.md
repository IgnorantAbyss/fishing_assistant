# GET Cross-session Diagnostics

## Outcome

- Known false-positive transition frames `124419:316`, `125441:9`, `125441:394`, and `131254:1` are no longer raw or qualified GET.
- Qualified false-positive frames across the seven replays fell from **56 to 0**.
- True GET raw / qualified / Fusion detections improved from **42 / 19 / 19** to **95 / 90 / 90** of 99 support frames.
- All seven replays completed with **0 actions applied** and **0 SYNC_REQUIRED frames** after the fix.
- `ready_for_prompt_observer` remains false only because `123210` still ends in PRESS; that separate PRESS-exit issue was not changed.

## Root causes

| case | before evidence | concrete cause | after |
| --- | --- | --- | --- |
| `124419:316` | bbox `[97,192,379,418]`; title/grid/button structure `1.0/1.0/1.0` | Localizer selected the low, bottom-clipped fishing UI instead of the fixed inventory panel. Bright prompt text saturated title/button ratios and ordinary UI edges produced 10 false grid cells. | Geometry rejected; raw/qualified/Fusion `false/false/false`. |
| `125441:9` | bbox `[108,267,379,418]`; structure `1.0/0.875/1.0` | Low fishing/quest UI was localized as a panel. Seven ordinary UI contours passed the overly broad cell test. | Geometry rejected; raw/qualified/Fusion `false/false/false`. |
| `125441:394` | bbox `[109,290,379,418]`; structure `1.0/0.875/1.0` | Same low, short false panel during HOOK; bright text and seven non-inventory contours saturated all features. | Geometry rejected; raw/qualified/Fusion `false/false/false`. |
| `131254:1` | bbox `[127,248,379,418]`; structure `1.0/0.75/1.0` | Startup scene localized a bottom UI block, not the inventory panel. The old single-frame qualifier allowed it to influence startup synchronization. | Geometry rejected; raw/qualified/Fusion `false/false/false`. |
| `192315:484-499` | localizer returned no panel; raw/qualified `0/16` | The night background and real panel merged into one full search-region dark contour (`[0,0,379,418]`), which correctly failed the size rule but prevented feature scoring. The actual fixed panel is visible and has 26-27 grid contours. Frame 484 is still fading in. | Fixed-geometry dark-panel fallback finds frames 485-499: raw `15/16`, qualified/Fusion `14/16`. |
| `130308:440-458` | stable bbox `[99,95,379,345]`, dark ratio `0.92`, 20 grid contours | Day scene separates the dark panel cleanly from the background, so contour localization and all three panel regions agree. | Preserved: raw `19/19`, qualified/Fusion `18/19`; only the first frame waits for confirmation. |

The false positives were primarily a **wrong panel location plus over-broad structure scoring**, with single-frame qualification as a secondary safety weakness. They were not caused by the adapter. The `192315` false negative was a **dark-background contour merge**, not a smaller/moved panel or insufficient visual content.

## GET-only changes

- Require the localized bbox to match the reviewed fixed UI geometry and reject low, short, or bottom-clipped candidates.
- Require `item_grid` plus either title or collect-button evidence; title/button alone cannot declare GET.
- Add a fixed-geometry fallback only when the crop is strongly dark and contains at least eight grid-cell contours.
- Require two distinct consecutive raw GET frames before qualification/Fusion.
- Clear temporal GET evidence immediately when the panel disappears or detector activation is OFF.
- Preserve raw candidate, qualified observation, and Fusion usage as separate values.

No detector confidence threshold or feature weight was lowered or changed. The only GET config addition is `panel_confirmation_frames: 2`.

## Replay before/after

GET columns are `raw / qualified / Fusion / support`; false is qualified outside GET support.

| session | before | after | before result/final/sync | after result/final/sync |
| --- | --- | --- | --- | --- |
| `192315` | `0/0/0/16`, false 0 | `15/14/14/16`, false 0 | WARNING / WAITING / 0 | WARNING / WAITING / 0 |
| `061220` | `0/0/0/0`, false 0 | `0/0/0/0`, false 0 | WARNING / WAITING / 0 | WARNING / WAITING / 0 |
| `123210` | `0/0/0/0`, false 0 | `0/0/0/0`, false 0 | FAIL / PRESS / 0 | FAIL / PRESS / 0 |
| `124419` | `1/0/0/23`, false 33 | `22/21/21/23`, false 0 | FAIL / SYNC_REQUIRED / 232 | WARNING / WAITING / 0 |
| `125441` | `0/0/0/18`, false 11 | `17/16/16/18`, false 0 | FAIL / SYNC_REQUIRED / 178 | WARNING / WAITING / 0 |
| `130308` | `19/19/19/19`, false 0 | `19/18/18/19`, false 0 | WARNING / WAITING / 0 | WARNING / WAITING / 0 |
| `131254` | `22/0/0/23`, false 12 | `22/21/21/23`, false 0 | FAIL / SYNC_REQUIRED / 577 | WARNING / IDLE / 0 |

After classification: **0 PASS / 6 PASS_WITH_WARNINGS / 1 FAIL**. Existing HOOK recall warnings remain deliberately unresolved. No new non-GET qualified false positive was introduced.

Focused image/crop diagnostics are generated under the ignored directory `reports/fishing_v2/get_cross_session_diagnostics/` and are not committed.
