# PRESS Exit Diagnostics: session_20260710_123210

## Root cause

- Last raw and qualified PRESS panel-present frame: **583**.
- Frame **584** is the first raw/qualified panel-absent frame. It is intentionally held as a single-frame miss.
- Frame **585** is the first stable panel-absent frame and the first post-PRESS visual acknowledgment (`PRESS -> RESULT_PENDING`).
- Frame **587** has four consecutive absent frames and acknowledges that the recorded session has no result panel (`RESULT_PENDING -> IDLE`).
- Frames 584-600 contain no scripted `IDLE_CAST` and no qualified GetPanel. The Prompt observation is `UNKNOWN` (`IGNORE` annotation), so neither Prompt nor GET could previously advance Runtime.
- The frozen `DW` sequence and its selected clean frame are cleared at frame 584. `PRESS_SEQUENCE` is proposed once at frame 576 and `action_applied` remains false. Neither value latches PRESS.
- The stale value was the broad diagnostic `panel_candidate=True`, not qualified panel-present evidence. The old FSM exit guard required both `detected=False` and `panel_candidate=False`, so frames 584-600 were rejected with `press_panel_active_awaiting_sequence` even though raw/qualified panel-present were already false.
- The recording contains 17 absent frames after PRESS, so this was not an end-of-file consensus shortage.

## Minimal fix

- PRESS aggregation now exposes `panel_absent_frames` and `panel_disappeared` independently of frozen sequence state.
- Recorded observation requires two consecutive panel-absent frames before leaving PRESS; one miss cannot exit.
- A diagnostic panel candidate no longer blocks a confirmed recorded disappearance.
- After four consecutive absent frames with no GET/result panel, recorded observation advances `RESULT_PENDING -> IDLE`. A later qualified GET still has existing priority, including from IDLE.
- Production mode keeps its existing action/apply and immediate non-candidate clear semantics.
- No detector confidence threshold, PRESS arrow/glyph rule, sequence ground truth, Prompt annotation, Hook detector, or GET detector was changed. New temporal counts are config values: disappearance `2`, recorded no-result exit `4`.

## Before/after

| item | before | after |
| --- | --- | --- |
| panel visible | frames 574-583 | frames 574-583 |
| first absent | frame 584, permanently held in PRESS | frame 584, held as one-frame miss |
| stable absent | ignored because `panel_candidate=True` | frame 585, `PRESS -> RESULT_PENDING` |
| extended no-result acknowledgment | none | frame 587, `RESULT_PENDING -> IDLE` |
| PRESS_SEQUENCE proposals / actions applied | `1 / 0` | `1 / 0` |
| final frame/state | `600 / PRESS` | `600 / IDLE` |
| SYNC_REQUIRED | `0` | `0` |

## Cross-session result

All seven annotations and replays pass structural/completion checks. Classification is **0 PASS / 7 PASS_WITH_WARNINGS / 0 FAIL**; all SYNC_REQUIRED counts and actions applied are zero. GET remains `95 raw / 90 qualified / 90 Fusion / 99 support`, with no qualified false positive. Existing HOOK recall and clean-frame PRESS warnings remain out of scope.

The compact per-frame evidence for frames 565-600 is stored in the adjacent JSON report. Global state is included there for offline evaluation only and was never used to drive Runtime.
