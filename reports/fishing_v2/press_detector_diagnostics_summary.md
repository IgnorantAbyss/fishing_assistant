# PRESS Detector Diagnostics Summary

- Sessions / PRESS episodes: **7 / 8**
- Panel presence recall: **147/151 (97.35%)**
- Raw panel false positives by global state: `{'IDLE': 0, 'WAITING': 1, 'READY': 0, 'HOOK': 0, 'GET': 0}`
- Temporally confirmed structural false positives by global state: `{'IDLE': 0, 'WAITING': 0, 'READY': 0, 'HOOK': 0, 'GET': 0}`
- Runtime Fusion false positives: **0**; diagnostic OFF evidence remains excluded from Fusion.
- Sequence-ready episodes: **0/8**
- Sequence accuracy: **N/A** — no manually confirmed sequence ground truth exists.
- Existing per-frame glyph threshold was not lowered; panel presence is structural and sequence readiness uses temporal confidence plus ambiguity margin.

## Sessions

- `session_20260709_192315`: PRESS frames 11/12; episodes 1; sequence ready 0
- `session_20260710_061220`: PRESS frames 8/11; episodes 1; sequence ready 0
- `session_20260710_123210`: PRESS frames 10/10; episodes 1; sequence ready 0
- `session_20260710_124419`: PRESS frames 22/22; episodes 1; sequence ready 0
- `session_20260710_125441`: PRESS frames 24/24; episodes 1; sequence ready 0
- `session_20260710_130308`: PRESS frames 24/24; episodes 1; sequence ready 0
- `session_20260710_131254`: PRESS frames 48/48; episodes 2; sequence ready 0

## Episodes

- `session_20260709_192315` 472-483: panel confirmed frame 473 (latency 1 frames); boxes `[8, 10, 10, 10, 10, 10, 8, 8, 8, 8, 10, 0]`; candidate `WWWSSSSADD`; status `sequence_not_recoverable_from_replay`.
- `session_20260710_061220` 507-517: panel confirmed frame 508 (latency 1 frames); boxes `[4, 4, 4, 4, 4, 4, 4, 5, 0, 0, 0]`; candidate `AWSA`; status `sequence_not_recoverable_from_replay`.
- `session_20260710_123210` 574-583: panel confirmed frame 575 (latency 1 frames); boxes `[2, 2, 2, 10, 2, 2, 2, 2, 3, 3]`; candidate `AW`; status `sequence_not_recoverable_from_replay`.
- `session_20260710_124419` 334-355: panel confirmed frame 335 (latency 1 frames); boxes `[10, 10, 10, 10, 10, 10, 10, 10, 10, 5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]`; candidate `DDWDDAAAAA`; status `sequence_not_recoverable_from_replay`.
- `session_20260710_125441` 396-419: panel confirmed frame 397 (latency 1 frames); boxes `[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 7, 10, 10, 10, 10, 10, 10, 10, 10]`; candidate `DDDDDDDSDA`; status `sequence_not_recoverable_from_replay`.
- `session_20260710_130308` 415-438: panel confirmed frame 416 (latency 1 frames); boxes `[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 6, 6, 6, 9, 8, 8, 8]`; candidate `AAAAAAAA`; status `sequence_not_recoverable_from_replay`.
- `session_20260710_131254` 225-249: panel confirmed frame 226 (latency 1 frames); boxes `[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 5, 5, 5, 7, 7, 7, 7, 7, 7]`; candidate `AAAAAAA`; status `sequence_not_recoverable_from_replay`.
- `session_20260710_131254` 554-576: panel confirmed frame 555 (latency 1 frames); boxes `[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 4, 4, 4, 5, 6, 6, 6, 6, 6, 6, 6]`; candidate `AAAAAA`; status `sequence_not_recoverable_from_replay`.

- Manual review: `D:\project\fishing_assistant\reports\fishing_v2\press_sequence_review\index.html`
- Review CSV: `D:\project\fishing_assistant\reports\fishing_v2\press_sequence_review\review_items.csv`
