# PRESS Detector Diagnostics Summary

- Sessions / PRESS episodes: **7 / 8**
- Panel presence recall: **147/148 (99.32%)**
- Raw panel false positives by global state: `{'IDLE': 0, 'WAITING': 1, 'READY': 0, 'HOOK': 0, 'GET': 0}`
- Temporally confirmed structural false positives by global state: `{'IDLE': 0, 'WAITING': 0, 'READY': 0, 'HOOK': 0, 'GET': 0}`
- Runtime Fusion false positives: **0**; diagnostic OFF evidence remains excluded from Fusion.
- Sequence-ready episodes: **5/8**
- Sequence exact match: **5/5** (1.0).
- Manual sequence ground truth: `D:\project\fishing_assistant\data\annotations\press_sequence_ground_truth.yaml` (8 confirmed episodes).
- Existing glyph threshold was not lowered; arrow direction is primary evidence and letter templates remain auxiliary diagnostics.

## Sessions

- `session_20260709_192315`: PRESS frames 11/12; episodes 1; sequence ready 1
- `session_20260710_061220`: PRESS frames 8/8; episodes 1; sequence ready 1
- `session_20260710_123210`: PRESS frames 10/10; episodes 1; sequence ready 1
- `session_20260710_124419`: PRESS frames 22/22; episodes 1; sequence ready 0
- `session_20260710_125441`: PRESS frames 24/24; episodes 1; sequence ready 0
- `session_20260710_130308`: PRESS frames 24/24; episodes 1; sequence ready 1
- `session_20260710_131254`: PRESS frames 48/48; episodes 2; sequence ready 1

## Episodes

- `session_20260709_192315` 472-483: candidate/present/clean/input `472/472/472/473`; selected `472` (earliest_clean_pre_input_frame); slots `8/10`; predicted `ASDWWDWS`; expected `ASDWWDWS`; exact `True`; status `sequence_ready`.
- `session_20260710_061220` 507-514: candidate/present/clean/input `507/507/507/512`; selected `507` (earliest_clean_pre_input_frame); slots `4/10`; predicted `AWSA`; expected `AWSA`; exact `True`; status `sequence_ready`.
- `session_20260710_123210` 574-583: candidate/present/clean/input `574/574/574/577`; selected `574` (earliest_clean_pre_input_frame); slots `2/10`; predicted `DW`; expected `DW`; exact `True`; status `sequence_ready`.
- `session_20260710_124419` 334-355: candidate/present/clean/input `334/334/None/343`; selected `344` (fallback_visual_review_no_clean_frame); slots `5/10`; predicted `N/A`; expected `DSWSS`; exact `None`; status `sequence_not_evaluable_from_existing_replay`.
- `session_20260710_125441` 396-419: candidate/present/clean/input `396/396/None/410`; selected `417` (fallback_visual_review_no_clean_frame); slots `7/10`; predicted `N/A`; expected `WASASDD`; exact `None`; status `sequence_not_evaluable_from_existing_replay`.
- `session_20260710_130308` 415-438: candidate/present/clean/input `415/415/415/427`; selected `415` (earliest_clean_pre_input_frame); slots `8/10`; predicted `WWDDWWSS`; expected `WWDDWWSS`; exact `True`; status `sequence_ready`.
- `session_20260710_131254` 225-249: candidate/present/clean/input `225/225/225/226`; selected `225` (earliest_clean_pre_input_frame); slots `7/10`; predicted `WDASADS`; expected `WDASADS`; exact `True`; status `sequence_ready`.
- `session_20260710_131254` 554-576: candidate/present/clean/input `554/554/None/554`; selected `564` (fallback_visual_review_no_clean_frame); slots `7/10`; predicted `N/A`; expected `WAAASA`; exact `None`; status `sequence_not_evaluable_from_existing_replay`.

- Manual review: `D:\project\fishing_assistant\reports\fishing_v2\press_sequence_review\index.html`
- Review CSV: `D:\project\fishing_assistant\reports\fishing_v2\press_sequence_review\review_items.csv`
