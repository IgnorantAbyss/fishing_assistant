# PRESS Detector Diagnostics Summary

- Sessions / PRESS episodes: **7 / 9**
- Panel presence recall: **150/151 (99.34%)**
- Raw panel false positives by global state: `{'IDLE': 0, 'WAITING': 1, 'READY': 0, 'HOOK': 0, 'GET': 0}`
- Temporally confirmed structural false positives by global state: `{'IDLE': 0, 'WAITING': 0, 'READY': 0, 'HOOK': 0, 'GET': 0}`
- Runtime Fusion false positives: **0**; diagnostic OFF evidence remains excluded from Fusion.
- Sequence-ready episodes: **9/9**
- Sequence-ready coverage: **9/9** (100.00%).
- All-episode exact sequence accuracy: **9/9** (100.00%).
- Abstained / wrong-sequence episodes: **0 / 0**.
- Manual sequence ground truth: `data/annotations/press_sequence_ground_truth.yaml` (8 confirmed episodes) plus one local SAS ROI fixture.
- Existing panel/glyph thresholds were not lowered; arrow direction is derived from local-contrast geometry and letter templates remain diagnostic only.
- Per-key confusion matrix: `{'W': {'W': 13, 'A': 0, 'S': 0, 'D': 0, 'ABSTAIN': 0, 'EXTRA': 0}, 'A': {'W': 0, 'A': 12, 'S': 0, 'D': 0, 'ABSTAIN': 0, 'EXTRA': 0}, 'S': {'W': 0, 'A': 0, 'S': 15, 'D': 0, 'ABSTAIN': 0, 'EXTRA': 0}, 'D': {'W': 0, 'A': 0, 'S': 0, 'D': 10, 'ABSTAIN': 0, 'EXTRA': 0}}`

## Sessions

- `session_20260709_192315`: PRESS frames 11/12; episodes 1; sequence ready 1
- `session_20260710_061220`: PRESS frames 8/8; episodes 1; sequence ready 1
- `session_20260710_123210`: PRESS frames 10/10; episodes 1; sequence ready 1
- `session_20260710_124419`: PRESS frames 22/22; episodes 1; sequence ready 1
- `session_20260710_125441`: PRESS frames 24/24; episodes 1; sequence ready 1
- `session_20260710_130308`: PRESS frames 24/24; episodes 1; sequence ready 1
- `session_20260710_131254`: PRESS frames 48/48; episodes 2; sequence ready 2

## Episodes

- `session_20260709_192315` 472-483: candidate/present/clean/input `472/472/472/473`; selected `472` (earliest_clean_pre_input_frame); slots `8/10`; predicted `ASDWWDWS`; expected `ASDWWDWS`; exact `True`; status `sequence_ready`.
- `session_20260710_061220` 507-514: candidate/present/clean/input `507/507/507/512`; selected `507` (earliest_clean_pre_input_frame); slots `4/10`; predicted `AWSA`; expected `AWSA`; exact `True`; status `sequence_ready`.
- `session_20260710_123210` 574-583: candidate/present/clean/input `574/574/574/577`; selected `574` (earliest_clean_pre_input_frame); slots `2/10`; predicted `DW`; expected `DW`; exact `True`; status `sequence_ready`.
- `session_20260710_124419` 334-355: candidate/present/clean/input `334/334/334/342`; selected `334` (earliest_clean_pre_input_frame); slots `5/10`; predicted `DSWSS`; expected `DSWSS`; exact `True`; status `sequence_ready`.
- `session_20260710_125441` 396-419: candidate/present/clean/input `396/396/396/405`; selected `396` (earliest_clean_pre_input_frame); slots `7/10`; predicted `WASASDD`; expected `WASASDD`; exact `True`; status `sequence_ready`.
- `session_20260710_130308` 415-438: candidate/present/clean/input `415/415/415/423`; selected `415` (earliest_clean_pre_input_frame); slots `8/10`; predicted `WWDDWWSS`; expected `WWDDWWSS`; exact `True`; status `sequence_ready`.
- `session_20260710_131254` 225-249: candidate/present/clean/input `225/225/225/231`; selected `225` (earliest_clean_pre_input_frame); slots `7/10`; predicted `WDASADS`; expected `WDASADS`; exact `True`; status `sequence_ready`.
- `session_20260710_131254` 554-576: candidate/present/clean/input `554/554/554/558`; selected `554` (earliest_clean_pre_input_frame); slots `6/10`; predicted `WAAASA`; expected `WAAASA`; exact `True`; status `sequence_ready`.
- `session_20260731_194416_local_fixture` 159-161: predicted `SAS`; expected `SAS`; exact `True`; status `sequence_ready`; source `local_small_live_roi_regression_fixture`.

- Manual review: `reports/fishing_v2/press_sequence_review/index.html`
- Review CSV: `reports/fishing_v2/press_sequence_review/review_items.csv`
