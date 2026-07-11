# Prompt ROI Review Report

- Fixed environment: **2560x1440**, fixed UI scale, zh-TW, borderless, fixed Prompt position.
- ROI source of truth: **pixel coordinates**; normalized coordinates are derived display metadata only.
- ROI status: **unapproved**.
- Manual review required: **true**.
- Automatic selection / bright-mask approval / classifier-accuracy selection: **false / false / false**.
- Sessions: session_20260709_192315, session_20260710_061220, session_20260710_123210, session_20260710_124419, session_20260710_125441, session_20260710_130308, session_20260710_131254
- Trial session excluded: `session_20260709_192231`
- Distinct Prompt appearances: **not yet manually confirmed**.

## Candidates

| Candidate | Pixel | Derived normalized | Size | Area | Reduction vs legacy |
| --- | --- | --- | --- | ---: | ---: |
| legacy_reference | `[768, 29, 1792, 144]` | `[0.3, 0.020139, 0.7, 0.1]` | 1024x115 | 117760 | 0.00% |
| tight_vertical | `[768, 32, 1792, 100]` | `[0.3, 0.022222, 0.7, 0.069444]` | 1024x68 | 69632 | 40.87% |
| icon_and_text_medium | `[880, 32, 1680, 104]` | `[0.34375, 0.022222, 0.65625, 0.072222]` | 800x72 | 57600 | 51.09% |
| icon_and_text_tight | `[940, 34, 1620, 100]` | `[0.367188, 0.023611, 0.632812, 0.069444]` | 680x66 | 44880 | 61.89% |
| text_only_medium | `[1000, 32, 1660, 104]` | `[0.390625, 0.022222, 0.648438, 0.072222]` | 660x72 | 47520 | 59.65% |
| text_only_tight | `[1060, 34, 1580, 98]` | `[0.414062, 0.023611, 0.617188, 0.068056]` | 520x64 | 33280 | 71.74% |

## Sampling

- Stable interior total: 304
- Stable counts by global state: `{'IDLE': 56, 'WAITING': 56, 'READY': 54, 'HOOK': 54, 'PRESS': 47, 'GET': 37}`
- Transition windows total: 50
- Transition samples total: 1000
- Transition window counts: `{'IDLE_to_WAITING': 11, 'WAITING_to_READY': 8, 'READY_to_HOOK': 9, 'HOOK_to_PRESS': 8, 'PRESS_to_GET': 2, 'GET_to_IDLE': 5, 'PRESS_to_IGNORE': 6, 'HOOK_to_IDLE': 1}`
- Large visual sheets: `reports/fishing_v2/roi_review/` (ignored by Git).

## Manual questions

- Does each candidate contain the complete visible prompt? — **manual_review_required**
- Does any candidate include unrelated bottom UI? — **manual_review_required**
- Does any candidate include excessive scene background? — **manual_review_required**
- Which candidate preserves the longest prompt? — **manual_review_required**
- Are icon/key cues required Prompt content? — **manual_review_required**
- Does READY remain visible after the global READY-to-HOOK boundary? — **manual_review_required**
- Is a distinct PRESS instruction visible? — **manual_review_required**
- Is GET represented by a Prompt or by NO_PROMPT? — **manual_review_required**
- Which transition frames should be IGNORE? — **manual_review_required**
- Are there visually distinct prompts within one suggested observation kind? — **manual_review_required**

## Provisional review order

`legacy_reference`, `tight_vertical`, `icon_and_text_medium`, `icon_and_text_tight`, `text_only_medium`, `text_only_tight`

This is a review order, not a final recommendation or approval.
