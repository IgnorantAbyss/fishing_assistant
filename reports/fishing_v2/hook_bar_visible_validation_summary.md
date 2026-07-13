# Hook Bar Visible Validation Summary

- Ground truth: **human-confirmed YAML only**.
- Ground truth validation: **PASS**.
- Global-HOOK qualified recall (comparison only): **164/260 (63.08%)**.
- Visible-range qualified recall: **164/214 (76.64%)**.
- Clear-range valid-fill recall: **164/214 (76.64%)**.
- Qualified false positives outside visible ranges: **26**.
- Rectangle-only candidates outside visible ranges: **544**.
- `hook_detector_ready = false`

| session | episode | visible support | qualified | visible recall | first latency | valid fill | safe-zone | outside FP | result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| session_20260709_192315 | 1 | 13 | 13 | 100.00% | 0 | 13/13 | 8 | 0 | PASS |
| session_20260710_061220 | 1 | 21 | 20 | 95.24% | 0 | 20/21 | 12 | 0 | PASS |
| session_20260710_123210 | 1 | 12 | 6 | 50.00% | 0 | 6/12 | 4 | 4 | PASS |
| session_20260710_124419 | 1 | 20 | 9 | 45.00% | 0 | 9/20 | 0 | 12 | FAIL |
| session_20260710_125441 | 1 | 30 | 17 | 56.67% | 0 | 17/30 | 0 | 10 | FAIL |
| session_20260710_130308 | 1 | 36 | 30 | 83.33% | 0 | 30/36 | 2 | 0 | PASS |
| session_20260710_130308 | 2 | 30 | 20 | 66.67% | 0 | 20/30 | 1 | 0 | PASS |
| session_20260710_131254 | 1 | 30 | 28 | 93.33% | 0 | 28/30 | 1 | 0 | PASS |
| session_20260710_131254 | 2 | 22 | 21 | 95.45% | 0 | 21/22 | 11 | 0 | PASS |

## session_20260710_131254

- Episode 1 visible `[195, 224]` / clear `[195, 224]`: raw `28`, qualified `28`, missing bar_fill `2`, zero fill `0`, valid fill `True`, safe-zone `True`, ROI `['hook_bar_precise']`, ROI misalignment proven `False`, reasons `{}`.
- Episode 2 visible `[532, 553]` / clear `[532, 553]`: raw `21`, qualified `21`, missing bar_fill `1`, zero fill `0`, valid fill `True`, safe-zone `True`, ROI `['hook_bar_precise']`, ROI misalignment proven `False`, reasons `{}`.
- Root cause: The detector returns active positive-fill evidence in both human-visible ranges when the normal BURST qualification path is enabled. The prior 0/62 result was caused by Runtime activation/Fusion availability, not missing bar_fill, zero fill_ratio, or proven ROI misalignment.

## Gate

- Failed episodes: `['session_20260710_124419#1', 'session_20260710_125441#1']`.
- Validation forces the normal BURST qualification path for offline detector eligibility; it does not use global state to advance Runtime.
- No detector threshold was changed.
- `session_20260710_124419#1` failed: no valid fill_ratio in configured range 0.65-0.85; observed 1.0000-1.0000; missed visible frames `[316, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333]`.
- `session_20260710_125441#1` failed: no valid fill_ratio in configured range 0.65-0.85; observed 1.0000-1.0000; missed visible frames `[369, 377, 378, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395]`.
