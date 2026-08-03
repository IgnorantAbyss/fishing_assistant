# PRESS Panel and Sequence Qualification

PRESS evidence has three deliberately separate levels. `PRESS_PANEL_CANDIDATE` is diagnostic geometry, `PRESS_PANEL_PRESENT` requires repeated grid structure and can confirm runtime PRESS, and `PRESS_SEQUENCE_READY` requires an eligible clean frame before the first input effect. A panel may therefore remain active while sequence decoding is unavailable.

The decoder infers occupancy for every panel slot and never equates total grid capacity with sequence length. Occupied arrows map as `LEFT=A`, `DOWN=S`, `RIGHT=D`, and `UP=W`. Arrow shape is primary evidence; letter templates are auxiliary diagnostics and cannot override an arrow conflict. Once the earliest clean sequence is frozen, later input/glow frames cannot change it.

Cross-session results are written to `reports/fishing_v2/press_detector_diagnostics_summary.md`. The local review bundle is `reports/fishing_v2/press_sequence_review/index.html` plus `review_items.csv`; large review images are ignored by Git. Confirmed answers come only from `data/annotations/press_sequence_ground_truth.yaml`. Episodes without a pre-input clean frame are marked non-evaluable and excluded from accuracy.

## Background-subtracted V3 modes

`PressBackgroundSubtractionDetectorV3` first locates the repeated ten-cell
strip, models its gray-black LAB background, and classifies binary foreground
shape without assuming a glyph hue. It is opt-in:

- `--press-detector-mode legacy` remains the Production default.
- `--press-detector-mode background-subtraction-shadow` runs V3 on the
  existing captured frame in a latest-only worker. Legacy remains the only
  source of Runtime/FSM/action evidence.
- `--press-detector-mode background-subtraction-live` explicitly selects the
  V3 adapter while retaining the existing temporal completeness, Safety, and
  mockable ActionSink path.

Optional bounded ROI evidence is enabled with `--press-v3-debug-evidence` and
limited by `--press-v3-debug-max-episodes` and
`--press-v3-debug-max-frames-per-episode`. It writes PRESS-panel crops, masks,
and JSON only; it neither captures another frame nor creates video.

Offline inspection uses:

```powershell
.\.venv\Scripts\python.exe tools\analyze_press_background_subtraction.py `
  --input tests\fixtures\press_structural_occupancy `
  --output reports\press_v3_analysis
```
