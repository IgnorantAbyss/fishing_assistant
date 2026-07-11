# Pilot Detector Diagnostics Summary

- Session: `session_20260710_130308`
- Detector thresholds modified: **false**
- Hook raw detected by global state: `{'READY': 30, 'HOOK': 63, 'WAITING': 115, 'IDLE': 11}`
- Hook qualified ACTIVE by global state: `{'HOOK': 50}`
- Non-HOOK qualified false positives: **0**
- Hook qualification: raw detected + `bar_fill` + positive `fill_ratio` + strong confidence; divider is optional.
- PRESS detected: **0/24**
- PRESS rejection counts: `{'incomplete_sequence': 18, 'glyph_confidence': 24, 'dark_panel_ratio': 14}`
- Frame 428: confidence=0.6, sequence=['D', 'W', 'S', 'W', 'A', 'W', 'W', 'W'], boxes=8, reasons=`['glyph_confidence:0.6000<0.68']`.
- PRESS detector has no temporal rejection rule; failures are frame-local sequence/key-box/confidence/panel checks.
- GET original baseline: **0/19**; the fixed whole-ROI template comparison was misaligned with the live panel scale and position.
- GET detected in visual panel range 440-458: **19/19**
- GET rejection counts: `{'accepted': 19}`
- Visual boundary supplied by manual review: panel appears at frame 440 and disappears at frame 459.

Large diagnostic sheets/details are under `reports/fishing_v2/pilot_detector_diagnostics/` and are Git-ignored.
