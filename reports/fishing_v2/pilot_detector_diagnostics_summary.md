# Pilot Detector Diagnostics Summary

- Session: `session_20260710_130308`
- Detector thresholds modified: **false**
- Hook raw detected by global state: `{'READY': 30, 'HOOK': 63, 'WAITING': 115, 'IDLE': 11}`
- Hook qualified ACTIVE by global state: `{'HOOK': 50}`
- Non-HOOK qualified false positives: **0**
- Hook qualification: raw detected + `bar_fill` + positive `fill_ratio` + strong confidence; divider is optional.
- PRESS panel present: **24/24**
- PRESS sequence ready (single-frame tool): **0/24**
- PRESS clean arrow candidate frames: **17/24**
- PRESS rejection counts: `{'panel_confirmation_required_for_arrow_freeze': 17, 'temporal_sequence_consensus_required': 7, 'sequence_confidence': 1}`
- Frame 428: panel_confidence=0.9702, sequence_candidate=['W', 'W', 'D', 'A', 'A', 'D', 'A', 'D'], sequence_confidence=0.6572, boxes=8, reasons=`['sequence_confidence:0.6572<0.68', 'temporal_sequence_consensus_required']`.
- Runtime sequence readiness is temporal; see `press_detector_diagnostics_summary.md` and the manual review bundle.
- GET original baseline: **0/19**; the fixed whole-ROI template comparison was misaligned with the live panel scale and position.
- GET detected in visual panel range 440-458: **19/19**
- GET rejection counts: `{'accepted': 19}`
- Visual boundary supplied by manual review: panel appears at frame 440 and disappears at frame 459.

Large diagnostic sheets/details are under `reports/fishing_v2/pilot_detector_diagnostics/` and are Git-ignored.
