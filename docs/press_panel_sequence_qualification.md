# PRESS Panel and Sequence Qualification

PRESS evidence has three deliberately separate levels. `PRESS_PANEL_CANDIDATE` is diagnostic geometry, `PRESS_PANEL_PRESENT` requires repeated grid structure and can confirm runtime PRESS, and `PRESS_SEQUENCE_READY` additionally requires a variable-length, geometrically aligned temporal glyph consensus. A panel may therefore remain active while sequence decoding is unavailable.

The temporal settings live under `press_detector` in `config/fishing_v2.yaml`. Existing per-frame glyph confidence remains 0.68; aggregated confidence also considers cross-frame agreement and top-candidate ambiguity. Repeated near-ties are not promoted merely because they are stable.

Cross-session results are written to `reports/fishing_v2/press_detector_diagnostics_summary.md`. The local review bundle is `reports/fishing_v2/press_sequence_review/index.html` plus `review_items.csv`; large review images are ignored by Git. `manually_confirmed_sequence` stays blank until a human supplies it, so sequence accuracy is reported as N/A rather than inferred from detector output.
