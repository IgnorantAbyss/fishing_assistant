# PRESS Panel and Sequence Qualification

PRESS evidence has three deliberately separate levels. `PRESS_PANEL_CANDIDATE` is diagnostic geometry, `PRESS_PANEL_PRESENT` requires repeated grid structure and can confirm runtime PRESS, and `PRESS_SEQUENCE_READY` requires an eligible clean frame before the first input effect. A panel may therefore remain active while sequence decoding is unavailable.

The decoder infers occupancy for every panel slot and never equates total grid capacity with sequence length. Occupied arrows map as `LEFT=A`, `DOWN=S`, `RIGHT=D`, and `UP=W`. Arrow shape is primary evidence; letter templates are auxiliary diagnostics and cannot override an arrow conflict. Once the earliest clean sequence is frozen, later input/glow frames cannot change it.

Cross-session results are written to `reports/fishing_v2/press_detector_diagnostics_summary.md`. The local review bundle is `reports/fishing_v2/press_sequence_review/index.html` plus `review_items.csv`; large review images are ignored by Git. Confirmed answers come only from `data/annotations/press_sequence_ground_truth.yaml`. Episodes without a pre-input clean frame are marked non-evaluable and excluded from accuracy.
