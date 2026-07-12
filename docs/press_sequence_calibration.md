# PRESS Sequence Calibration Plan

Do not tune arrow or occupancy rules from one episode. Collect at least 10–20 manually confirmed PRESS episodes spanning different backgrounds and effect timing. Keep panel appearance through at least 0.5 seconds after the clean phase so the first input-effect boundary is observable.

For every episode, preserve the original frame range and manually confirm the variable-length sequence. Review the earliest structurally complete frame before any glow, mixed progress colour, uncertain occupancy, or occupied-after-empty conflict. Record per-slot occupancy, arrow direction, confidence, ambiguity margin, and auxiliary letter candidate.

Calibration should split by episode/session, compare exact sequence match only on evaluable pre-input clean frames, and report non-evaluable episodes separately. Never tune on the final evaluation sessions, never infer labels from predictions, and never lower the existing glyph threshold to compensate for poor frame selection.
