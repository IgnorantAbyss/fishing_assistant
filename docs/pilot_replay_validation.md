# Pilot Replay Validation

Pilot `session_20260710_130308` uses the user-authored Prompt annotation committed unchanged. Replay action mode `recorded_observation` reports proposed intents but never applies them. State advances only from Prompt phase acknowledgements or qualified specialized evidence already visible in the recording.

Hook raw diagnostics preserve rectangle candidates, while Fusion receives only active bars with `bar_fill`, positive fill ratio, and strong confidence. OFF results remain diagnostic-only. PRESS/GET use the same raw-versus-qualified contract.

PRESS panel presence is now independent from glyph decoding. The Pilot reports structural panel presence on 24/24 PRESS frames, enters runtime PRESS after two-frame confirmation, and reports no sequence-ready frame or PRESS intent because the ambiguous glyph candidates do not satisfy temporal confidence and margin requirements. GET used a fixed whole-ROI comparison even when the panel scale differed; the minimal fix localizes the panel before structural title/grid/button comparison. No existing per-frame glyph threshold or detector feature weight was lowered.

Large images and per-frame CSV/JSON remain ignored. Tracked summaries are `reports/fishing_v2/pilot_detector_diagnostics_summary.*` and `reports/fishing_v2/pilot_replay_summary.*`.
