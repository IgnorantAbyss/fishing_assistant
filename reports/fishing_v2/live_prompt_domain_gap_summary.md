# Live Bite Prompt Semantic Correction

## Correction

The Live prompt `好像釣到什麼，請按下 'Space' 開始吧。` is **READY_BITE**, not IDLE_CAST. The previous report and candidate metadata were semantically wrong. The incorrect IDLE candidate lineage has been removed; no IDLE fallback remains in the deployment bundle.

The same text is visibly present in formal READY_BITE frames, including `session_20260710_061220/frame 469`. The original seven-session annotations are unchanged. The failure category is READY prototype coverage / Live-domain generalization after reducing each session/class to one medoid.

## Candidate and bundle lineage

- Reference: `assets/reference/prompt/live_ready_bite_20260713_000035.png`
- Status: `live_calibration_candidate`
- Review: `human_confirmed_live_candidate`
- Label: `READY_BITE`
- Ground Truth: **false**
- Used for threshold calibration: **false**
- Formal medoids retained: **35**
- Live deployment candidates: **1 READY_BITE**
- Final counts: IDLE 7, WAITING 7, READY 8, HOOK 7, PRESS 7, IGNORE 0
- Bundle: `prototype_v1_final_3`
- Bundle SHA-256: `6ff79c60cc7bd3adb8f63042edde21c641cc1c3ee2776c472544cf8b1387333e`

LOSO remains isolated to the seven formal sessions. The Live candidate is appended only to the final deployment bundle. Class thresholds, ambiguity margin `0.2791204953`, and IDLE stability `4` are unchanged.

## Calibration and independent holdout

Frame 35 is the calibration crop. Frame 39 is a separately saved Live frame and is used as holdout; the same ROI is not duplicated and presented as multiple Live samples.

| source | prediction | top similarity | second class | second similarity | margin | matched prototype |
|---|---|---:|---|---:|---:|---|
| Formal medoids only, frame 35 | UNKNOWN | 0.527797 | PRESS_INSTRUCTION | 0.519986 | 0.007811 | HOOK_INSTRUCTION formal medoid |
| Corrected calibration frame 35 | READY_BITE | 1.000000 | HOOK_INSTRUCTION | 0.527797 | 0.472203 | live_ready_bite_20260713_000035 |
| Corrected holdout frame 39 | READY_BITE | 0.999845 | HOOK_INSTRUCTION | 0.528097 | 0.471747 | live_ready_bite_20260713_000035 |

Neither corrected frame is IDLE_CAST, UNKNOWN, HOOK_INSTRUCTION, or PRESS_INSTRUCTION.

## Runtime startup regression

A deterministic detect-only regression uses the independent frame-39 holdout ROI and the real PromptObserver, synchronizer, Fusion, FSM, Safety, and would-fire deduplicator. It does not emit input.

- PromptObservation: **READY_BITE** from the first observation
- Startup consensus: reached READY; no `startup_consensus_not_reached`
- Final Runtime state: **READY**
- Raw START_HOOK proposals: **21**
- Unique `WOULD_START_HOOK`: **1**
- `WOULD_CAST`: **0**
- IDLE stability guard entered: **false**
- `actions_applied`: **0**
- Action sink: **none**

## Regression status

- LOSO macro F1: **0.992316**
- IGNORE rejection: **1.000000**
- Live candidate included in LOSO: **false**
- Corrected final-bundle Replay: **7/7 PASS**
- False intents / missed events / SYNC_REQUIRED / actions applied: **0 / 0 / 0 / 0**

Diagnostic calibration/holdout crops and nearest formal prototypes are in `reports/fishing_v2/live_prompt_diagnostics/`.
