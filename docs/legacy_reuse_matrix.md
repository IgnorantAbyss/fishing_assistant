# Legacy Reuse Matrix

This classification is based on reading the implementations, not filenames alone.

| Module/tool | Classification | Reason and v2 boundary |
| --- | --- | --- |
| `src/screen_capture.py` | reuse_as_is | Pure monitor enumeration/capture; it sends no input. Runtime access still belongs behind an infrastructure port. |
| `src/replay_session.py` | reuse_via_adapter | Manifest/frame reader is useful. The v2 replay source adapter prevents core code from depending on its storage shape. Latest-session ordering uses manifest `created_at`. |
| `src/replay_ground_truth.py` | reuse_as_is | Strict complete global-segment validation remains useful for evaluation only. It does not define prompt observations. |
| `src/detectors/hook_detector.py` | reuse_via_adapter | Offline detector returns detected/confidence/debug/fill evidence. Runtime applies its configurable safe zone and does not chase Perfect; detector algorithm remains untouched. |
| `src/detectors/press_detector.py` | reuse_via_adapter | Offline detector returns detected/confidence/sequence/debug. PressPanel confirms ACTIVE; action history prevents residual-panel sequence duplication. |
| `src/detectors/get_detector.py` | reuse_via_adapter | Strict title/grid/button evidence outranks IDLE Prompt, guards CAST, and gates bounded COLLECT retries. Algorithm remains untouched. |
| Specialized ROIs in `config/roi.yaml` | reference_only | Adapters may use the legacy detector's config; v2 core does not own or reinterpret these ROIs. |
| Specialized reference fixtures/tests | reuse_as_is | Contract tests compare adapter observations to unchanged detector outputs. |
| `tools/agent_check.py` specialized tests | reuse_as_is | Continues guarding existing detectors without becoming a runtime dependency. |
| Failure-analysis tool/report | reference_only | Establishes label/ROI problems and informs design; it is not a runtime observer. |
| `src/state_detector.py` | deprecated_for_v2 | Mixes prompt templates, static references, specialized gates, thresholds, and global state in one per-frame decision. |
| `src/state_smoother.py` | deprecated_for_v2 | Repairs completed state sequences without actions, timeouts, sync state, or safety decisions; not a runtime FSM. |
| Live prompt template bank | deprecated_for_v2 | Template prompt output was fused directly into global state and inherits the v1 semantic conflict. |
| `src/ml/prompt_*` / PromptClassifier v1 | reference_only | Model is frozen. It is not a v2 observer, is not used by activation/runtime, and its classes/threshold are not reused. |
| `src/dataset/roi_exporter.py` prompt mapping | deprecated_for_v2 | `HOOK/PRESS/GET -> NONE` derives prompt labels from global state, which failure analysis disproved. |
| v1 prompt threshold | deprecated_for_v2 | Selected for the conflicted v1 task and cannot define v2 observation confidence. |
| UNKNOWN-to-IDLE behavior | deprecated_for_v2 | v2 explicitly preserves UNKNOWN and requires sync/evidence. |
| Per-frame state overwrite | deprecated_for_v2 | Violates legal transitions, stability, action history, and conflict handling. |
| Real foreground-window detector | unknown_needs_review | Only a port is defined in this phase; platform-specific implementation and semantics need user confirmation. |
| Real action sink | unknown_needs_review | Explicitly absent. A future implementation requires separate authorization and safety review. |
| Detector activation scheduler | new_v2_contract | OFF/ARMED/BURST/ACTIVE is driven by runtime state, Prompt hints, action history, and specialized confirmation; initial FPS values require detect-only validation. |
