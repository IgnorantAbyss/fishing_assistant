# ADR 0004: Prompt-Specific Labels

Status: accepted

`prompt_ground_truth.yaml` is human-authored from visible Prompt ROI content. It is never generated from global state. Final runtime hints are IDLE_CAST, WAITING_IN_PROGRESS, READY_BITE, HOOK_INSTRUCTION, PRESS_INSTRUCTION, and UNKNOWN; annotation additionally permits IGNORE. IGNORE is never emitted at runtime. Deprecated v1 names are read-only compatibility data and cannot become new classifier targets.
