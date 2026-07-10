# ADR 0004: Prompt-Specific Labels

Status: accepted

`prompt_ground_truth.yaml` is human-authored from visible Prompt ROI content. It is never generated from global state. Runtime kinds include OTHER_PROMPT, NO_PROMPT, and UNKNOWN; annotation additionally permits IGNORE. Global HOOK may coexist with READY_PROMPT and global PRESS with OTHER_PROMPT.
