# Prompt Annotation Policy Draft

Prompt annotation records only the text/instruction visibly present in the fixed top Prompt ROI. It must never be inferred from global state, specialized detector output, or classifier prediction.

## Allowed new annotations

- `IDLE_CAST`
- `WAITING_IN_PROGRESS`
- `READY_BITE`
- `HOOK_INSTRUCTION`
- `PRESS_INSTRUCTION`
- `IGNORE`

`IGNORE` is for ambiguous fades/transitions and never becomes a runtime observation or classifier target. Runtime absence/rejection becomes `UNKNOWN`; `NO_PROMPT` is not a v2 target. `OTHER_PROMPT` is also not a target: a newly discovered instruction requires explicit policy review before schema expansion.

Central notifications do not override the visible top Prompt. For example, if a result notification appears in the center while `IDLE_CAST` remains in the top ROI, annotate the top ROI as `IDLE_CAST`.

Optional `prompt_id` and `notes` metadata remain supported. For new final annotations, `prompt_id` should normally match the visible Prompt ID. Deprecated v1 annotation values remain readable only for lineage/backward compatibility; the new writer rejects them, the dataset builder refuses them as targets, and replay exposes them to runtime only as `UNKNOWN`.

No official Prompt ground truth is created in this phase.
