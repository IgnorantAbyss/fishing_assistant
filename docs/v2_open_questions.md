# Hybrid Runtime v2 Open Questions

## Fixed for the current supported environment

- Resolution: 2560x1440 only.
- UI scale: fixed.
- Language: zh-TW.
- Window mode: borderless.
- Prompt position: fixed.
- Prompt ROI source of truth: pixel coordinates; normalized coordinates are derived display metadata only.

Other resolutions, UI scales, languages, window modes, and Prompt positions are unsupported. This phase does not attempt anchor detection, adaptation, localization, or window calibration.

## Still unresolved and requiring user evidence

- What is the foreground game window's executable/process identity?
- Which unapproved pixel Prompt ROI contains complete required content without unrelated bottom UI or excessive background?
- What visible Prompt kinds actually occur, including PRESS and GET flows, and which belong to `OTHER_PROMPT` or `NO_PROMPT`?
- Which visually distinct prompts need separate optional `prompt_id` values?
- Which transition frames should be annotated `IGNORE`?
- Who will manually approve the final Prompt ROI and record that approval?
- Which new, untouched sessions will be collected for v2 final testing?
- What cooldowns and timeout values are safe in live play?

Until these are answered, Prompt ROI remains `unapproved`, PromptObserver remains `unimplemented`, action emission remains disabled, and v2 final test remains `not_collected`. Contact sheets and inventory rows are review evidence only; they are not approval, Prompt ground truth, or a final recommendation.
