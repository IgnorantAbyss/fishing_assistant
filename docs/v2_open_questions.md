# Hybrid Runtime v2 Open Questions

The following conditions are deliberately not assumed fixed:

- Which screen resolutions must be supported?
- Which game UI-scale settings must be supported?
- Which game languages/localizations must be recognized?
- Is capture expected in fullscreen, borderless, or windowed mode?
- What prompt kinds exist beyond IDLE, WAITING, and READY? Which belong to `OTHER_PROMPT`?
- Which normalized Prompt ROI candidate contains every required localized prompt while minimizing scenery and bottom UI?
- Who will approve the final Prompt ROI and record that approval?
- Which new, untouched sessions will be collected for v2 final testing?
- What constitutes foreground-window confirmation on the user's platform?
- What cooldowns and timeout values are safe in live play?

Until answered, Prompt ROI remains `unapproved`, PromptObserver remains `unimplemented`, action emission remains disabled, and v2 final test remains `not_collected`.
