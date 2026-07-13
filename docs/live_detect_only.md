# Live detect-only runtime

This runtime observes a supported game window and records what the frozen v2
pipeline **would** do. It never emits keyboard or mouse input. The final Prompt
bundle is immutable at runtime: thresholds, ambiguity margin, IDLE stability,
ROI, preprocessing and all 35 session medoids are loaded from the verified
artifact under `artifacts/prompt_observer/prototype_v1`.

## Supported environment

- Windows, borderless window, exact window-title lookup
- 2560 x 1440, fixed UI scale, zh-TW
- approved Prompt ROI `[940, 36, 1620, 100]`
- `config/fishing_v2.yaml` with `safety.emit_actions: false`

Preflight stops before the observation loop when capture fails, resolution is
wrong, the ROI is invalid, the final bundle/hash is invalid, or action emission
is enabled. The controller has no action sink. `--emit-actions true` is refused
at the CLI boundary before capture is opened.

## Build and verify the final bundle

```powershell
.\.venv\Scripts\python.exe tools\build_final_prompt_bundle.py
.\.venv\Scripts\python.exe tools\validate_final_prompt_bundle.py
```

The validation is a deployment regression over all seven formal sessions. It
does not replace the leave-one-session-out generalization report.

## Short first live run

Use the exact visible game-window title. A two-minute run is a reasonable first
check; no raw frame stream is retained.

```powershell
.\.venv\Scripts\python.exe tools\run_live_detect_only.py `
  --window-title "<exact game window title>" `
  --duration-seconds 120 `
  --output-dir reports\fishing_v2\live_detect_only `
  --show-overlay `
  --save-transition-frames `
  --prompt-bundle artifacts\prompt_observer\prototype_v1 `
  --max-fps 25 `
  --emit-actions false
```

Keep the game window in the foreground. The overlay is diagnostic and is made
non-activating on Windows. Stop safely with `Ctrl+C`. A capture exception or a
resolution change also causes a safe stop and writes the reason to the session.

## Output and review

Each run creates a new, never-overwritten directory:

```text
reports/fishing_v2/live_detect_only/session_<timestamp>/
  session_summary.md
  session_summary.json
  events.jsonl
  transitions.csv
  review_items.csv
  screenshots/
```

`events.jsonl` separates raw ActionIntent proposal counts from deduplicated
would-fire opportunities (`WOULD_CAST`, `WOULD_START_HOOK`,
`WOULD_HOOK_ACTION`, `WOULD_PRESS_SEQUENCE`, `WOULD_COLLECT`). Screenshots are
bounded to transitions when requested, would-fire events, sustained UNKNOWN,
SYNC_REQUIRED, and detector conflicts. `action_applied` must remain `false` in
every event and zero in the summary.

Review `review_items.csv` and the small event screenshot set after each run.
Do not treat a successful detect-only session as authorization for production
input: live precision, event timing, capture stability, and achieved burst FPS
still require human review.
