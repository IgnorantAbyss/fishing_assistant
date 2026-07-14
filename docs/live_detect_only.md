# Live detect-only runtime

This runtime observes a supported game window and records what the frozen v2
pipeline **would** do. It never emits keyboard or mouse input. The final Prompt
bundle is immutable at runtime: thresholds, ambiguity margin, IDLE stability,
ROI, preprocessing, all 35 formal-session medoids, and the bounded reviewed
Live calibration candidate set are loaded from the verified artifact under
`artifacts/prompt_observer/prototype_v1`.

## Supported environment

- Windows, borderless window, exact window-title lookup
- 2560 x 1440, fixed UI scale, zh-TW
- approved Prompt ROI `[940, 36, 1620, 100]`
- `config/fishing_v2.yaml` with `safety.emit_actions: false`

## Capture backends

`--capture-backend mss-region` is the default and preserves the original
behavior. It uses the exact title to resolve an HWND, validates that its process
is `BlackDesert64`, maps the client rectangle to desktop coordinates, and then
calls `mss.grab(region)`. The pixels are still desktop pixels: covering windows
and the diagnostic overlay can therefore appear in the frame. Use
`--no-overlay` with this backend. A borderless 2560 x 1440 client area can cover
the complete monitor even though it was selected through an HWND.

`--capture-backend windows-graphics-capture` is the HWND-surface adapter. Its
native provider contract receives the already validated HWND, must create the
capture item from that same handle, and converts one BGRA/RGBA surface to
contiguous `uint8` BGR at the adapter boundary. It does not use `PrintWindow`,
`BitBlt`, monitor capture, fixed border offsets, or resizing. This repository's
Python 3.14 environment currently has no verified WGC binding, so that option
fails preflight with a compatibility diagnostic instead of claiming that MSS
pixels are window capture. The minimal deployment follow-up is a tested native
WGC bridge that implements the documented HWND provider contract.

Fallback is opt-in only. Add `--allow-mss-fallback` to a WGC command to permit
an initialization failure to use `mss-region`; the warning and fallback reason
are written to the event log and session summary. Once capture starts, HWND
loss, minimization, invalid capture item, empty/invalid frames, capture failure,
or client/frame size changes stop the session safely and never switch sources.

Every backend reports its backend name, HWND, process, exact title, raw and
converted shapes, dtype, channel order, client size/region, and first-frame
SHA-256. The shared output contract is contiguous `uint8` `H x W x 3` BGR.

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

## Windows Graphics Capture command

This command is ready for the adapter contract, but on the current Python 3.14
environment it will stop at preflight until a verified native WGC provider is
available:

```powershell
.\.venv\Scripts\python.exe tools\run_live_detect_only.py `
  --window-title "黑色沙漠 - 524983" `
  --capture-backend windows-graphics-capture `
  --duration-seconds 30 `
  --output-dir reports\fishing_v2\live_detect_only `
  --show-overlay `
  --save-transition-frames `
  --prompt-bundle artifacts\prompt_observer\prototype_v1 `
  --max-fps 25 `
  --emit-actions false
```

An independent no-activate overlay is not part of the target HWND content by
design, but this still needs a real-window smoke test after a provider becomes
available. Whether a minimized DirectX window continues to produce frames is
also not guaranteed; this runtime deliberately stops when the window is
minimized.

## MSS desktop-region fallback command

Use the exact visible game-window title. A two-minute run is a reasonable first
check; no raw frame stream is retained.

```powershell
.\.venv\Scripts\python.exe tools\run_live_detect_only.py `
  --window-title "黑色沙漠 - 524983" `
  --capture-backend mss-region `
  --duration-seconds 30 `
  --output-dir reports\fishing_v2\live_detect_only `
  --no-overlay `
  --save-transition-frames `
  --prompt-bundle artifacts\prompt_observer\prototype_v1 `
  --max-fps 25 `
  --emit-actions false
```

Keep the game window in the foreground. The optional overlay is a separate,
non-activating diagnostic window. Stop safely with `Ctrl+C`. A capture exception
or resolution change also causes a safe stop and writes the reason to the
session.

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
