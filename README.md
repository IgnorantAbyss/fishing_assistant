# Fishing Assistant

This project is a local, screenshot-based helper for recognising a fishing mini-game UI. It does not read game memory, inspect packets, inject into a process, or bypass anti-cheat measures.

## Scope

The project supports offline/replay analysis and an explicitly invoked guarded
Live runtime. Real input remains opt-in, foreground-only, allowlisted, and
fail-closed; tests and recorded-observation modes never emit input.

## Setup

Python 3.10 to 3.12 is preferred. If none is installed, the available Python version can be used as a temporary development fallback.

```powershell
py -0p
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the labelled PNG files in `assets/reference/` using the names defined by `src/state_detector.py`.

## Phase 1: static-image detection

The static detector compares UI-only image regions with the supplied references and reports `IDLE`, `WAITING`, `READY`, `HOOK`, `PRESS`, or `GET` with confidence and debug evidence. It also returns `UNKNOWN` when the result is unsafe to classify, including a black or uniform image.

```powershell
python tools\visual_state_report.py --image assets\reference\idle.png
python tools\visual_state_report.py --all
python tools\visual_state_report.py --all --json
python -m pytest -q
```

Static-report overlays are written to `logs/static_reports/` and never replace the source images.

## Phase 2: configurable ROI and safety thresholds

`config/roi.yaml` defines normalized (`0.0` to `1.0`) UI rectangles. `screen_reference` documents the source capture size; every ROI is converted to the actual image dimensions at runtime. If the file is unavailable, safe built-in ROI defaults are used.

`config/thresholds.yaml` controls the confidence floor, the `UNKNOWN` cutoff, a future FSM stability value, and debug-file retention. A low-confidence state never falls back to `IDLE`.

Preview configured rectangles without modifying an input image:

```powershell
python tools\roi_preview.py --image assets\reference\ready.png
python tools\roi_preview.py --all
```

Preview copies are written to `logs/roi_preview/`. To calibrate a single ROI from a static screenshot with the OpenCV mouse selector:

```powershell
python tools\calibrate_roi.py --image assets\reference\ready.png --roi top_prompt
```

`src/debug_saver.py` is event-driven. It saves only state changes, low-confidence results, `UNKNOWN`, exceptions, or an explicit `save_all=True` request. Count, size, and age retention limits are read from `thresholds.yaml`.

Run the full offline health check:

```powershell
python tools\agent_check.py
```

It checks reference files and configuration, runs the static report and pytest, then writes `reports/agent_check_latest.md`.

## Phase 3: offline HOOK and PRESS components

The HOOK detector measures the configured `hook_bar` ROI, reporting the coloured fill ratio and an optional bright divider line. It always returns `should_press_space: false`; this project phase contains no input behaviour.

```powershell
python tools\hook_report.py --image assets\reference\hook.png
python tools\hook_report.py --image assets\reference\hook2.png
```

The PRESS detector finds the purple key-cell glyphs inside `press_sequence` and uses image templates dynamically extracted from the supplied canonical `press.png`; it does not use general OCR.

```powershell
python tools\press_report.py --image assets\reference\press.png
python tools\press_report.py --image assets\reference\press2.png
```

These report overlays are written to `logs/hook_reports/` and `logs/press_reports/`, which are ignored by Git.

## Phase 4: capture-only and replay

Capture is opt-in through `collect_screenshot.py`. It uses `mss` only after the command starts, saves JPEG frames to an ignored replay-session directory, and can be stopped safely with `Ctrl+C`. It does not send or listen for any game input.

```powershell
python tools\collect_screenshot.py --list-monitors
python tools\collect_screenshot.py --duration 120 --interval 0.2 --jpg-quality 80 --monitor 1
```

Run detectors offline over an existing saved session:

```powershell
python tools\replay_detector.py --session assets\replay\sessions\session_YYYYMMDD_HHMMSS
python tools\replay_detector.py --latest
python tools\replay_summary.py --session assets\replay\sessions\session_YYYYMMDD_HHMMSS
```

Each replay session contains a `manifest.json`, `frames/`, `replay_results.csv`, and `replay_report.md`. Replay sessions and raw captures are ignored by Git; only the implementation and tests are tracked.

## Live window targeting

The Live runtime can resolve the supported borderless game window by executable
basename and an optional title prefix. Resolution succeeds only when exactly one
visible, non-minimized top-level window has a non-empty title and non-zero client
area. The resolved HWND and PID remain fixed for the session; the runtime never
switches to another window automatically.

```powershell
.\.venv\Scripts\python.exe tools\run_live_detect_only.py `
  --process-name BlackDesert64 `
  --window-title-prefix "黑色沙漠" `
  --capture-backend mss-region `
  --duration-seconds 600 `
  --evidence-mode diagnostic `
  --max-completed-cycles 3 `
  --no-overlay `
  --emit-actions true `
  --action-sink sendinput `
  --action-allowlist CAST,START_HOOK,COLLECT `
  --panic-key F12
```

The existing exact-title mode remains available as an explicit override:

```powershell
.\.venv\Scripts\python.exe tools\run_live_detect_only.py `
  --window-title "黑色沙漠 - 525411" `
  --capture-backend mss-region `
  --duration-seconds 600 `
  --evidence-mode diagnostic `
  --max-completed-cycles 3 `
  --no-overlay `
  --emit-actions false
```

## Replay labelling and dataset preparation

The primary Ground Truth workflow uses the original, unobstructed screenshots. Detector predictions are never converted into labels automatically.

```powershell
# 1. 擷取素材
python tools\collect_screenshot.py --duration 120 --interval 0.2 --jpg-quality 80

# 2. 在 frames/ 查看原圖，找出狀態切換幀

# 3. 一次建立 Ground Truth（PowerShell 反引號必須位於每行最後）
python tools\create_ground_truth.py --latest `
  --range 1-451:WAITING `
  --range 452-479:READY `
  --range 480-506:HOOK `
  --range 507-517:PRESS `
  --range 518-529:IGNORE `
  --range 530-558:IDLE `
  --range 559-600:WAITING

# 4. 所有 session 標註完成後，統一準備資料集
python tools\prepare_training_dataset.py --seed 42 --rebuild

# 5. 查看資料報告
python tools\dataset_report.py
```

`create_ground_truth.py` accepts `--latest` or `--session session_xxx`, one or more `--range START-END:STATE` arguments, and `--force` for an intentional replacement. Valid states are `IDLE`, `WAITING`, `READY`, `HOOK`, `PRESS`, `GET`, and `IGNORE`; a session does not need a `GET` segment. Ranges must cover frame 1 through the final frame exactly once, without gaps or overlaps. The command previews the complete coverage and refuses a non-interactive overwrite unless `--force` is supplied.

`tools/annotate_ground_truth.py` remains available only as an optional experimental overlay tool. Because its overlay can obscure UI details, it is not part of the recommended quick-start labelling flow.

The fixed dataset split is session-based: four train sessions, one validation session, and two held-out test sessions. Only train rows are balanced. Validation and test preserve every non-`IGNORE` frame and are marked `evaluation_only` in the manifests. This stage prepares crops and manifests only; it does not train a model.

## Future phases

- Per-state detectors for hook-bar progress and WASD sequences.
- Screen-capture-only `detect-only` mode with saved debug screenshots.
- A guarded FSM and, only after detection validation, optional keyboard assistance with a safe pause/stop mechanism.

## Troubleshooting

- Different UI scale or resolution: calibrate normalized ROI values and recollect UI-only templates.
- Detection inaccuracies: inspect the static-report overlays and raw state scores.
- Future input support: `pydirectinput` will have a `pyautogui` fallback, but neither is used in the current phases.
