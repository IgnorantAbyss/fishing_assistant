# Fishing Assistant

This project is a local, screenshot-only helper for recognising a fishing mini-game UI. It does not read game memory, inspect packets, inject into a process, or bypass anti-cheat measures.

## Scope

The project currently performs offline detection on existing image files only. It contains no live screen capture, assist mode, or keyboard input code.

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

## Future phases

- Per-state detectors for hook-bar progress and WASD sequences.
- Screen-capture-only `detect-only` mode with saved debug screenshots.
- A guarded FSM and, only after detection validation, optional keyboard assistance with a safe pause/stop mechanism.

## Troubleshooting

- Different UI scale or resolution: calibrate normalized ROI values and recollect UI-only templates.
- Detection inaccuracies: inspect the static-report overlays and raw state scores.
- Future input support: `pydirectinput` will have a `pyautogui` fallback, but neither is used in the current phases.
