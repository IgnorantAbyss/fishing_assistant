# Fishing Assistant

This project is a local, screenshot-only helper for recognising the fishing mini-game UI. It does not read game memory, inspect packets, inject into a process, or bypass anti-cheat measures.

## Phase 1: static-image detection

Phase 1 is intentionally detection-only. It reads the supplied images in `assets/reference/`, compares fixed UI regions with OpenCV, and reports a state with evidence. It contains no screen capture or keyboard automation.

The currently recognised states are `IDLE`, `WAITING`, `READY`, `HOOK`, `PRESS`, and `GET`. Unknown screenshots are not yet assigned the `UNKNOWN` state; that confidence policy belongs to the later live-detection/FSM phase.

### Set up the virtual environment

Python 3.10–3.12 is preferred. If none is installed, use the available Python version as a temporary development fallback.

```powershell
py -0p
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the labelled PNG files in `assets/reference/` using the names listed in `src/state_detector.py`.

### Run a report

```powershell
python tools/visual_state_report.py --image assets/reference/idle.png
python tools/visual_state_report.py --all
python tools/visual_state_report.py --all --json
```

Each result includes `state`, `confidence`, `matched_features`, and a debug-image-path field. The report saves annotated static-image overlays in `logs/static_reports/`. ROI calibration, live `detect-only`, and `assist` mode are intentionally not implemented in this phase.

### Run the static tests

```powershell
python -m pytest -q
```

## Planned later phases

- Calibratable normalized ROI settings and a calibration tool.
- Per-state detectors for hook-bar progress and WASD sequences.
- Screen-capture-only `detect-only` mode with saved debug screenshots.
- A guarded FSM and, only after detection validation, optional keyboard assistance with a safe pause/stop mechanism.

## Troubleshooting (for later phases)

- Different resolution or UI scale: recalibrate normalized ROI values and recollect UI-only templates.
- State accuracy issues: save debug crops and compare them with the reference UI areas.
- `pydirectinput` input issues: the future input layer will provide a `pyautogui` fallback; neither package is used in Phase 1.
