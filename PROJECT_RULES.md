# Fishing Assistant Project Rules

## Environment

- This is a Windows project.
- Use the project-local `.venv`.
- Prefer running Python with:
  `.\.venv\Scripts\python.exe`
- Do not migrate the project to Pipenv, Poetry, or another environment manager.
- Do not assume another Python installation or version is available.
- Keep application code under `src/`, command-line tools under `tools/`, configuration under `config/`, and tests under `tests/`.

## Change Scope

- Prefer the smallest change that completely fixes the requested problem.
- When fixing one issue, do not proactively refactor unrelated modules.
- Unless the task explicitly requires it, do not modify:
  - PromptObserver
  - model bundles
  - detectors
  - detector thresholds
  - ROI
  - Ground Truth
  - datasets
- Do not weaken safety conditions for test convenience.
- Preserve existing recorded-observation and detect-only contracts when changing Live behavior.

## Action Safety

- Real input must fail closed.
- Preserve and maintain:
  - HWND validation
  - PID validation
  - process executable validation
  - client-size validation
  - foreground validation
  - integrity-level preflight
  - action allowlist
  - one-shot/deduplication guards
  - F12 panic latch
- Detector `OFF` or `None` must not be inferred directly as absent.
- `action_applied` means a complete OS input emission only; it does not mean the game visually accepted the action.
- Record visual acknowledgement separately from OS input emission.
- Tests and recorded-observation runs must not emit real input.

## Forbidden Techniques

Unless the user explicitly requests otherwise, do not add:

- mouse automation
- background input
- `PostMessage`
- automatic foreground-window switching
- automatic elevation
- process injection
- driver injection
- anti-cheat bypass
- any technique intended to evade game protections

## Testing

- Pytest must not send real keyboard or mouse input.
- Win32 APIs must be mockable.
- When fixing a Live bug, prefer a regression test that reproduces the affected session timeline.
- Test multi-cycle behavior and retained state, not only a clean controller instance.
- After modifications, run:
  - `.\.venv\Scripts\python.exe -m pytest -q`
  - `.\.venv\Scripts\python.exe tools\agent_check.py`
- If the project's actual correct command differs, use the command supported by the current repository.
- Use an isolated `--basetemp` when the local pytest cache or temp directory is unavailable.

## Git

Unless the task explicitly requires them, do not commit:

- `reports/agent_check_latest.md`
- Live sessions
- diagnostic videos
- caches
- `.pytest_cache`
- local absolute paths
- temporary files

- Do not push.
- Do not overwrite the user's existing uncommitted changes.
- Run `git diff --check` before committing.
- Report the commit hash and working-tree status in the final response.

## Task Instructions

- Read this file at the start of every task.
- Explicit instructions in the current prompt take precedence over this file.
- If the current request conflicts with this file, identify the conflict before proceeding instead of guessing.
- Keep the final report focused on actual changes, test results, and incomplete items; do not repeat the full specification.
