# Fishing Assistant Agent Check

- Generated (UTC): 2026-07-10 19:53:46
- Python: 3.14.5 (tags/v3.14.5:5607950, May 10 2026, 10:43:50) [MSC v.1944 64 bit (AMD64)]
- OS: Windows-11-10.0.26200-SP0
- Reference image count: 11
- ROI config: OK (D:\project\fishing_assistant\config\roi.yaml)
- Threshold config: OK (D:\project\fishing_assistant\config\thresholds.yaml)
- Capture config: OK (D:\project\fishing_assistant\config\capture.yaml)

## Static state report

| Image | Expected state | Detected state | Confidence |
| --- | --- | --- | ---: |
| idle.png | IDLE | IDLE | 98.18% |
| idle2.png | IDLE | IDLE | 98.18% |
| waiting.png | WAITING | WAITING | 100.00% |
| ready.png | READY | READY | 97.62% |
| ready2.png | READY | READY | 97.62% |
| hook.png | HOOK | HOOK | 100.00% |
| hook2.png | HOOK | HOOK | 100.00% |
| press.png | PRESS | PRESS | 100.00% |
| press2.png | PRESS | PRESS | 100.00% |
| get.png | GET | GET | 97.52% |
| get2.png | GET | GET | 97.52% |

## Command results

- `visual_state_report.py --all --json`: exit code 0
- `pytest -q`: exit code 0
- replay session/detector tests: exit code 0

## Pytest output

```text
........................................................................ [ 56%]
.......................................................                  [100%]
127 passed in 15.55s
```

## Failure summary

- None

## Replay session/detector test output

```text
...                                                                      [100%]
3 passed in 4.25s
```

## Debug output paths

- D:\project\fishing_assistant\logs\static_reports\idle_idle_debug.png
- D:\project\fishing_assistant\logs\static_reports\idle2_idle_debug.png
- D:\project\fishing_assistant\logs\static_reports\waiting_waiting_debug.png
- D:\project\fishing_assistant\logs\static_reports\ready_ready_debug.png
- D:\project\fishing_assistant\logs\static_reports\ready2_ready_debug.png
- D:\project\fishing_assistant\logs\static_reports\hook_hook_debug.png
- D:\project\fishing_assistant\logs\static_reports\hook2_hook_debug.png
- D:\project\fishing_assistant\logs\static_reports\press_press_debug.png
- D:\project\fishing_assistant\logs\static_reports\press2_press_debug.png
- D:\project\fishing_assistant\logs\static_reports\get_get_debug.png
- D:\project\fishing_assistant\logs\static_reports\get2_get_debug.png
