# Prototype PromptObserver LOSO Summary

- Frames: **4200** (evaluated 4059, IGNORE 141)
- Macro F1: **0.9923**
- Accuracy: **0.9899**
- Non-IGNORE UNKNOWN rate: **0.0101**
- IGNORE rejection rate: **1.0000**
- IGNORE -> IDLE_CAST: **0**
- Feature: per-frame CLAHE/local background subtraction plus luminance text mask and Sobel edges; cosine prototype similarity.
- Isolation: each fold uses six sessions; nested training-session holdouts calibrate similarity and ambiguity rejection.
- IDLE transition guard: fold-local training IGNORE runs determine the required consecutive IDLE observations.
- IGNORE is calibration/evaluation negative evidence and never a prototype.

## Per-class metrics

| label | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| IDLE_CAST | 1.0000 | 0.9313 | 0.9644 | 451 |
| WAITING_IN_PROGRESS | 1.0000 | 0.9968 | 0.9984 | 2806 |
| READY_BITE | 1.0000 | 0.9976 | 0.9988 | 409 |
| HOOK_INSTRUCTION | 1.0000 | 1.0000 | 1.0000 | 245 |
| PRESS_INSTRUCTION | 1.0000 | 1.0000 | 1.0000 | 148 |

## Confusion matrix

| actual \ predicted | IDLE_CAST | WAITING_IN_PROGRESS | READY_BITE | HOOK_INSTRUCTION | PRESS_INSTRUCTION | UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|
| IDLE_CAST | 420 | 0 | 0 | 0 | 0 | 31 |
| WAITING_IN_PROGRESS | 0 | 2797 | 0 | 0 | 0 | 9 |
| READY_BITE | 0 | 0 | 408 | 0 | 0 | 1 |
| HOOK_INSTRUCTION | 0 | 0 | 0 | 245 | 0 | 0 |
| PRESS_INSTRUCTION | 0 | 0 | 0 | 0 | 148 | 0 |

## Dataset counts by session

- session_20260709_192315: `{'IDLE_CAST': 53, 'WAITING_IN_PROGRESS': 491, 'READY_BITE': 11, 'HOOK_INSTRUCTION': 18, 'PRESS_INSTRUCTION': 12, 'IGNORE': 15}`
- session_20260710_061220: `{'IDLE_CAST': 29, 'WAITING_IN_PROGRESS': 493, 'READY_BITE': 28, 'HOOK_INSTRUCTION': 27, 'PRESS_INSTRUCTION': 8, 'IGNORE': 15}`
- session_20260710_123210: `{'IDLE_CAST': 48, 'WAITING_IN_PROGRESS': 453, 'READY_BITE': 55, 'HOOK_INSTRUCTION': 17, 'PRESS_INSTRUCTION': 10, 'IGNORE': 17}`
- session_20260710_124419: `{'IDLE_CAST': 72, 'WAITING_IN_PROGRESS': 414, 'READY_BITE': 52, 'HOOK_INSTRUCTION': 25, 'PRESS_INSTRUCTION': 22, 'IGNORE': 15}`
- session_20260710_125441: `{'IDLE_CAST': 81, 'WAITING_IN_PROGRESS': 357, 'READY_BITE': 87, 'HOOK_INSTRUCTION': 35, 'PRESS_INSTRUCTION': 24, 'IGNORE': 16}`
- session_20260710_130308: `{'IDLE_CAST': 96, 'WAITING_IN_PROGRESS': 309, 'READY_BITE': 79, 'HOOK_INSTRUCTION': 61, 'PRESS_INSTRUCTION': 24, 'IGNORE': 31}`
- session_20260710_131254: `{'IDLE_CAST': 72, 'WAITING_IN_PROGRESS': 289, 'READY_BITE': 97, 'HOOK_INSTRUCTION': 62, 'PRESS_INSTRUCTION': 48, 'IGNORE': 32}`

## Fold calibration

| held out | prototypes | macro F1 | margin | IDLE stability | class thresholds |
|---|---:|---:|---:|---:|---|
| session_20260709_192315 | 30 | 0.9880 | 0.0189 | 4 | IDLE_CAST=0.7546, WAITING_IN_PROGRESS=0.0730, READY_BITE=-0.0784, HOOK_INSTRUCTION=-0.1128, PRESS_INSTRUCTION=0.2614 |
| session_20260710_061220 | 30 | 0.9887 | 0.2814 | 4 | IDLE_CAST=0.1811, WAITING_IN_PROGRESS=0.8910, READY_BITE=0.9029, HOOK_INSTRUCTION=-0.1158, PRESS_INSTRUCTION=0.3142 |
| session_20260710_123210 | 30 | 0.9939 | 0.0227 | 3 | IDLE_CAST=0.8979, WAITING_IN_PROGRESS=0.0619, READY_BITE=0.9109, HOOK_INSTRUCTION=0.9088, PRESS_INSTRUCTION=0.1805 |
| session_20260710_124419 | 30 | 0.9906 | 0.2836 | 4 | IDLE_CAST=0.1791, WAITING_IN_PROGRESS=0.0924, READY_BITE=0.9049, HOOK_INSTRUCTION=-0.1128, PRESS_INSTRUCTION=0.5931 |
| session_20260710_125441 | 30 | 0.9934 | 0.0202 | 4 | IDLE_CAST=0.1759, WAITING_IN_PROGRESS=0.8949, READY_BITE=0.9029, HOOK_INSTRUCTION=0.9099, PRESS_INSTRUCTION=0.8762 |
| session_20260710_130308 | 30 | 0.9940 | 0.2791 | 4 | IDLE_CAST=0.1827, WAITING_IN_PROGRESS=0.8961, READY_BITE=0.6282, HOOK_INSTRUCTION=-0.1128, PRESS_INSTRUCTION=0.4151 |
| session_20260710_131254 | 30 | 0.9939 | 0.2802 | 4 | IDLE_CAST=0.1764, WAITING_IN_PROGRESS=0.0624, READY_BITE=0.2750, HOOK_INSTRUCTION=-0.1128, PRESS_INSTRUCTION=0.1805 |

## Event metrics

- IDLE_CAST: 12/12 episodes; mean/max latency=2.5833333333333335/3 frames
- WAITING_IN_PROGRESS: 13/13 episodes; mean/max latency=0.0/0 frames
- READY_BITE: 9/9 episodes; mean/max latency=0.0/0 frames
- HOOK_INSTRUCTION: 9/9 episodes; mean/max latency=0.0/0 frames
- PRESS_INSTRUCTION: 8/8 episodes; mean/max latency=0.0/0 frames

## Failure classification

- {'E_transition_idle_stability': 31, 'D_top1_top2_ambiguity': 6, 'C_rejection_threshold': 4}
