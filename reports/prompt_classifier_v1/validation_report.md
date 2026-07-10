# PromptClassifier v1 Validation Report

- Samples: 599
- Threshold: 0.55
- Accuracy: 0.863105
- Balanced accuracy: 0.752306
- Macro precision / recall / F1: 0.891276 / 0.752306 / 0.783820
- IDLE -> WAITING: 0
- WAITING -> IDLE: 0
- IDLE/WAITING mutual confusion: 0.000000
- UNKNOWN ratio: 0.071786
- Mean / p50 / p95 latency (ms): 0.9350 / 0.8544 / 1.7503
- Error count: 82

## Per class

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| IDLE | 0.719101 | 0.820513 | 0.766467 | 78 |
| WAITING | 0.997167 | 0.985994 | 0.991549 | 357 |
| READY | 0.848837 | 0.839080 | 0.843931 | 87 |
| NONE | 1.000000 | 0.363636 | 0.533333 | 77 |

## Per session

- session_20260710_125441: macro-F1=0.783820, IDLE=0.820513, WAITING=0.985994, READY=0.839080, UNKNOWN=0.071786
