# PromptClassifier v1 Training Report

- Python: 3.14.5
- torch / torchvision: 2.13.0+cpu / 0.28.0+cpu
- Device: cpu
- CUDA device: None
- Train counts: `{'IDLE': 195, 'WAITING': 294, 'READY': 147, 'NONE': 180}`
- Validation counts: `{'IDLE': 78, 'WAITING': 357, 'READY': 87, 'NONE': 77}`
- Class weights: `{'IDLE': 1.0461539030075073, 'WAITING': 0.6938775777816772, 'READY': 1.3877551555633545, 'NONE': 1.1333333253860474}`
- Best epoch: 14
- Best validation macro-F1: 0.777355
- Early stopped: False
- Duration: 72.08 seconds
- Checkpoint: `D:\project\fishing_assistant\models\prompt_classifier_v1\best_model.pt`

## Epoch history

| Epoch | Stage | Train loss | Train accuracy | Validation loss | Validation accuracy | Validation macro-F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | head | 1.1288 | 0.5257 | 0.9572 | 0.6043 | 0.5210 |
| 2 | head | 0.7805 | 0.7145 | 0.7922 | 0.7646 | 0.5944 |
| 3 | head | 0.7013 | 0.7353 | 0.5509 | 0.8097 | 0.6485 |
| 4 | head | 0.6108 | 0.7696 | 0.4648 | 0.8080 | 0.6532 |
| 5 | head | 0.5653 | 0.7990 | 0.4985 | 0.8097 | 0.6538 |
| 6 | finetune_last_block | 0.5812 | 0.8064 | 0.5614 | 0.8130 | 0.6636 |
| 7 | finetune_last_block | 0.5010 | 0.8235 | 0.5181 | 0.8197 | 0.6708 |
| 8 | finetune_last_block | 0.4629 | 0.8480 | 0.4963 | 0.8180 | 0.6689 |
| 9 | finetune_last_block | 0.4565 | 0.8407 | 0.4618 | 0.8180 | 0.6699 |
| 10 | finetune_last_block | 0.4225 | 0.8676 | 0.4615 | 0.8264 | 0.6785 |
| 11 | finetune_last_block | 0.4102 | 0.8652 | 0.4104 | 0.8531 | 0.7277 |
| 12 | finetune_last_block | 0.3771 | 0.8909 | 0.4431 | 0.8464 | 0.7170 |
| 13 | finetune_last_block | 0.3736 | 0.8787 | 0.4281 | 0.8497 | 0.7193 |
| 14 | finetune_last_block | 0.3969 | 0.8591 | 0.3908 | 0.8898 | 0.7774 |
| 15 | finetune_last_block | 0.3672 | 0.8848 | 0.4220 | 0.8664 | 0.7445 |
