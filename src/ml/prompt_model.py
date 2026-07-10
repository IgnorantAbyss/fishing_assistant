"""MobileNetV3 Small model and checkpoint helpers for PromptClassifier v1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.ml.prompt_dataset import CLASS_NAMES


MODEL_VERSION = "prompt_classifier_v1"


def build_prompt_model(*, num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    if num_classes != len(CLASS_NAMES):
        raise ValueError(f"PromptClassifier v1 requires {len(CLASS_NAMES)} classes")
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    final = model.classifier[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("Unexpected MobileNetV3 classifier layout")
    model.classifier[-1] = nn.Linear(final.in_features, num_classes)
    return model


def freeze_for_head_training(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_feature_block(model: nn.Module) -> None:
    freeze_for_head_training(model)
    for parameter in model.features[-1].parameters():
        parameter.requires_grad = True


def save_prompt_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    epoch: int,
    validation_macro_f1: float,
    input_width: int = 320,
    input_height: int = 96,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_version": MODEL_VERSION,
            "class_names": list(CLASS_NAMES),
            "input_width": input_width,
            "input_height": input_height,
            "epoch": epoch,
            "validation_macro_f1": validation_macro_f1,
            "state_dict": model.state_dict(),
        },
        destination,
    )


def load_prompt_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Prompt checkpoint class order does not match PromptClassifier v1")
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError("Unsupported prompt checkpoint model version")
    model = build_prompt_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
