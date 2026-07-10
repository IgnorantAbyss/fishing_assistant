"""Train PromptClassifier v1 with two-stage MobileNetV3 transfer learning."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.prompt_dataset import CLASS_NAMES, PromptManifestDataset  # noqa: E402
from src.ml.prompt_metrics import classification_metrics  # noqa: E402
from src.ml.prompt_model import (  # noqa: E402
    freeze_for_head_training,
    save_prompt_checkpoint,
    unfreeze_last_feature_block,
    build_prompt_model,
)
from src.ml.prompt_transforms import build_prompt_transform  # noqa: E402
from tools.inspect_prompt_dataset import inspect_prompt_dataset  # noqa: E402


DATASET_ROOT = PROJECT_ROOT / "datasets" / "fishing_v2"
MANIFEST_PATH = DATASET_ROOT / "prompt" / "manifest.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "prompt_classifier_v1"
REPORT_DIR = PROJECT_ROOT / "reports" / "prompt_classifier_v1"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Prompt classifier config root must be a mapping")
    model = data.get("model", {})
    if model.get("name") != "mobilenet_v3_small" or model.get("num_classes") != 4:
        raise ValueError("PromptClassifier v1 requires mobilenet_v3_small with four classes")
    return data


def _class_weights(dataset: PromptManifestDataset, device: torch.device) -> torch.Tensor:
    counts = dataset.class_counts
    if any(counts[name] <= 0 for name in CLASS_NAMES):
        raise ValueError(f"Every prompt class needs selected train samples: {counts}")
    total = sum(counts.values())
    return torch.tensor(
        [total / (len(CLASS_NAMES) * counts[name]) for name in CLASS_NAMES],
        dtype=torch.float32,
        device=device,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    freeze_features: bool,
    use_amp: bool,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    if training and freeze_features:
        model.features.eval()
    elif training:
        # Frozen blocks must not mutate BatchNorm running statistics during Stage B.
        model.features[:-1].eval()
        model.features[-1].train()
    total_loss = 0.0
    labels_all: list[str] = []
    predictions_all: list[str] = []
    scaler = torch.amp.GradScaler("cuda") if use_amp and training else None
    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += float(loss.item()) * labels.shape[0]
        predicted = logits.detach().argmax(dim=1).cpu().tolist()
        labels_all.extend(CLASS_NAMES[index] for index in labels.cpu().tolist())
        predictions_all.extend(CLASS_NAMES[index] for index in predicted)
    metrics = classification_metrics(labels_all, predictions_all)
    return (
        total_loss / len(loader.dataset),
        float(metrics["accuracy"]),
        float(metrics["macro_f1"]),
    )


def _draw_training_curve(history: list[dict[str, Any]], destination: Path) -> None:
    width, height = 900, 520
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    left, top, right, bottom = 70, 35, width - 30, height - 65
    cv2.rectangle(image, (left, top), (right, bottom), (70, 70, 70), 1)
    for fraction in np.linspace(0, 1, 6):
        y = round(bottom - fraction * (bottom - top))
        cv2.line(image, (left, y), (right, y), (220, 220, 220), 1)
        cv2.putText(image, f"{fraction:.1f}", (20, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    if len(history) > 1:
        for key, color in (("train_accuracy", (230, 120, 20)), ("validation_macro_f1", (30, 150, 30))):
            points = []
            for index, item in enumerate(history):
                x = round(left + index * (right - left) / (len(history) - 1))
                y = round(bottom - float(item[key]) * (bottom - top))
                points.append((x, y))
            cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, color, 2)
    cv2.putText(image, "blue: train accuracy", (left, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 120, 20), 1)
    cv2.putText(image, "green: validation macro-F1", (left + 230, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 150, 30), 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), image)


def train(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    qa = inspect_prompt_dataset(create_contact_sheets=False)
    if not qa["valid"]:
        raise RuntimeError(f"Critical prompt dataset QA failure: {qa['critical_errors']}")
    seed = int(config["seed"])
    seed_everything(seed)
    model_config = config["model"]
    training = config["training"]
    width, height = int(model_config["input_width"]), int(model_config["input_height"])
    train_dataset = PromptManifestDataset(
        MANIFEST_PATH,
        DATASET_ROOT,
        "train",
        build_prompt_transform(True, width=width, height=height),
    )
    validation_dataset = PromptManifestDataset(
        MANIFEST_PATH,
        DATASET_ROOT,
        "validation",
        build_prompt_transform(False, width=width, height=height),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=int(training["num_workers"]),
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=int(training["num_workers"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model = build_prompt_model(
        num_classes=int(model_config["num_classes"]),
        pretrained=bool(model_config["pretrained"]),
    ).to(device)
    weights = _class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    global_epoch = 0
    early_stopped = False
    started = time.perf_counter()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    stages = (
        (
            "head",
            int(training["head_epochs"]),
            float(training["head_learning_rate"]),
            True,
        ),
        (
            "finetune_last_block",
            int(training["finetune_epochs"]),
            float(training["finetune_learning_rate"]),
            False,
        ),
    )
    for stage, epochs, learning_rate, freeze_features in stages:
        if stage == "head":
            freeze_for_head_training(model)
        else:
            unfreeze_last_feature_block(model)
        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=float(training["weight_decay"]),
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        no_improvement = 0
        for _ in range(epochs):
            global_epoch += 1
            train_loss, train_accuracy, train_f1 = _run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                freeze_features=freeze_features,
                use_amp=use_amp,
            )
            validation_loss, validation_accuracy, validation_f1 = _run_epoch(
                model,
                validation_loader,
                criterion,
                device,
                optimizer=None,
                freeze_features=False,
                use_amp=use_amp,
            )
            scheduler.step(validation_f1)
            record = {
                "epoch": global_epoch,
                "stage": stage,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "train_macro_f1": train_f1,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": validation_f1,
            }
            history.append(record)
            print(json.dumps(record))
            save_prompt_checkpoint(
                MODEL_DIR / "last_model.pt",
                model,
                epoch=global_epoch,
                validation_macro_f1=validation_f1,
                input_width=width,
                input_height=height,
            )
            if validation_f1 > best_f1 + 1e-9:
                best_f1 = validation_f1
                best_epoch = global_epoch
                no_improvement = 0
                save_prompt_checkpoint(
                    MODEL_DIR / "best_model.pt",
                    model,
                    epoch=global_epoch,
                    validation_macro_f1=validation_f1,
                    input_width=width,
                    input_height=height,
                )
            else:
                no_improvement += 1
            if stage != "head" and no_improvement >= int(training["early_stopping_patience"]):
                early_stopped = True
                break
        if early_stopped:
            break

    duration = time.perf_counter() - started
    (MODEL_DIR / "class_names.json").write_text(
        json.dumps(list(CLASS_NAMES), indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(config_path, MODEL_DIR / "training_config.yaml")
    (MODEL_DIR / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mixed_precision": use_amp,
        "seed": seed,
        "deterministic_algorithms": True,
        "train_counts": train_dataset.class_counts,
        "validation_counts": validation_dataset.class_counts,
        "class_weights": {name: float(weights[index].item()) for index, name in enumerate(CLASS_NAMES)},
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "early_stopped": early_stopped,
        "duration_seconds": duration,
        "checkpoint_path": str(MODEL_DIR / "best_model.pt"),
    }
    (REPORT_DIR / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# PromptClassifier v1 Training Report",
        "",
        f"- Python: {report['python_version']}",
        f"- torch / torchvision: {report['torch_version']} / {report['torchvision_version']}",
        f"- Device: {report['device']}",
        f"- CUDA device: {report['cuda_device_name']}",
        f"- Train counts: `{report['train_counts']}`",
        f"- Validation counts: `{report['validation_counts']}`",
        f"- Class weights: `{report['class_weights']}`",
        f"- Best epoch: {best_epoch}",
        f"- Best validation macro-F1: {best_f1:.6f}",
        f"- Early stopped: {early_stopped}",
        f"- Duration: {duration:.2f} seconds",
        f"- Checkpoint: `{report['checkpoint_path']}`",
        "",
        "## Epoch history",
        "",
        "| Epoch | Stage | Train loss | Train accuracy | Validation loss | Validation accuracy | Validation macro-F1 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in history:
        lines.append(
            f"| {item['epoch']} | {item['stage']} | {item['train_loss']:.4f} | "
            f"{item['train_accuracy']:.4f} | {item['validation_loss']:.4f} | "
            f"{item['validation_accuracy']:.4f} | {item['validation_macro_f1']:.4f} |"
        )
    (REPORT_DIR / "training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _draw_training_curve(history, REPORT_DIR / "training_curve.png")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PromptClassifier v1.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "prompt_classifier.yaml"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = train(args.config)
    print(f"best_validation_macro_f1: {report['best_validation_macro_f1']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
