"""Standalone inference API for PromptClassifier v1 (not production-integrated)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image
import torch

from src.config_loader import load_roi_config
from src.ml.prompt_dataset import CLASS_NAMES
from src.ml.prompt_model import MODEL_VERSION, load_prompt_checkpoint
from src.ml.prompt_transforms import build_prompt_transform


@dataclass(frozen=True)
class PromptPrediction:
    state: str
    confidence: float
    probabilities: Mapping[str, float]
    is_unknown: bool
    model_version: str


def _as_rgb_image(frame_or_crop: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(frame_or_crop, (str, Path)):
        with Image.open(frame_or_crop) as source:
            return source.convert("RGB")
    if isinstance(frame_or_crop, Image.Image):
        return frame_or_crop.convert("RGB")
    if isinstance(frame_or_crop, np.ndarray):
        if frame_or_crop.ndim != 3 or frame_or_crop.shape[2] not in {3, 4}:
            raise ValueError("NumPy prompt input must be an HxWx3 or HxWx4 image")
        # Repository capture frames are OpenCV BGR/BGRA arrays.
        channels = frame_or_crop[:, :, :3][:, :, ::-1]
        return Image.fromarray(np.ascontiguousarray(channels).astype(np.uint8), mode="RGB")
    raise TypeError(f"Unsupported prompt input type: {type(frame_or_crop)!r}")


def prepare_prompt_crop(frame_or_crop: str | Path | Image.Image | np.ndarray) -> Image.Image:
    image = _as_rgb_image(frame_or_crop)
    # Prompt crops are characteristically wide. Full game frames are <= 4:1.
    if image.width / max(image.height, 1) <= 4.0:
        roi = load_roi_config()
        left, top, right, bottom = roi.pixel_roi("top_prompt", image.width, image.height)
        image = image.crop((left, top, right, bottom))
    return image


class PromptClassifierV1:
    def __init__(
        self,
        model_path: str | Path,
        threshold_path: str | Path,
        device: str | None = None,
    ) -> None:
        requested = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        self.device = torch.device(requested)
        self.model, checkpoint = load_prompt_checkpoint(model_path, device=self.device)
        threshold_data = json.loads(Path(threshold_path).read_text(encoding="utf-8"))
        self.threshold = float(threshold_data["selected_threshold"])
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Prompt confidence threshold must be between zero and one")
        self.model_version = str(checkpoint.get("model_version", MODEL_VERSION))
        self.transform = build_prompt_transform(
            False,
            width=int(checkpoint.get("input_width", 320)),
            height=int(checkpoint.get("input_height", 96)),
        )

    def predict(
        self, frame_or_crop: str | Path | Image.Image | np.ndarray
    ) -> PromptPrediction:
        crop = prepare_prompt_crop(frame_or_crop)
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities_tensor = torch.softmax(self.model(tensor), dim=1)[0].cpu()
        confidence, index = probabilities_tensor.max(dim=0)
        confidence_value = float(confidence.item())
        raw_state = CLASS_NAMES[int(index.item())]
        is_unknown = confidence_value < self.threshold
        state = "UNKNOWN" if is_unknown else raw_state
        probabilities = {
            name: float(probabilities_tensor[position].item())
            for position, name in enumerate(CLASS_NAMES)
        }
        return PromptPrediction(
            state=state,
            confidence=confidence_value,
            probabilities=probabilities,
            is_unknown=is_unknown,
            model_version=self.model_version,
        )
