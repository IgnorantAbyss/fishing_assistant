import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from src.ml.prompt_inference import PromptClassifierV1, prepare_prompt_crop
from src.ml.prompt_model import build_prompt_model, save_prompt_checkpoint
from src.ml.prompt_transforms import build_prompt_transform


def _classifier(tmp_path: Path, bias: list[float], threshold: float = 0.5) -> PromptClassifierV1:
    model = build_prompt_model(pretrained=False)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.classifier[-1].bias.data.copy_(torch.tensor(bias))
    checkpoint = tmp_path / ("_".join(str(value) for value in bias) + ".pt")
    save_prompt_checkpoint(checkpoint, model, epoch=1, validation_macro_f1=0.0)
    threshold_path = tmp_path / (checkpoint.stem + ".json")
    threshold_path.write_text(json.dumps({"selected_threshold": threshold}), encoding="utf-8")
    return PromptClassifierV1(checkpoint, threshold_path, device="cpu")


def test_low_confidence_prediction_is_unknown(tmp_path: Path) -> None:
    classifier = _classifier(tmp_path, [0.0, 0.0, 0.0, 0.0], threshold=0.5)
    prediction = classifier.predict(Image.new("RGB", (320, 96), "black"))
    assert prediction.state == "UNKNOWN"
    assert prediction.is_unknown is True
    assert prediction.confidence == pytest.approx(0.25)


def test_none_prediction_stays_none(tmp_path: Path) -> None:
    classifier = _classifier(tmp_path, [0.0, 0.0, 0.0, 10.0], threshold=0.5)
    prediction = classifier.predict(Image.new("RGB", (320, 96), "white"))
    assert prediction.state == "NONE"
    assert prediction.is_unknown is False


def test_unknown_never_falls_back_to_idle(tmp_path: Path) -> None:
    classifier = _classifier(tmp_path, [1.0, 0.0, 0.0, 0.0], threshold=0.9)
    assert classifier.predict(Image.new("RGB", (320, 96))).state == "UNKNOWN"


def test_full_frame_is_cropped_with_top_prompt_roi() -> None:
    crop = prepare_prompt_crop(Image.new("RGB", (2048, 1151), "white"))
    assert crop.size == (820, 92)


def test_wide_prompt_crop_is_not_cropped_again() -> None:
    crop = prepare_prompt_crop(Image.new("RGB", (1024, 115), "white"))
    assert crop.size == (1024, 115)


def test_inference_transform_matches_evaluation_transform(tmp_path: Path) -> None:
    classifier = _classifier(tmp_path, [0.0, 0.0, 10.0, 0.0])
    image = Image.new("RGB", (1024, 115), "red")
    assert torch.equal(classifier.transform(image), build_prompt_transform(False)(image))


def test_invalid_numpy_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="HxWx3"):
        prepare_prompt_crop(np.zeros((10, 10), dtype=np.uint8))
