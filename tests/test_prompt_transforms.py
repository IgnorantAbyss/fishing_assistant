import random

import numpy as np
from PIL import Image
import torch

from src.ml.prompt_transforms import (
    LetterboxResize,
    PromptTransform,
    RandomJPEGCompression,
    build_prompt_transform,
)


def _image(width: int = 800, height: int = 90) -> Image.Image:
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    pixels[:, :, 1] = 80
    return Image.fromarray(pixels, mode="RGB")


def test_transform_output_shape_is_three_by_96_by_320() -> None:
    output = build_prompt_transform(False)(_image())
    assert output.shape == (3, 96, 320)


def test_letterbox_preserves_aspect_ratio_and_pads() -> None:
    output = np.asarray(LetterboxResize(320, 96)(Image.new("RGB", (100, 100), "red")))
    assert np.all(output[:, :112] == 0)
    assert np.all(output[:, 112:208, 0] == 255)
    assert np.all(output[:, 208:] == 0)


def test_validation_transform_is_deterministic() -> None:
    transform = build_prompt_transform(False)
    first = transform(_image())
    second = transform(_image())
    assert torch.equal(first, second)


def test_training_policy_has_no_horizontal_or_vertical_flip() -> None:
    names = {type(item).__name__ for item in PromptTransform(True).operations}
    assert "RandomHorizontalFlip" not in names
    assert "RandomVerticalFlip" not in names


def test_training_affine_has_zero_rotation_and_small_translation() -> None:
    affine = next(item for item in PromptTransform(True).operations if type(item).__name__ == "RandomAffine")
    assert affine.degrees == [0.0, 0.0]
    assert affine.translate == (0.03, 0.03)
    assert affine.scale == (0.97, 1.03)


def test_normalized_tensor_is_finite() -> None:
    output = build_prompt_transform(False)(_image())
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_jpeg_recompression_keeps_rgb_and_size() -> None:
    random.seed(42)
    source = _image(320, 96)
    output = RandomJPEGCompression(65, 95)(source)
    assert output.mode == "RGB"
    assert output.size == source.size
