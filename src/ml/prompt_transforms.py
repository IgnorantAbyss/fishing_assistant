"""Aspect-preserving prompt transforms shared by training and inference."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as functional


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_INPUT_SIZE = (320, 96)


@dataclass(frozen=True)
class LetterboxResize:
    width: int = DEFAULT_INPUT_SIZE[0]
    height: int = DEFAULT_INPUT_SIZE[1]
    fill: tuple[int, int, int] = (0, 0, 0)

    def __call__(self, image: Image.Image) -> Image.Image:
        source = image.convert("RGB")
        scale = min(self.width / source.width, self.height / source.height)
        resized_width = max(1, round(source.width * scale))
        resized_height = max(1, round(source.height * scale))
        resized = source.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.width, self.height), self.fill)
        offset = ((self.width - resized_width) // 2, (self.height - resized_height) // 2)
        canvas.paste(resized, offset)
        return canvas


class RandomGamma:
    def __init__(self, minimum: float = 0.8, maximum: float = 1.2) -> None:
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, image: Image.Image) -> Image.Image:
        return functional.adjust_gamma(image, random.uniform(self.minimum, self.maximum))


class RandomJPEGCompression:
    def __init__(self, minimum_quality: int = 65, maximum_quality: int = 95) -> None:
        self.minimum_quality = minimum_quality
        self.maximum_quality = maximum_quality

    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=random.randint(self.minimum_quality, self.maximum_quality),
        )
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


class PromptTransform:
    """Callable transform with an inspectable policy for regression tests."""

    def __init__(self, train: bool, width: int = 320, height: int = 96) -> None:
        self.train = train
        operations: list[object] = [LetterboxResize(width, height)]
        if train:
            operations.extend(
                [
                    transforms.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.10,
                        hue=0.02,
                    ),
                    transforms.RandomApply([RandomGamma(0.8, 1.2)], p=0.5),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2
                    ),
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(0.03, 0.03),
                        scale=(0.97, 1.03),
                        fill=0,
                    ),
                    transforms.RandomApply([RandomJPEGCompression(65, 95)], p=0.25),
                ]
            )
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.operations = tuple(operations)
        self.pipeline = transforms.Compose(list(self.operations))

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.pipeline(image)


def build_prompt_transform(
    train: bool, *, width: int = DEFAULT_INPUT_SIZE[0], height: int = DEFAULT_INPUT_SIZE[1]
) -> PromptTransform:
    return PromptTransform(train=train, width=width, height=height)
