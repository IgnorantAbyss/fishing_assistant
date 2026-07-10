"""Export only configured ROI crops for future model training."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from src.config_loader import ROIConfig, normalized_to_pixel_roi


PROMPT_LABEL_MAP: dict[str, str | None] = {
    "IDLE": "IDLE",
    "WAITING": "WAITING",
    "READY": "READY",
    "HOOK": "NONE",
    "PRESS": "NONE",
    "GET": "NONE",
    "IGNORE": None,
}

SPECIAL_LABEL_MAP: dict[str, str | None] = {
    "IDLE": "NONE",
    "WAITING": "NONE",
    "READY": "NONE",
    "HOOK": "HOOK",
    "PRESS": "PRESS",
    "GET": "GET",
    "IGNORE": None,
}

SPECIAL_CELL_SIZE = (320, 192)
SPECIAL_MOSAIC_SIZE = (SPECIAL_CELL_SIZE[0] * 3, SPECIAL_CELL_SIZE[1])


def _crop(frame: np.ndarray, roi_config: ROIConfig, name: str) -> np.ndarray:
    left, top, right, bottom = normalized_to_pixel_roi(
        roi_config.rois[name], frame.shape[1], frame.shape[0]
    )
    return frame[top:bottom, left:right]


def prompt_crop(frame: np.ndarray, roi_config: ROIConfig) -> np.ndarray:
    return _crop(frame, roi_config, "top_prompt")


def _fit_cell(image: np.ndarray) -> np.ndarray:
    cell_width, cell_height = SPECIAL_CELL_SIZE
    scale = min(cell_width / image.shape[1], cell_height / image.shape[0])
    resized_width = max(1, round(image.shape[1] * scale))
    resized_height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
    x = (cell_width - resized_width) // 2
    y = (cell_height - resized_height) // 2
    cell[y : y + resized_height, x : x + resized_width] = resized
    return cell


def special_mosaic(frame: np.ndarray, roi_config: ROIConfig) -> np.ndarray:
    hook_roi = "hook_bar_precise" if "hook_bar_precise" in roi_config.rois else "hook_bar"
    panels = (
        _fit_cell(_crop(frame, roi_config, hook_roi)),
        _fit_cell(_crop(frame, roi_config, "press_sequence")),
        _fit_cell(_crop(frame, roi_config, "get_window")),
    )
    mosaic = cv2.hconcat(panels)
    if (mosaic.shape[1], mosaic.shape[0]) != SPECIAL_MOSAIC_SIZE:
        raise ValueError("Special-state mosaic dimensions are not fixed")
    return mosaic


def encode_crop(image: np.ndarray, image_format: str, jpg_quality: int) -> bytes:
    suffix = ".jpg" if image_format == "jpg" else ".png"
    parameters = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality] if image_format == "jpg" else []
    ok, encoded = cv2.imencode(suffix, image, parameters)
    if not ok:
        raise OSError(f"Could not encode dataset crop as {image_format}")
    return encoded.tobytes()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_encoded_crop(path: str | Path, data: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
