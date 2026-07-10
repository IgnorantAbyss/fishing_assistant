"""Small live UI-template bank extracted from reviewed replay calibration frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVE_TEMPLATE_ROOT = PROJECT_ROOT / "assets" / "templates" / "live"
ROI_NAMES = ("top_prompt", "center_space", "hook_bar", "press_sequence", "get_window")


@dataclass(frozen=True)
class LiveTemplate:
    state: str
    roi_name: str
    path: Path
    image: np.ndarray


class LiveTemplateBank:
    """Read a small state/ROI template set; no full replay frames are accepted."""

    def __init__(self, root: str | Path = DEFAULT_LIVE_TEMPLATE_ROOT) -> None:
        self.root = Path(root)
        self.templates = self._load()

    def _load(self) -> list[LiveTemplate]:
        templates: list[LiveTemplate] = []
        if not self.root.is_dir():
            return templates
        for state_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            for path in sorted(state_dir.glob("*.png")):
                roi_name = next((name for name in ROI_NAMES if path.stem.endswith(f"_{name}")), None)
                if roi_name is None:
                    continue
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is not None:
                    templates.append(LiveTemplate(state_dir.name.upper(), roi_name, path, image))
        return templates

    def score(self, state: str, roi_name: str, crop: np.ndarray, similarity) -> float | None:
        scores = [similarity(crop, template.image) for template in self.templates if template.state == state and template.roi_name == roi_name]
        return max(scores) if scores else None

    def available(self, state: str, roi_name: str) -> bool:
        return any(template.state == state and template.roi_name == roi_name for template in self.templates)
