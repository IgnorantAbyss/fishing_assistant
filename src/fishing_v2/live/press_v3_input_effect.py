"""Episode-local PRESS V3 input-effect tracking.

The detector remains stateless.  This tracker owns the immutable clean-strip
baseline and a monotonic *visual* input-effect latch for one physical PRESS
episode.  Its input_started/post_input_frame fields are backward-compatible
visual inference telemetry only; they are not authoritative Runtime emission
state and must not gate Production sequence qualification.  It compares
appearance, never key labels, so glyph classification cannot feed back into
the visual effect signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PressV3InputEffectConfig:
    slot_count: int = 10
    pixel_delta_e: float = 18.0
    changed_pixel_ratio: float = 0.16
    largest_component_ratio: float = 0.08
    delta_e_p90: float = 45.0
    trailing_stable_ratio: float = 0.14
    trailing_stable_fraction: float = 0.60


class PressV3InputEffectTracker:
    """Report visual strip changes without claiming Runtime input ownership."""

    def __init__(
        self, config: PressV3InputEffectConfig | None = None
    ) -> None:
        self.config = config or PressV3InputEffectConfig()
        self.reset()

    def reset(self) -> None:
        self._baseline: np.ndarray | None = None
        self._baseline_frame_index: int | None = None
        self._baseline_timestamp: float | None = None
        self._input_started = False
        self._input_started_frame_index: int | None = None

    @property
    def baseline_frame_index(self) -> int | None:
        return self._baseline_frame_index

    @property
    def input_started(self) -> bool:
        return self._input_started

    @staticmethod
    def _component_metrics(mask: np.ndarray) -> tuple[float, list[int] | None]:
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if count <= 1:
            return 0.0, None
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, width, height, area = (
            int(value) for value in stats[label]
        )
        return (
            float(area) / max(1, mask.size),
            [x, y, x + width, y + height],
        )

    def _slot_deltas(
        self, current: np.ndarray
    ) -> list[dict[str, Any]]:
        assert self._baseline is not None
        baseline = cv2.resize(
            self._baseline,
            (current.shape[1], current.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        current_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(
            np.float32
        )
        baseline_lab = cv2.cvtColor(baseline, cv2.COLOR_BGR2LAB).astype(
            np.float32
        )
        records: list[dict[str, Any]] = []
        for index in range(self.config.slot_count):
            x1 = round(index * current.shape[1] / self.config.slot_count)
            x2 = round((index + 1) * current.shape[1] / self.config.slot_count)
            delta = np.linalg.norm(
                current_lab[:, x1:x2] - baseline_lab[:, x1:x2], axis=2
            )
            changed = delta >= self.config.pixel_delta_e
            largest_ratio, largest_bbox = self._component_metrics(changed)
            changed_ratio = float(np.mean(changed))
            p90 = float(np.quantile(delta, 0.90))
            diffuse = bool(
                changed_ratio >= self.config.changed_pixel_ratio
                and largest_ratio >= self.config.largest_component_ratio
                and p90 >= self.config.delta_e_p90
            )
            records.append({
                "slot_index": index,
                "mean_delta_e": round(float(np.mean(delta)), 4),
                "p90_delta_e": round(p90, 4),
                "changed_pixel_ratio": round(changed_ratio, 4),
                "largest_delta_component_ratio": round(largest_ratio, 4),
                "largest_delta_component_bbox": largest_bbox,
                "diffuse_halo_or_flash": diffuse,
            })
        return records

    def evaluate(
        self,
        result: dict[str, Any],
        *,
        frame_index: int,
        source_capture_timestamp: float,
        physical_key_activity: tuple[dict[str, Any], ...] = (),
    ) -> dict[str, Any]:
        """Return a result with episode-local clean/input semantics attached."""
        updated = dict(result)
        structurally_complete = bool(result.get("frame_complete", False))
        frame_local_effect = bool(
            result.get(
                "frame_local_input_effect_detected",
                result.get("input_effect_detected", False),
            )
        )
        strip = result.get("key_strip_crop")
        slot_deltas: list[dict[str, Any]] = []
        temporal_effect = False
        temporal_reason: str | None = None
        if (
            self._baseline is not None
            and isinstance(strip, np.ndarray)
            and strip.size > 0
        ):
            slot_deltas = self._slot_deltas(strip)
            changed = [
                int(item["slot_index"])
                for item in slot_deltas
                if item["diffuse_halo_or_flash"]
            ]
            if changed:
                last_changed = max(changed)
                trailing = slot_deltas[last_changed + 1:]
                stable_trailing = [
                    item for item in trailing
                    if float(item["changed_pixel_ratio"])
                    < self.config.trailing_stable_ratio
                ]
                stable_fraction = (
                    len(stable_trailing) / len(trailing) if trailing else 1.0
                )
                leading_pattern = min(changed) <= 1
                temporal_effect = bool(
                    leading_pattern
                    and stable_fraction >= self.config.trailing_stable_fraction
                )
                if temporal_effect:
                    temporal_reason = (
                        "episode_baseline_leading_slot_diffuse_delta:"
                        + ",".join(str(item) for item in changed)
                    )

        direct_effect = bool(frame_local_effect or temporal_effect)
        if direct_effect and not self._input_started:
            self._input_started = True
            self._input_started_frame_index = int(frame_index)

        if (
            self._baseline is None
            and structurally_complete
            and not direct_effect
            and isinstance(strip, np.ndarray)
            and strip.size > 0
        ):
            self._baseline = strip.copy()
            self._baseline_frame_index = int(frame_index)
            self._baseline_timestamp = float(source_capture_timestamp)

        clean_eligible = bool(
            structurally_complete and not self._input_started
        )
        reason = (
            temporal_reason
            or (
                str(result.get("input_effect_reason"))
                if frame_local_effect else None
            )
            or ("input_started_latched" if self._input_started else "none")
        )
        updated.update({
            "frame_structurally_complete": structurally_complete,
            "frame_clean_eligible": clean_eligible,
            "clean_frame_eligible": clean_eligible,
            "episode_input_started": self._input_started,
            "post_input_frame": self._input_started,
            "input_effect_detected": direct_effect,
            "input_effect_reason": reason,
            "input_started_latched": self._input_started,
            "input_started_frame_index": self._input_started_frame_index,
            "input_effect_baseline_frame_index": self._baseline_frame_index,
            "input_effect_baseline_capture_timestamp": self._baseline_timestamp,
            "per_slot_input_effect_deltas": slot_deltas,
            "per_slot_halo_flash_metrics": slot_deltas,
            "physical_key_activity_correlation": [
                dict(item) for item in physical_key_activity
            ],
            "panel_phase": (
                "PANEL_INPUT_STARTED"
                if self._input_started else
                "PANEL_CLEAN"
                if structurally_complete else
                str(result.get("panel_phase", "PANEL_APPEARING"))
            ),
            "input_effect_evidence": {
                "frame_local_effect": frame_local_effect,
                "temporal_baseline_effect": temporal_effect,
                "baseline_frame_index": self._baseline_frame_index,
                "input_started_frame_index": self._input_started_frame_index,
                "reason": reason,
                "per_slot_deltas": slot_deltas,
            },
        })
        return updated
