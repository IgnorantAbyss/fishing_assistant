"""Event-driven debug-image persistence with bounded local retention."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from src.config_loader import DebugSettings


class DebugSaver:
    """Save debug frames only for meaningful events, never every observation."""

    def __init__(self, output_dir: str | Path, settings: DebugSettings) -> None:
        self.output_dir = Path(output_dir)
        self.settings = settings
        self._last_signature: tuple[str, str] | None = None
        self._sequence = 0

    def _reason(
        self,
        *,
        state: str,
        previous_state: str | None,
        low_confidence: bool,
        exception: Exception | str | None,
        save_all: bool,
    ) -> str | None:
        if not self.settings.save_debug_images:
            return None
        if save_all:
            return "manual"
        if exception is not None:
            return "exception"
        if state == "UNKNOWN" and self.settings.save_on_unknown:
            return "unknown"
        if low_confidence and self.settings.save_on_low_confidence:
            return "low_confidence"
        if previous_state is not None and previous_state != state and self.settings.save_on_state_change:
            return "state_change"
        return None

    def save_if_needed(
        self,
        frame: np.ndarray,
        *,
        state: str,
        previous_state: str | None = None,
        low_confidence: bool = False,
        exception: Exception | str | None = None,
        save_all: bool = False,
    ) -> Path | None:
        """Save one frame for an event, suppressing identical consecutive events."""
        reason = self._reason(
            state=state,
            previous_state=previous_state,
            low_confidence=low_confidence,
            exception=exception,
            save_all=save_all,
        )
        if reason is None:
            self._last_signature = None
            return None
        signature = (state, reason)
        if signature == self._last_signature and not save_all:
            return None
        self._last_signature = signature

        self.output_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        self._sequence += 1
        filename = f"{now:%Y%m%dT%H%M%S%fZ}_{self._sequence:04d}_{state.lower()}_{reason}.png"
        destination = self.output_dir / filename
        if not cv2.imwrite(str(destination), frame):
            raise OSError(f"Could not save debug image: {destination}")
        self.enforce_retention(now=now)
        return destination if destination.exists() else None

    def enforce_retention(self, *, now: datetime | None = None) -> None:
        """Remove expired files first, then enforce count and aggregate-size limits."""
        if not self.output_dir.exists():
            return
        current_time = now or datetime.now(timezone.utc)
        files = sorted(
            (path for path in self.output_dir.iterdir() if path.is_file()),
            key=lambda path: path.stat().st_mtime,
        )
        if self.settings.max_debug_age_days >= 0:
            expires_before = current_time - timedelta(days=self.settings.max_debug_age_days)
            retained: list[Path] = []
            for path in files:
                modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if modified < expires_before:
                    path.unlink()
                else:
                    retained.append(path)
            files = retained

        while len(files) > self.settings.max_debug_images:
            files.pop(0).unlink()

        max_bytes = self.settings.max_debug_size_mb * 1024 * 1024
        total_bytes = sum(path.stat().st_size for path in files)
        while files and total_bytes > max_bytes:
            oldest = files.pop(0)
            total_bytes -= oldest.stat().st_size
            oldest.unlink()
