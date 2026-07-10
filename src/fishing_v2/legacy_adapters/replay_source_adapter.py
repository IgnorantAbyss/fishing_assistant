from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.fishing_v2.domain.frame_context import FrameContext
from src.replay_ground_truth import load_ground_truth
from src.replay_session import ReplaySession


@dataclass(frozen=True)
class ReplayFrame:
    path: Path
    context: FrameContext
    global_ground_truth: str


class LegacyReplaySourceAdapter:
    """Read-only access; global truth is exposed for reporting, never observation."""

    def __init__(self, session_path: str | Path) -> None:
        self.session = ReplaySession.load(session_path)
        self.paths = tuple(self.session.frame_paths())
        self.global_labels = load_ground_truth(
            self.session.path / "ground_truth.yaml", len(self.paths)
        )
        self.interval = float(self.session.manifest.get("interval_sec", 0.0))

    def frames(self) -> Iterator[ReplayFrame]:
        for frame_index, path in enumerate(self.paths, start=1):
            timestamp = (frame_index - 1) * self.interval
            yield ReplayFrame(
                path,
                FrameContext(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    source_path=path,
                    metadata={"session_id": self.session.path.name},
                ),
                self.global_labels[frame_index],
            )
