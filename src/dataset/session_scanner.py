"""Discover and validate replay sessions eligible for dataset building."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.replay_ground_truth import load_annotations, load_ground_truth
from src.replay_session import DEFAULT_SESSION_ROOT, ReplaySession


class SessionScanError(ValueError):
    """Raised when a candidate replay session is internally inconsistent."""


@dataclass(frozen=True)
class ScannedSession:
    session_id: str
    path: Path
    frames: tuple[Path, ...]
    frame_count: int
    resolution: tuple[int, int]
    image_format: str
    ground_truth_coverage: int
    state_counts: Mapping[str, int]
    labels: Mapping[int, str]
    annotations: Mapping[str, object]
    source_size_bytes: int


@dataclass(frozen=True)
class SessionScanStatus:
    session_id: str
    is_trial: bool
    frames_exists: bool
    manifest_exists: bool
    ground_truth_exists: bool
    frame_count: int
    ground_truth_valid: bool
    eligible: bool
    excluded_reason: str
    session: ScannedSession | None


def scan_session(path: str | Path) -> ScannedSession:
    session_path = Path(path)
    required = (session_path / "frames", session_path / "manifest.json", session_path / "ground_truth.yaml")
    if not all(item.exists() for item in required):
        raise SessionScanError(
            f"Session {session_path.name} requires frames/, manifest.json, and ground_truth.yaml"
        )
    try:
        session = ReplaySession.load(session_path)
        frames = tuple(session.frame_paths())
        manifest_count = int(session.manifest.get("frame_count", -1))
        if manifest_count != len(frames):
            raise SessionScanError(
                f"Session {session_path.name} manifest frame_count={manifest_count}, files={len(frames)}"
            )
        labels = load_ground_truth(session_path / "ground_truth.yaml", manifest_count)
        annotations = load_annotations(session_path / "annotations.yaml", manifest_count)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        if isinstance(exc, SessionScanError):
            raise
        raise SessionScanError(f"Session {session_path.name} validation failed: {exc}") from exc

    raw_resolution = session.manifest.get("screen_size")
    if (
        not isinstance(raw_resolution, list)
        or len(raw_resolution) != 2
        or not all(isinstance(value, int) and value > 0 for value in raw_resolution)
    ):
        raise SessionScanError(f"Session {session_path.name} has invalid screen_size")
    image_format = session.manifest.get("image_format")
    if image_format not in {"jpg", "png"}:
        raise SessionScanError(f"Session {session_path.name} has invalid image_format")
    return ScannedSession(
        session_id=session_path.name,
        path=session_path,
        frames=frames,
        frame_count=manifest_count,
        resolution=(int(raw_resolution[0]), int(raw_resolution[1])),
        image_format=str(image_format),
        ground_truth_coverage=len(labels),
        state_counts=dict(Counter(labels.values())),
        labels=labels,
        annotations=annotations,
        source_size_bytes=sum(frame.stat().st_size for frame in frames),
    )


def scan_sessions(
    root: str | Path = DEFAULT_SESSION_ROOT, *, session_id: str | None = None
) -> list[ScannedSession]:
    root_path = Path(root)
    candidates = sorted(path for path in root_path.glob("session_*") if path.is_dir())
    if session_id is not None:
        candidates = [path for path in candidates if path.name == session_id]
        if not candidates:
            raise SessionScanError(f"Replay session not found: {session_id}")

    scanned: list[ScannedSession] = []
    for path in candidates:
        required = (path / "frames", path / "manifest.json", path / "ground_truth.yaml")
        if not all(item.exists() for item in required):
            if session_id is not None:
                raise SessionScanError(
                    f"Session {path.name} requires frames/, manifest.json, and ground_truth.yaml"
                )
            continue
        scanned.append(scan_session(path))
    return scanned


def scan_session_statuses(
    root: str | Path = DEFAULT_SESSION_ROOT, *, trial_session_ids: set[str] | None = None
) -> list[SessionScanStatus]:
    root_path = Path(root)
    trial_ids = trial_session_ids or set()
    statuses: list[SessionScanStatus] = []
    for path in sorted(item for item in root_path.glob("session_*") if item.is_dir()):
        frames_exists = (path / "frames").is_dir()
        manifest_exists = (path / "manifest.json").is_file()
        ground_truth_exists = (path / "ground_truth.yaml").is_file()
        is_trial = path.name in trial_ids
        if is_trial:
            statuses.append(
                SessionScanStatus(
                    path.name,
                    True,
                    frames_exists,
                    manifest_exists,
                    ground_truth_exists,
                    0,
                    False,
                    False,
                    "trial_session",
                    None,
                )
            )
            continue
        missing = [
            name
            for name, exists in (
                ("frames", frames_exists),
                ("manifest.json", manifest_exists),
                ("ground_truth.yaml", ground_truth_exists),
            )
            if not exists
        ]
        if missing:
            statuses.append(
                SessionScanStatus(
                    path.name,
                    False,
                    frames_exists,
                    manifest_exists,
                    ground_truth_exists,
                    0,
                    False,
                    False,
                    "missing: " + ", ".join(missing),
                    None,
                )
            )
            continue
        try:
            session = scan_session(path)
        except SessionScanError as exc:
            statuses.append(
                SessionScanStatus(
                    path.name,
                    False,
                    frames_exists,
                    manifest_exists,
                    ground_truth_exists,
                    0,
                    False,
                    False,
                    str(exc),
                    None,
                )
            )
            continue
        statuses.append(
            SessionScanStatus(
                path.name,
                False,
                True,
                True,
                True,
                session.frame_count,
                True,
                True,
                "",
                session,
            )
        )
    return statuses
