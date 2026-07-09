"""Replay-session storage for capture-only and offline detector workflows."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config_loader import ReplayRetentionSettings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_ROOT = PROJECT_ROOT / "assets" / "replay" / "sessions"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Replay manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid replay manifest: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Replay manifest must contain an object: {path}")
    return data


@dataclass
class ReplaySession:
    """A session directory plus the manifest updated whenever a frame is saved."""

    path: Path
    manifest: dict[str, Any]

    @property
    def frames_dir(self) -> Path:
        return self.path / "frames"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @classmethod
    def create(
        cls,
        root: str | Path = DEFAULT_SESSION_ROOT,
        *,
        interval_sec: float,
        duration_sec: float,
        image_format: str,
        screen_size: tuple[int, int],
        monitor_index: int,
        notes: str = "",
    ) -> "ReplaySession":
        if image_format not in {"jpg", "png"}:
            raise ValueError("image_format must be 'jpg' or 'png'")
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        timestamp = _utc_now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp}"
        path = root_path / session_id
        suffix = 2
        while path.exists():
            path = root_path / f"{session_id}_{suffix:02d}"
            suffix += 1
        frames_dir = path / "frames"
        frames_dir.mkdir(parents=True)
        manifest: dict[str, Any] = {
            "session_id": path.name,
            "created_at": _utc_now().isoformat(),
            "interval_sec": float(interval_sec),
            "duration_sec": float(duration_sec),
            "image_format": image_format,
            "frame_count": 0,
            "screen_size": [int(screen_size[0]), int(screen_size[1])],
            "monitor_index": int(monitor_index),
            "notes": notes,
            "markers": [],
        }
        session = cls(path=path, manifest=manifest)
        session.write_manifest()
        return session

    @classmethod
    def load(cls, path: str | Path) -> "ReplaySession":
        session_path = Path(path)
        manifest = _read_manifest(session_path / "manifest.json")
        frames_dir = session_path / "frames"
        if not frames_dir.is_dir():
            raise FileNotFoundError(f"Replay frames directory not found: {frames_dir}")
        return cls(path=session_path, manifest=manifest)

    def write_manifest(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def save_frame(self, frame: np.ndarray, *, jpg_quality: int = 92) -> Path:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with three channels")
        image_format = str(self.manifest["image_format"])
        frame_index = int(self.manifest["frame_count"]) + 1
        destination = self.frames_dir / f"{frame_index:06d}.{image_format}"
        parameters = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality] if image_format == "jpg" else []
        if not cv2.imwrite(str(destination), frame, parameters):
            raise OSError(f"Could not write replay frame: {destination}")
        self.manifest["frame_count"] = frame_index
        self.write_manifest()
        return destination

    def add_marker(self, label: str, timestamp_sec: float) -> None:
        markers = self.manifest.setdefault("markers", [])
        if not isinstance(markers, list):
            raise ValueError("Replay manifest markers must be a list")
        markers.append({"label": label, "timestamp_sec": float(timestamp_sec)})
        self.write_manifest()

    def frame_paths(self) -> list[Path]:
        image_format = str(self.manifest.get("image_format", "jpg"))
        return sorted(self.frames_dir.glob(f"*.{image_format}"))

    def size_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.path.rglob("*") if path.is_file())


def latest_session(root: str | Path = DEFAULT_SESSION_ROOT) -> Path:
    root_path = Path(root)
    sessions = [path for path in root_path.glob("session_*") if path.is_dir()]
    if not sessions:
        raise FileNotFoundError(f"No replay sessions found in {root_path}")
    return max(sessions, key=lambda path: path.stat().st_mtime)


def _session_created_at(path: Path) -> datetime:
    manifest_path = path / "manifest.json"
    try:
        created_at = _read_manifest(manifest_path).get("created_at")
        if isinstance(created_at, str):
            return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (ValueError, FileNotFoundError):
        pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _session_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def apply_retention(
    root: str | Path,
    settings: ReplayRetentionSettings,
    *,
    keep_session: str | Path | None = None,
) -> list[Path]:
    """Delete oldest completed sessions without ever deleting ``keep_session``."""
    root_path = Path(root)
    if not root_path.exists():
        return []
    keep_path = Path(keep_session).resolve() if keep_session is not None else None
    sessions = sorted((path for path in root_path.glob("session_*") if path.is_dir()), key=_session_created_at)
    deleted: list[Path] = []

    def remove(path: Path) -> None:
        if keep_path is not None and path.resolve() == keep_path:
            return
        shutil.rmtree(path)
        deleted.append(path)

    expires_before = _utc_now() - timedelta(days=settings.max_age_days)
    for path in list(sessions):
        if _session_created_at(path) < expires_before and (keep_path is None or path.resolve() != keep_path):
            remove(path)
            sessions.remove(path)

    while len(sessions) > settings.max_sessions:
        candidate = next((path for path in sessions if keep_path is None or path.resolve() != keep_path), None)
        if candidate is None:
            break
        remove(candidate)
        sessions.remove(candidate)

    max_bytes = settings.max_total_size_mb * 1024 * 1024
    while sessions and sum(_session_size(path) for path in sessions) > max_bytes:
        candidate = next((path for path in sessions if keep_path is None or path.resolve() != keep_path), None)
        if candidate is None:
            break
        remove(candidate)
        sessions.remove(candidate)
    return deleted
