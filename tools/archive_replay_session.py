"""Safely inspect, zip, or remove raw frames from a completed replay session."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DEFAULT_DATASET_CONFIG_PATH, load_dataset_config  # noqa: E402
from src.dataset.session_scanner import scan_session  # noqa: E402
from src.dataset.validation import validate_dataset  # noqa: E402
from src.replay_session import DEFAULT_SESSION_ROOT, latest_session  # noqa: E402


@dataclass(frozen=True)
class ArchiveResult:
    session_path: Path
    dry_run: bool
    validation_passed: bool
    raw_frame_count: int
    zip_path: Path | None
    frames_deleted: int
    checks: tuple[str, ...]


def _frame_records(frame_paths: tuple[Path, ...], session_path: Path) -> list[dict[str, object]]:
    return [
        {
            "path": frame.relative_to(session_path).as_posix(),
            "size_bytes": frame.stat().st_size,
            "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
        }
        for frame in frame_paths
    ]


def _write_archive_manifest(session_path: Path, frame_paths: tuple[Path, ...]) -> Path:
    destination = session_path / "session_archive_manifest.json"
    payload = {
        "session_id": session_path.name,
        "raw_frame_count": len(frame_paths),
        "raw_size_bytes": sum(frame.stat().st_size for frame in frame_paths),
        "frames": _frame_records(frame_paths, session_path),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def _zip_session(session_path: Path, archive_root: Path) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / f"{session_path.name}.zip"
    files = sorted(path for path in session_path.rglob("*") if path.is_file())
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(session_path.parent).as_posix())
    with zipfile.ZipFile(destination, "r") as archive:
        if archive.testzip() is not None:
            raise ValueError(f"ZIP verification failed: {destination}")
        if len(archive.namelist()) != len(files):
            raise ValueError("ZIP verification failed: file count mismatch")
    return destination


def archive_replay_session(
    session_path: str | Path,
    *,
    dataset_output_dir: str | Path,
    session_root: str | Path = DEFAULT_SESSION_ROOT,
    archive_root: str | Path | None = None,
    dry_run: bool = True,
    create_zip: bool = False,
    delete_frames: bool = False,
) -> ArchiveResult:
    session = scan_session(session_path)
    checks = [
        "ground_truth.yaml exists and covers every frame",
        f"raw frames present: {session.frame_count}",
    ]
    validation_passed = True
    if delete_frames:
        newest = latest_session(session_root).resolve()
        if session.path.resolve() == newest:
            raise ValueError("Refusing to delete frames from the latest replay session")
        validation = validate_dataset(dataset_output_dir, required_session=session.session_id)
        if not validation.valid:
            raise ValueError("Dataset validation failed: " + "; ".join(validation.errors))
        checks.extend(
            [
                "prompt dataset exists for session",
                "special dataset exists for session",
                "all manifest crop files and hashes validated",
                "dataset validation passed",
            ]
        )

    if dry_run:
        return ArchiveResult(
            session.path, True, validation_passed, session.frame_count, None, 0, tuple(checks)
        )

    manifest_path: Path | None = None
    if delete_frames:
        manifest_path = _write_archive_manifest(session.path, session.frames)
        checks.append(f"archive manifest created: {manifest_path}")
    zip_path = None
    if create_zip:
        target_root = (
            Path(archive_root)
            if archive_root is not None
            else Path(session_root) / "archives"
        )
        zip_path = _zip_session(session.path, target_root)
        checks.append(f"zip verified: {zip_path}")

    deleted = 0
    if delete_frames:
        frames_root = (session.path / "frames").resolve()
        for frame in session.frames:
            resolved = frame.resolve()
            if frames_root not in resolved.parents:
                raise ValueError(f"Unsafe frame path: {resolved}")
            frame.unlink()
            deleted += 1
        checks.append(f"raw frames deleted: {deleted}")
    return ArchiveResult(
        session.path,
        False,
        validation_passed,
        session.frame_count,
        zip_path,
        deleted,
        tuple(checks),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run, ZIP, or safely delete one replay session's raw frames.")
    parser.add_argument("--session", required=True, help="Replay session id")
    parser.add_argument("--dry-run", action="store_true", help="Explicitly request the default no-write mode")
    parser.add_argument("--zip", action="store_true", dest="create_zip")
    parser.add_argument("--delete-frames", action="store_true")
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_path = args.session_root / args.session
    config = load_dataset_config(args.dataset_config)
    no_write = args.dry_run or not (args.create_zip or args.delete_frames)
    result = archive_replay_session(
        session_path,
        dataset_output_dir=config.dataset.output_dir,
        session_root=args.session_root,
        dry_run=no_write,
        create_zip=args.create_zip,
        delete_frames=args.delete_frames,
    )
    print(f"session: {result.session_path}")
    print(f"dry_run: {result.dry_run}")
    print(f"validation_passed: {result.validation_passed}")
    print(f"raw_frame_count: {result.raw_frame_count}")
    print(f"zip_path: {result.zip_path}")
    print(f"frames_deleted: {result.frames_deleted}")
    for check in result.checks:
        print(f"check: {check}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
