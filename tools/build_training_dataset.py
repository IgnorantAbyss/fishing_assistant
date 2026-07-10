"""Build balanced prompt and special-state ROI datasets without training models."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_roi_config  # noqa: E402
from src.dataset import DEFAULT_DATASET_CONFIG_PATH, DatasetConfig, load_dataset_config  # noqa: E402
from src.dataset.manifest import ManifestIndex, ManifestRow, read_manifest, write_manifest  # noqa: E402
from src.dataset.roi_exporter import (  # noqa: E402
    PROMPT_LABEL_MAP,
    SPECIAL_LABEL_MAP,
    encode_crop,
    prompt_crop,
    sha256_bytes,
    special_mosaic,
    write_encoded_crop,
)
from src.dataset.sampling import SampledFrame, sample_session_frames  # noqa: E402
from src.dataset.session_scanner import ScannedSession, scan_sessions  # noqa: E402
from src.dataset.validation import (  # noqa: E402
    PROMPT_LABELS,
    SPECIAL_LABELS,
    SessionSplit,
    load_session_split,
    write_session_split,
)
from src.replay_session import DEFAULT_SESSION_ROOT  # noqa: E402


@dataclass(frozen=True)
class DatasetBuildResult:
    output_dir: Path
    report: dict[str, Any]
    dry_run: bool


def _display_size(byte_count: int) -> str:
    return f"{byte_count / (1024 * 1024):.2f} MiB"


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _crop_relative_path(
    dataset_name: str, session_id: str, frame_index: int, label: str, image_format: str
) -> Path:
    return (
        Path(dataset_name)
        / "crops"
        / f"{session_id}_{frame_index:06d}_{label.lower()}.{image_format}"
    )


def _safe_rebuild(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    project = PROJECT_ROOT.resolve()
    if resolved == project or project not in resolved.parents:
        raise ValueError(f"Refusing to rebuild unsafe dataset path: {resolved}")
    for name in ("prompt", "special"):
        target = resolved / name
        if target.exists():
            shutil.rmtree(target)
    for name in ("dataset_report.json", "dataset_report.md"):
        target = resolved / name
        if target.is_file():
            target.unlink()


def _sampled_state_counts(samples: list[SampledFrame]) -> dict[str, int]:
    return dict(sorted(Counter(sample.original_state for sample in samples).items()))


def _missing_labels(counts: dict[str, Counter[str]]) -> dict[str, list[str]]:
    return {
        "prompt": sorted(PROMPT_LABELS - set(counts["prompt"])),
        "special": sorted(SPECIAL_LABELS - set(counts["special"])),
    }


def _write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Fishing Dataset Report",
        "",
        f"- Session count: {report['session_count']}",
        f"- Original frames: {report['original_frame_count']}",
        f"- Original size: {_display_size(report['original_size_bytes'])}",
        f"- Crop dataset size: {_display_size(report['crop_dataset_size_bytes'])}",
        f"- Compression ratio: {report['compression_ratio']:.2f}x",
        f"- Boundary samples: {report['boundary_samples']}",
        f"- Duplicate images: {report['duplicate_images']}",
        f"- Excluded images: {report['excluded_images']}",
        "",
        "## Label counts",
        "",
        f"- Prompt: {report['label_counts']['prompt']}",
        f"- Special: {report['label_counts']['special']}",
        "",
        "## Session split",
        "",
        f"- {report['split_distribution']}",
        "",
        "## Missing classes",
        "",
        f"- {report['missing_classes']}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in report["warnings"]] or ["- None"])
    (output_dir / "dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _process_crop(
    *,
    dataset_name: str,
    image,
    label: str,
    sample: SampledFrame,
    session: ScannedSession,
    source_frame: Path,
    config: DatasetConfig,
    index: ManifestIndex,
    dry_run: bool,
) -> tuple[str, int]:
    encoded = encode_crop(
        image, config.dataset.crop_format, config.dataset.crop_jpg_quality
    )
    digest = sha256_bytes(encoded)
    relative_crop = _crop_relative_path(
        dataset_name,
        session.session_id,
        sample.frame_index,
        label,
        config.dataset.crop_format,
    )
    row = ManifestRow(
        session_id=session.session_id,
        frame_index=sample.frame_index,
        source_frame=_relative_path(source_frame),
        crop_path=relative_crop.as_posix(),
        label=label,
        original_state=sample.original_state,
        is_boundary=sample.is_boundary,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        sha256=digest,
    )
    classification = index.classify(row)
    if classification != "new":
        return classification, 0
    index.add(row)
    if config.dataset.materialize_crops and not dry_run:
        write_encoded_crop(config.dataset.output_dir / relative_crop, encoded)
    return "new", len(encoded)


def build_dataset(
    *,
    config_path: str | Path = DEFAULT_DATASET_CONFIG_PATH,
    session_root: str | Path = DEFAULT_SESSION_ROOT,
    session_id: str | None = None,
    rebuild: bool = False,
    dry_run: bool = False,
    write_reports: bool = True,
    strict_split: bool = True,
) -> DatasetBuildResult:
    config = load_dataset_config(config_path)
    sessions = scan_sessions(session_root, session_id=session_id)
    if not sessions:
        raise ValueError("No validated replay sessions with ground truth were found")
    output_dir = config.dataset.output_dir
    if rebuild and not dry_run:
        _safe_rebuild(output_dir)

    prompt_manifest = output_dir / "prompt" / "manifest.csv"
    special_manifest = output_dir / "special" / "manifest.csv"
    initial_prompt_rows = [] if rebuild else read_manifest(prompt_manifest)
    initial_special_rows = [] if rebuild else read_manifest(special_manifest)
    prompt_index = ManifestIndex(initial_prompt_rows)
    special_index = ManifestIndex(initial_special_rows)
    split_warnings: list[str] = []
    try:
        split = load_session_split(
            output_dir / "session_split.yaml", [s.session_id for s in sessions]
        )
    except ValueError as exc:
        if strict_split:
            raise
        split_warnings.append(str(exc))
        split = SessionSplit((), (), (), tuple(sorted(s.session_id for s in sessions)))
    roi_config = load_roi_config()
    duplicate_images = 0
    excluded_images = 0
    new_crop_bytes = 0
    session_reports: list[dict[str, Any]] = []

    for session in sessions:
        samples = sample_session_frames(
            session.labels,
            config.sampling,
            config.boundary,
            config.dataset.sample_every_n_frames,
        )
        session_reports.append(
            {
                "session_id": session.session_id,
                "frame_count": session.frame_count,
                "resolution": list(session.resolution),
                "image_format": session.image_format,
                "ground_truth_coverage": session.ground_truth_coverage,
                "original_size_bytes": session.source_size_bytes,
                "raw_state_counts": dict(session.state_counts),
                "sampled_state_counts": _sampled_state_counts(samples),
            }
        )
        excluded_images += session.frame_count - len(samples)
        for sample in samples:
            source_frame = session.frames[sample.frame_index - 1]
            frame = cv2.imread(str(source_frame), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Could not read replay frame: {source_frame}")
            for dataset_name, label_map, cropper, index in (
                ("prompt", PROMPT_LABEL_MAP, prompt_crop, prompt_index),
                ("special", SPECIAL_LABEL_MAP, special_mosaic, special_index),
            ):
                label = label_map[sample.original_state]
                if label is None:
                    excluded_images += 1
                    continue
                classification, byte_count = _process_crop(
                    dataset_name=dataset_name,
                    image=cropper(frame, roi_config),
                    label=label,
                    sample=sample,
                    session=session,
                    source_frame=source_frame,
                    config=config,
                    index=index,
                    dry_run=dry_run,
                )
                if classification == "duplicate_crop":
                    duplicate_images += 1
                    excluded_images += 1
                elif classification == "existing_source":
                    excluded_images += 1
                else:
                    new_crop_bytes += byte_count

    if not dry_run:
        write_manifest(prompt_manifest, prompt_index.rows)
        write_manifest(special_manifest, special_index.rows)
        write_session_split(output_dir / "session_split.yaml", split)

    label_counts = {
        "prompt": Counter(row.label for row in prompt_index.rows),
        "special": Counter(row.label for row in special_index.rows),
    }
    existing_crop_bytes = 0
    if not rebuild:
        for row in [*initial_prompt_rows, *initial_special_rows]:
            crop_path = output_dir / row.crop_path
            if crop_path.is_file():
                existing_crop_bytes += crop_path.stat().st_size
    crop_size = existing_crop_bytes + new_crop_bytes if dry_run else sum(
        path.stat().st_size
        for path in (output_dir / "prompt" / "crops", output_dir / "special" / "crops")
        if path.is_dir()
        for path in path.rglob("*")
        if path.is_file()
    )
    original_size = sum(session.source_size_bytes for session in sessions)
    split_sample_distribution = Counter(
        split.split_for(row.session_id) for row in [*prompt_index.rows, *special_index.rows]
    )
    split_session_distribution = Counter(
        split.split_for(session.session_id) for session in sessions
    )
    warnings = list(split_warnings)
    warnings.extend(
        f"{name} has fewer than one assigned session"
        for name in ("train", "validation", "test")
        if not getattr(split, name)
    )
    missing = _missing_labels(label_counts)
    warnings.extend(
        f"{dataset_name} label {label} has no samples"
        for dataset_name, labels in missing.items()
        for label in labels
    )
    report: dict[str, Any] = {
        "session_count": len(sessions),
        "sessions": session_reports,
        "original_frame_count": sum(session.frame_count for session in sessions),
        "original_size_bytes": original_size,
        "crop_dataset_size_bytes": crop_size,
        "compression_ratio": original_size / crop_size if crop_size else 0.0,
        "label_counts": {
            name: dict(sorted(counts.items())) for name, counts in label_counts.items()
        },
        "boundary_samples": sum(
            row.is_boundary for row in [*prompt_index.rows, *special_index.rows]
        ),
        "session_split": split.as_dict(),
        "split_distribution": {
            "sessions": dict(sorted(split_session_distribution.items())),
            "samples": dict(sorted(split_sample_distribution.items())),
        },
        "duplicate_images": duplicate_images,
        "excluded_images": excluded_images,
        "missing_classes": missing,
        "warnings": warnings,
        "dry_run": dry_run,
    }
    if not dry_run and write_reports:
        _write_reports(output_dir, report)
    return DatasetBuildResult(output_dir, report, dry_run)


def print_summary(result: DatasetBuildResult) -> None:
    report = result.report
    print(f"dry_run: {result.dry_run}")
    print(f"sessions_scanned: {report['session_count']}")
    print(f"original_frames: {report['original_frame_count']}")
    print(f"original_size: {_display_size(report['original_size_bytes'])}")
    print(f"estimated_dataset_size: {_display_size(report['crop_dataset_size_bytes'])}")
    print(f"prompt_labels: {report['label_counts']['prompt']}")
    print(f"special_labels: {report['label_counts']['special']}")
    for session in report["sessions"]:
        raw = session["raw_state_counts"].get("WAITING", 0)
        sampled = session["sampled_state_counts"].get("WAITING", 0)
        print(f"waiting_sampling[{session['session_id']}]: {raw} -> {sampled}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced ROI datasets from validated replay sessions.")
    parser.add_argument("--session", help="Build only one session id")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild crops/manifests while preserving split assignments")
    parser.add_argument("--dry-run", action="store_true", help="Plan and encode in memory without writing files")
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG_PATH)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_dataset(
        config_path=args.dataset_config,
        session_root=args.session_root,
        session_id=args.session,
        rebuild=args.rebuild,
        dry_run=args.dry_run,
    )
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
