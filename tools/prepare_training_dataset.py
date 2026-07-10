"""Prepare fixed-split ROI datasets and balance only the training selection."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_roi_config  # noqa: E402
from src.dataset import DEFAULT_DATASET_CONFIG_PATH, DatasetConfig, load_dataset_config  # noqa: E402
from src.dataset.manifest import ManifestConflictError, ManifestRow, write_manifest  # noqa: E402
from src.dataset.roi_exporter import (  # noqa: E402
    PROMPT_LABEL_MAP,
    SPECIAL_LABEL_MAP,
    encode_crop,
    prompt_crop,
    sha256_bytes,
    special_mosaic,
    write_encoded_crop,
)
from src.dataset.sampling import (  # noqa: E402
    BalanceCandidate,
    BalanceSelection,
    boundary_frame_indexes,
    select_balanced_candidates,
)
from src.dataset.session_scanner import (  # noqa: E402
    ScannedSession,
    SessionScanStatus,
    scan_session_statuses,
)
from src.dataset.validation import (  # noqa: E402
    PROMPT_LABELS,
    SPECIAL_LABELS,
    SessionSplit,
    load_session_split,
    validate_session_split,
    write_session_split,
)
from src.replay_ground_truth import IGNORE_STATE  # noqa: E402
from src.replay_session import DEFAULT_SESSION_ROOT  # noqa: E402


DEFAULT_FIXED_SPLIT_PATH = PROJECT_ROOT / "config" / "session_split.yaml"


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _crop_path(
    dataset_name: str, session_id: str, frame_index: int, label: str, image_format: str
) -> Path:
    return (
        Path(dataset_name)
        / "crops"
        / f"{session_id}_{frame_index:06d}_{label.lower()}.{image_format}"
    )


def _fixed_split(path: str | Path, statuses: list[SessionScanStatus]) -> SessionSplit:
    split_path = Path(path)
    all_ids = [status.session_id for status in statuses]
    split = load_session_split(split_path, all_ids)
    validate_session_split(split)
    if split.unassigned:
        raise ValueError(f"Fixed split contains unassigned sessions: {list(split.unassigned)}")
    return split


def _validated_formal_sessions(
    statuses: list[SessionScanStatus], split: SessionSplit
) -> list[ScannedSession]:
    by_id = {status.session_id: status for status in statuses}
    formal_ids = (*split.train, *split.validation, *split.test)
    errors: list[str] = []
    sessions: list[ScannedSession] = []
    for session_id in formal_ids:
        status = by_id.get(session_id)
        if status is None:
            errors.append(f"{session_id}: session directory is missing")
        elif not status.eligible or status.session is None:
            errors.append(f"{session_id}: {status.excluded_reason}")
        else:
            sessions.append(status.session)
    if errors:
        raise ValueError("Formal replay validation failed: " + "; ".join(errors))
    return sessions


def _candidates(
    sessions: list[ScannedSession],
    split: SessionSplit,
    config: DatasetConfig,
    label_map: dict[str, str | None],
) -> list[BalanceCandidate]:
    candidates: list[BalanceCandidate] = []
    for session in sessions:
        boundaries = boundary_frame_indexes(session.labels, config.boundary)
        split_name = split.split_for(session.session_id)
        for frame_index, original_state in sorted(session.labels.items()):
            label = label_map[original_state]
            if label is None:
                continue
            candidates.append(
                BalanceCandidate(
                    session.session_id,
                    frame_index,
                    original_state,
                    label,
                    split_name,
                    frame_index in boundaries,
                )
            )
    return candidates


def _train_counts(candidates: list[BalanceCandidate]) -> Counter[str]:
    return Counter(candidate.label for candidate in candidates if candidate.split == "train")


def _positive_minimum(counts: Counter[str], labels: tuple[str, ...]) -> int:
    positive = [counts[label] for label in labels if counts[label] > 0]
    return min(positive) if positive else 0


def _select_majority(
    candidates: list[BalanceCandidate],
    label: str,
    cap: int,
    group_key: Callable[[BalanceCandidate], object],
) -> BalanceSelection:
    matching = [
        candidate
        for candidate in candidates
        if candidate.split == "train" and candidate.label == label
    ]
    return select_balanced_candidates(matching, cap, group_key=group_key)


def balance_training_candidates(
    prompt_candidates: list[BalanceCandidate], special_candidates: list[BalanceCandidate]
) -> tuple[dict[str, frozenset[tuple[str, int, str]]], dict[str, Any]]:
    prompt_before = _train_counts(prompt_candidates)
    prompt_minority = _positive_minimum(
        prompt_before, ("IDLE", "WAITING", "READY", "NONE")
    )
    prompt_cap = max(200, prompt_minority * 2)
    prompt_waiting = _select_majority(
        prompt_candidates, "WAITING", prompt_cap, lambda candidate: candidate.session_id
    )
    prompt_none = _select_majority(
        prompt_candidates, "NONE", prompt_cap, lambda candidate: candidate.session_id
    )

    special_before = _train_counts(special_candidates)
    special_rare_min = _positive_minimum(special_before, ("HOOK", "PRESS", "GET"))
    special_none_cap = max(250, special_rare_min * 3)
    special_none = _select_majority(
        special_candidates,
        "NONE",
        special_none_cap,
        lambda candidate: (candidate.original_state, candidate.session_id),
    )

    prompt_selected = {
        candidate.key
        for candidate in prompt_candidates
        if candidate.split == "train" and candidate.label in {"IDLE", "READY"}
    }
    prompt_selected.update(prompt_waiting.selected_keys)
    prompt_selected.update(prompt_none.selected_keys)
    special_selected = {
        candidate.key
        for candidate in special_candidates
        if candidate.split == "train" and candidate.label in {"HOOK", "PRESS", "GET"}
    }
    special_selected.update(special_none.selected_keys)
    selections = {
        "prompt": frozenset(prompt_selected),
        "special": frozenset(special_selected),
    }
    details = {
        "prompt_minority_count": prompt_minority,
        "prompt_majority_cap": prompt_cap,
        "special_rare_min": special_rare_min,
        "special_none_cap": special_none_cap,
        "boundary_overflow": {
            "prompt_waiting": prompt_waiting.boundary_overflow,
            "prompt_none": prompt_none.boundary_overflow,
            "special_none": special_none.boundary_overflow,
        },
    }
    return selections, details


def _selection_fields(
    candidate: BalanceCandidate, selected_keys: frozenset[tuple[str, int, str]]
) -> tuple[bool, str]:
    if candidate.split in {"validation", "test"}:
        return False, "evaluation_only"
    if candidate.key in selected_keys:
        return True, ""
    return False, "majority_downsample"


def _label_counts(
    candidates: list[BalanceCandidate],
    *,
    split_name: str,
    selected_keys: frozenset[tuple[str, int, str]] | None = None,
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                candidate.label
                for candidate in candidates
                if candidate.split == split_name
                and (selected_keys is None or candidate.key in selected_keys)
            ).items()
        )
    )


def _missing_classes(
    prompt_candidates: list[BalanceCandidate], special_candidates: list[BalanceCandidate]
) -> list[str]:
    warnings: list[str] = []
    for split_name in ("train", "validation", "test"):
        prompt_present = {
            candidate.label for candidate in prompt_candidates if candidate.split == split_name
        }
        special_present = {
            candidate.label for candidate in special_candidates if candidate.split == split_name
        }
        warnings.extend(
            f"{split_name} prompt label {label} has no samples"
            for label in sorted(PROMPT_LABELS - prompt_present)
        )
        warnings.extend(
            f"{split_name} special label {label} has no samples"
            for label in sorted(SPECIAL_LABELS - special_present)
        )
    return warnings


def _status_dict(status: SessionScanStatus) -> dict[str, object]:
    return {
        "session_id": status.session_id,
        "is_trial": status.is_trial,
        "frames_exists": status.frames_exists,
        "manifest_exists": status.manifest_exists,
        "ground_truth_exists": status.ground_truth_exists,
        "frame_count": status.frame_count,
        "ground_truth_valid": status.ground_truth_valid,
        "eligible": status.eligible,
        "excluded_reason": status.excluded_reason,
    }


def _write_training_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Training Dataset Report",
        "",
        f"- Replay directories: {report['replay_directory_count']}",
        f"- Trial sessions: {report['trial_sessions']}",
        f"- Formal sessions: {report['formal_session_count']}",
        f"- Original frames: {report['original_frame_count']}",
        f"- Train: {report['split']['train']}",
        f"- Validation: {report['split']['validation']}",
        f"- Test: {report['split']['test']}",
        f"- Session leakage: {report['session_leakage']}",
        f"- Original size: {report['original_size_bytes'] / (1024 * 1024):.2f} MiB",
        f"- Crop size: {report['dataset_crop_size_bytes'] / (1024 * 1024):.2f} MiB",
        "",
        "## Train balance",
        "",
        f"- Prompt before: {report['prompt_train_before']}",
        f"- Prompt after: {report['prompt_train_after']}",
        f"- Special before: {report['special_train_before']}",
        f"- Special after: {report['special_train_after']}",
        "",
        "## Evaluation distribution",
        "",
        f"- Validation prompt: {report['validation_prompt_labels']}",
        f"- Test prompt: {report['test_prompt_labels']}",
        f"- Validation special: {report['validation_special_labels']}",
        f"- Test special: {report['test_special_labels']}",
        "",
        "## Exclusions",
        "",
        f"- WAITING: {report['excluded_waiting']}",
        f"- Prompt NONE: {report['excluded_prompt_none']}",
        f"- Special NONE: {report['excluded_special_none']}",
        f"- IGNORE frames: {report['ignore_excluded']}",
        f"- Boundary retained: {report['boundary_retained']}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in report["warnings"]] or ["- None"])
    (output_dir / "training_dataset_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def prepare_training_dataset(
    *,
    seed: int,
    dry_run: bool,
    rebuild: bool,
    config_path: str | Path = DEFAULT_DATASET_CONFIG_PATH,
    split_path: str | Path = DEFAULT_FIXED_SPLIT_PATH,
    session_root: str | Path = DEFAULT_SESSION_ROOT,
) -> dict[str, Any]:
    if dry_run == rebuild:
        raise ValueError("Choose exactly one of dry_run or rebuild")
    config = load_dataset_config(config_path)
    split_probe = load_session_split(split_path, [])
    trial_ids = {item.session_id for item in split_probe.excluded}
    statuses = scan_session_statuses(session_root, trial_session_ids=trial_ids)
    split = _fixed_split(split_path, statuses)
    sessions = _validated_formal_sessions(statuses, split)
    if len(sessions) != 7:
        raise ValueError(f"Expected 7 formal sessions, validated {len(sessions)}")

    prompt_candidates = _candidates(sessions, split, config, PROMPT_LABEL_MAP)
    special_candidates = _candidates(sessions, split, config, SPECIAL_LABEL_MAP)
    selections, balance = balance_training_candidates(prompt_candidates, special_candidates)
    candidates_by_dataset = {
        "prompt": prompt_candidates,
        "special": special_candidates,
    }
    candidate_lookup = {
        name: {(candidate.session_id, candidate.frame_index): candidate for candidate in candidates}
        for name, candidates in candidates_by_dataset.items()
    }
    rows: dict[str, list[ManifestRow]] = {"prompt": [], "special": []}
    hash_registry: dict[str, dict[str, tuple[str, Path]]] = {"prompt": {}, "special": {}}
    crop_bytes = 0
    duplicate_images = 0
    roi_config = load_roi_config()

    for session in sorted(sessions, key=lambda item: item.session_id):
        for frame_index, original_state in sorted(session.labels.items()):
            if original_state == IGNORE_STATE:
                continue
            frame_path = session.frames[frame_index - 1]
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Could not read replay frame: {frame_path}")
            for dataset_name, cropper in (
                ("prompt", prompt_crop),
                ("special", special_mosaic),
            ):
                candidate = candidate_lookup[dataset_name][(session.session_id, frame_index)]
                crop = cropper(frame, roi_config)
                encoded = encode_crop(
                    crop, config.dataset.crop_format, config.dataset.crop_jpg_quality
                )
                digest = sha256_bytes(encoded)
                registry = hash_registry[dataset_name]
                existing = registry.get(digest)
                if existing is not None and existing[0] != candidate.label:
                    raise ManifestConflictError(
                        f"{dataset_name} crop {digest} has labels {existing[0]} and {candidate.label}"
                    )
                if existing is None:
                    relative_crop = _crop_path(
                        dataset_name,
                        session.session_id,
                        frame_index,
                        candidate.label,
                        config.dataset.crop_format,
                    )
                    registry[digest] = (candidate.label, relative_crop)
                    crop_bytes += len(encoded)
                    if rebuild and config.dataset.materialize_crops:
                        write_encoded_crop(config.dataset.output_dir / relative_crop, encoded)
                else:
                    relative_crop = existing[1]
                    duplicate_images += 1
                selected, excluded_reason = _selection_fields(
                    candidate, selections[dataset_name]
                )
                rows[dataset_name].append(
                    ManifestRow(
                        session_id=session.session_id,
                        frame_index=frame_index,
                        source_frame=_relative(frame_path),
                        crop_path=relative_crop.as_posix(),
                        label=candidate.label,
                        original_state=original_state,
                        split=candidate.split,
                        selected_for_training=selected,
                        excluded_reason=excluded_reason,
                        is_boundary=candidate.is_boundary,
                        width=int(crop.shape[1]),
                        height=int(crop.shape[0]),
                        sha256=digest,
                    )
                )

    if rebuild:
        write_manifest(config.dataset.output_dir / "prompt" / "manifest.csv", rows["prompt"])
        write_manifest(config.dataset.output_dir / "special" / "manifest.csv", rows["special"])
        write_session_split(config.dataset.output_dir / "session_split.yaml", split)

    prompt_train_before = _label_counts(prompt_candidates, split_name="train")
    prompt_train_after = _label_counts(
        prompt_candidates, split_name="train", selected_keys=selections["prompt"]
    )
    special_train_before = _label_counts(special_candidates, split_name="train")
    special_train_after = _label_counts(
        special_candidates, split_name="train", selected_keys=selections["special"]
    )
    boundary_train = [
        candidate
        for candidates in candidates_by_dataset.values()
        for candidate in candidates
        if candidate.split == "train" and candidate.is_boundary
    ]
    boundary_retained = sum(
        candidate.key in selections[dataset_name]
        for dataset_name, candidates in candidates_by_dataset.items()
        for candidate in candidates
        if candidate.split == "train" and candidate.is_boundary
    )
    warnings = _missing_classes(prompt_candidates, special_candidates)
    warnings.extend(
        f"{name} boundary count exceeded cap by {overflow}; cap expanded"
        for name, overflow in balance["boundary_overflow"].items()
        if overflow
    )
    excluded_waiting = prompt_train_before.get("WAITING", 0) - prompt_train_after.get("WAITING", 0)
    excluded_prompt_none = prompt_train_before.get("NONE", 0) - prompt_train_after.get("NONE", 0)
    excluded_special_none = special_train_before.get("NONE", 0) - special_train_after.get("NONE", 0)
    original_size = sum(session.source_size_bytes for session in sessions)
    report: dict[str, Any] = {
        "seed": seed,
        "replay_directory_count": len(statuses),
        "session_scan": [_status_dict(status) for status in statuses],
        "trial_sessions": [item.session_id for item in split.excluded],
        "formal_session_count": len(sessions),
        "original_frame_count": sum(session.frame_count for session in sessions),
        "invalid_sessions": [
            _status_dict(status)
            for status in statuses
            if not status.is_trial and not status.eligible
        ],
        "split": {
            "train": list(split.train),
            "validation": list(split.validation),
            "test": list(split.test),
        },
        "session_state_counts": {
            session.session_id: dict(session.state_counts) for session in sessions
        },
        "prompt_train_before": prompt_train_before,
        "prompt_train_after": prompt_train_after,
        "special_train_before": special_train_before,
        "special_train_after": special_train_after,
        "validation_prompt_labels": _label_counts(prompt_candidates, split_name="validation"),
        "test_prompt_labels": _label_counts(prompt_candidates, split_name="test"),
        "validation_special_labels": _label_counts(special_candidates, split_name="validation"),
        "test_special_labels": _label_counts(special_candidates, split_name="test"),
        "excluded_waiting": excluded_waiting,
        "excluded_prompt_none": excluded_prompt_none,
        "excluded_special_none": excluded_special_none,
        "boundary_train": len(boundary_train),
        "boundary_retained": boundary_retained,
        "ignore_excluded": sum(session.state_counts.get(IGNORE_STATE, 0) for session in sessions),
        "original_size_bytes": original_size,
        "dataset_crop_size_bytes": crop_bytes,
        "compression_ratio": original_size / crop_bytes if crop_bytes else 0.0,
        "duplicate_images": duplicate_images,
        "session_leakage": False,
        "ground_truth_coverage_valid": all(
            session.ground_truth_coverage == session.frame_count for session in sessions
        ),
        "balance": balance,
        "warnings": warnings,
        "dry_run": dry_run,
    }
    if rebuild:
        _write_training_report(config.dataset.output_dir, report)
    return report


def print_report(report: dict[str, Any]) -> None:
    print(f"dry_run: {report['dry_run']}")
    print(f"replay_directories: {report['replay_directory_count']}")
    print(f"trial_sessions: {report['trial_sessions']}")
    print(f"formal_sessions: {report['formal_session_count']}")
    print(f"split: {report['split']}")
    print(f"session_leakage: {report['session_leakage']}")
    print(f"prompt_train_before: {report['prompt_train_before']}")
    print(f"prompt_train_after: {report['prompt_train_after']}")
    print(f"special_train_before: {report['special_train_before']}")
    print(f"special_train_after: {report['special_train_after']}")
    print(f"validation_prompt: {report['validation_prompt_labels']}")
    print(f"test_prompt: {report['test_prompt_labels']}")
    print(f"validation_special: {report['validation_special_labels']}")
    print(f"test_special: {report['test_special_labels']}")
    print(f"excluded_waiting: {report['excluded_waiting']}")
    print(f"excluded_prompt_none: {report['excluded_prompt_none']}")
    print(f"excluded_special_none: {report['excluded_special_none']}")
    print(f"boundary_retained: {report['boundary_retained']}/{report['boundary_train']}")
    print(f"original_size_mib: {report['original_size_bytes'] / (1024 * 1024):.2f}")
    print(f"dataset_size_mib: {report['dataset_crop_size_bytes'] / (1024 * 1024):.2f}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the fixed 4/1/2 replay dataset split without model training.")
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--rebuild", action="store_true")
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG_PATH)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_FIXED_SPLIT_PATH)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = prepare_training_dataset(
        seed=args.seed,
        dry_run=args.dry_run,
        rebuild=args.rebuild,
        config_path=args.dataset_config,
        split_path=args.split_config,
        session_root=args.session_root,
    )
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
