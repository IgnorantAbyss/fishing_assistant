"""Audit PromptClassifier manifests/crops and produce bounded contact sheets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.manifest import read_manifest  # noqa: E402
from src.dataset.validation import load_session_split  # noqa: E402
from src.ml.prompt_dataset import CLASS_NAMES, select_prompt_rows  # noqa: E402


DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "fishing_v2"
DEFAULT_MANIFEST = DEFAULT_DATASET_ROOT / "prompt" / "manifest.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "prompt_classifier_v1"
CANONICAL_SPLIT = PROJECT_ROOT / "config" / "session_split.yaml"


def _contact_sheet(
    rows,
    dataset_root: Path,
    destination: Path,
    *,
    maximum: int = 25,
) -> None:
    chosen = rows[:maximum]
    if not chosen:
        return
    cell_width, cell_height = 360, 130
    sheet = np.zeros((cell_height * 5, cell_width * 5, 3), dtype=np.uint8)
    for position, row in enumerate(chosen):
        image = cv2.imread(str(dataset_root / row.crop_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        available_height = 90
        scale = min(cell_width / image.shape[1], available_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        x = (cell_width - resized.shape[1]) // 2
        y = (available_height - resized.shape[0]) // 2
        cell[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        boundary = "Y" if row.is_boundary else "N"
        line1 = f"{row.session_id} #{row.frame_index} {row.label}"
        line2 = f"state={row.original_state} boundary={boundary}"
        cv2.putText(cell, line1, (4, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)
        cv2.putText(cell, line2, (4, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 220, 255), 1)
        row_index, column_index = divmod(position, 5)
        top, left = row_index * cell_height, column_index * cell_width
        sheet[top : top + cell_height, left : left + cell_width] = cell
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 86]):
        raise OSError(f"Could not write contact sheet: {destination}")


def inspect_prompt_dataset(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    create_contact_sheets: bool = True,
) -> dict[str, Any]:
    manifest = Path(manifest_path)
    root = Path(dataset_root)
    destination = Path(report_dir)
    critical: list[str] = []
    warnings: list[str] = []
    try:
        rows = read_manifest(manifest)
    except (OSError, ValueError) as exc:
        rows = []
        critical.append(f"manifest: {exc}")
    if not manifest.is_file():
        critical.append(f"manifest is missing: {manifest}")
    if not rows:
        critical.append("prompt manifest has no rows")

    invalid_labels = sorted({row.label for row in rows} - set(CLASS_NAMES))
    if invalid_labels:
        critical.append(f"invalid labels: {invalid_labels}")
    invalid_splits = sorted({row.split for row in rows} - {"train", "validation", "test"})
    if invalid_splits:
        critical.append(f"invalid splits: {invalid_splits}")

    sessions_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sessions_to_splits[row.session_id].add(row.split)
    leakage = {
        session: sorted(splits)
        for session, splits in sessions_to_splits.items()
        if len(splits) > 1
    }
    if leakage:
        critical.append(f"session split leakage: {leakage}")

    try:
        canonical = load_session_split(CANONICAL_SPLIT, sessions_to_splits)
        trial_ids = {item.session_id for item in canonical.excluded}
        leaked_trials = sorted(trial_ids & set(sessions_to_splits))
        if leaked_trials:
            critical.append(f"trial sessions present in prompt dataset: {leaked_trials}")
        for session_id, split_names in sessions_to_splits.items():
            expected = canonical.split_for(session_id)
            if split_names != {expected}:
                critical.append(
                    f"canonical split mismatch for {session_id}: {sorted(split_names)} != {expected}"
                )
    except (OSError, ValueError) as exc:
        trial_ids = set()
        critical.append(f"canonical session split: {exc}")

    for row in rows:
        if row.split == "train" and row.selected_for_training and row.excluded_reason:
            critical.append(f"selected train row has exclusion reason: {row.crop_path}")
        if row.split in {"validation", "test"} and (
            row.selected_for_training or row.excluded_reason != "evaluation_only"
        ):
            critical.append(f"evaluation row selection is invalid: {row.crop_path}")

    image_errors: list[str] = []
    nonzero_failures: list[str] = []
    for row in rows:
        crop = root / row.crop_path
        image = cv2.imread(str(crop), cv2.IMREAD_COLOR)
        if image is None:
            image_errors.append(row.crop_path)
            continue
        if image.shape[0] <= 0 or image.shape[1] <= 0 or row.width <= 0 or row.height <= 0:
            image_errors.append(f"zero-area:{row.crop_path}")
        if int(image.max()) == 0 or float(image.mean()) <= 0.0:
            nonzero_failures.append(row.crop_path)
    if image_errors:
        critical.append(f"unreadable/zero-area crops ({len(image_errors)}): {image_errors[:10]}")
    if nonzero_failures:
        critical.append(f"all-black crops ({len(nonzero_failures)}): {nonzero_failures[:10]}")

    natural_counts: dict[str, dict[str, int]] = {}
    loader_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        natural = Counter(row.label for row in rows if row.split == split)
        selected = Counter(row.label for row in select_prompt_rows(rows, split))
        natural_counts[split] = {name: natural.get(name, 0) for name in CLASS_NAMES}
        loader_counts[split] = {name: selected.get(name, 0) for name in CLASS_NAMES}
        for label in CLASS_NAMES:
            if selected.get(label, 0) == 0:
                critical.append(f"{split} loader has no {label} samples")

    if create_contact_sheets and not image_errors:
        sheets = destination / "contact_sheets"
        for split in ("train", "validation", "test"):
            selected_rows = select_prompt_rows(rows, split)
            for label in CLASS_NAMES:
                examples = sorted(
                    (row for row in selected_rows if row.label == label),
                    key=lambda row: (row.session_id, row.frame_index),
                )
                _contact_sheet(examples, root, sheets / f"{split}_{label}.jpg")

    report: dict[str, Any] = {
        "valid": not critical,
        "manifest_path": str(manifest),
        "dataset_root": str(root),
        "manifest_rows": len(rows),
        "natural_counts": natural_counts,
        "loader_counts": loader_counts,
        "sessions": {key: sorted(value) for key, value in sorted(sessions_to_splits.items())},
        "trial_session_ids": sorted(trial_ids),
        "critical_errors": critical,
        "warnings": warnings,
        "contact_sheet_max_samples": 25,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "dataset_qa.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# PromptClassifier v1 Dataset QA",
        "",
        f"- Valid: {report['valid']}",
        f"- Manifest rows: {len(rows)}",
        f"- Natural counts: `{natural_counts}`",
        f"- Loader counts: `{loader_counts}`",
        f"- Sessions: `{report['sessions']}`",
        "",
        "## Critical errors",
        "",
        *([f"- {item}" for item in critical] or ["- None"]),
        "",
        "## Warnings",
        "",
        *([f"- {item}" for item in warnings] or ["- None"]),
    ]
    (destination / "dataset_qa.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect PromptClassifier v1 dataset inputs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = inspect_prompt_dataset(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        report_dir=args.report_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
