"""Materialize the seven-session fixed-ROI Prompt Observation v1 dataset."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fishing_v2.data.prompt_observation_dataset import (  # noqa: E402
    build_prompt_observation_dataset,
    resolve_roi_path,
)


def _contact_sheets(rows, dataset_root: Path, output_root: Path) -> None:
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    for label, candidates in sorted(by_label.items()):
        indices = np.linspace(0, len(candidates) - 1, min(8, len(candidates)), dtype=int)
        tiles = []
        for index in indices:
            row = candidates[int(index)]
            image = cv2.imread(str(resolve_roi_path(row, dataset_root)), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(resolve_roi_path(row, dataset_root))
            tile = image.copy()
            cv2.putText(
                tile,
                f"{row['session_id'][-6:]}:{int(row['frame_index']):06d}",
                (4, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)
        sheet = np.vstack(tiles)
        if not cv2.imwrite(str(output_root / f"{label.lower()}.jpg"), sheet):
            raise OSError(f"Could not write contact sheet for {label}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "fishing_v2.yaml")
    parser.add_argument("--session-root", type=Path, default=PROJECT_ROOT / "assets" / "replay" / "sessions")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "prompt_observation_v1")
    parser.add_argument(
        "--contact-sheets",
        type=Path,
        default=PROJECT_ROOT / "reports" / "fishing_v2" / "prompt_observer_contact_sheets",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_prompt_observation_dataset(
        config_path=args.config,
        session_root=args.session_root,
        output_root=args.output,
        project_root=PROJECT_ROOT,
    )
    _contact_sheets(result.rows, result.output_root, args.contact_sheets)
    print(f"manifest: {result.manifest_path}")
    print(f"frames: {result.summary['frame_count']}")
    print(f"label_counts: {result.summary['label_counts']}")
    print(f"cross_non_ignore_label_conflicts: {result.summary['cross_non_ignore_label_conflicts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
