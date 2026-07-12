"""Build visual-only Hook Bar contact sheets and a minimal boundary review pack."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SESSION_ROOT = ROOT / "assets" / "replay" / "sessions"
OUTPUT_ROOT = ROOT / "reports" / "fishing_v2" / "hook_bar_candidate_review"
CANDIDATE_PATH = ROOT / "data" / "annotations" / "hook_bar_ground_truth.yaml"

FORMAL_SESSIONS = (
    "session_20260709_192315",
    "session_20260710_061220",
    "session_20260710_123210",
    "session_20260710_124419",
    "session_20260710_125441",
    "session_20260710_130308",
    "session_20260710_131254",
)
TRIAL_SESSION = "session_20260709_192231"


def hook_episodes(session_id: str) -> list[tuple[int, int]]:
    data = yaml.safe_load(
        (SESSION_ROOT / session_id / "ground_truth.yaml").read_text(encoding="utf-8")
    )
    return [
        (int(item["start"]), int(item["end"]))
        for item in data["segments"]
        if item["state"] == "HOOK"
    ]


def _frame(session_id: str, frame_index: int) -> np.ndarray:
    path = SESSION_ROOT / session_id / "frames" / f"{frame_index:06d}.jpg"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _contact_tile(image: np.ndarray, session_id: str, frame_index: int) -> np.ndarray:
    full = cv2.resize(image, (320, 180))
    height, width = image.shape[:2]
    focus = image[round(height * 0.12):round(height * 0.46), round(width * 0.25):round(width * 0.75)]
    focus = cv2.resize(focus, (480, 180))
    tile = np.hstack((full, focus))
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (90, 90, 90), 2)
    cv2.putText(
        tile, f"{session_id}  frame {frame_index}", (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
    )
    return tile


def create_contact_sheets() -> list[Path]:
    destination = OUTPUT_ROOT / "contact_sheets"
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for session_id in FORMAL_SESSIONS:
        for episode_index, (start, end) in enumerate(hook_episodes(session_id), start=1):
            frame_indices = list(range(max(1, start - 5), end + 6))
            tiles = [_contact_tile(_frame(session_id, index), session_id, index) for index in frame_indices]
            for page_index in range(0, len(tiles), 18):
                page_tiles = tiles[page_index:page_index + 18]
                rows = []
                for row_start in range(0, len(page_tiles), 3):
                    row = page_tiles[row_start:row_start + 3]
                    while len(row) < 3:
                        row.append(np.zeros_like(page_tiles[0]))
                    rows.append(np.hstack(row))
                sheet = np.vstack(rows)
                output = destination / (
                    f"{session_id}_episode_{episode_index:02d}_page_{page_index // 18 + 1:02d}.jpg"
                )
                cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
                outputs.append(output)
    return outputs


def _load_candidates() -> dict[str, Any]:
    data = yaml.safe_load(CANDIDATE_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("status") != "auto_candidate":
        raise ValueError("Hook Bar candidate YAML must have status: auto_candidate")
    return data


def _labelled_image(
    session_id: str, frame_index: int, label: str, destination: Path
) -> None:
    image = _frame(session_id, frame_index)
    colour = {
        "VISIBLE": (0, 220, 0),
        "NOT_VISIBLE": (0, 0, 255),
        "UNCERTAIN": (0, 180, 255),
    }[label]
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 92), (0, 0, 0), -1)
    cv2.putText(
        image, f"{session_id}  frame {frame_index}  AUTO: {label}",
        (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, colour, 3,
    )
    resized = cv2.resize(image, (1280, 720))
    cv2.imwrite(str(destination), resized, [cv2.IMWRITE_JPEG_QUALITY, 92])


def create_review_pack() -> tuple[Path, list[Path]]:
    data = _load_candidates()
    image_root = OUTPUT_ROOT / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    images: list[Path] = []
    for session_id, session in data["sessions"].items():
        for episode_index, episode in enumerate(session["episodes"], start=1):
            start = episode.get("visible_start")
            end = episode.get("visible_end")
            uncertain = {int(value) for value in episode.get("uncertain_frames", [])}
            selected: list[int] = []
            if start is not None and end is not None:
                selected.extend([
                    max(1, int(start) - 1), int(start), int(episode["first_clear_frame"]),
                    int(episode["last_clear_frame"]), int(end), int(end) + 1,
                ])
            selected = list(dict.fromkeys(selected))[:6]
            for frame_index in selected:
                if frame_index in uncertain:
                    label = "UNCERTAIN"
                elif start is not None and int(start) <= frame_index <= int(end):
                    label = "VISIBLE"
                else:
                    label = "NOT_VISIBLE"
                output = image_root / (
                    f"{session_id}_episode_{episode_index:02d}_frame_{frame_index:06d}_{label.lower()}.jpg"
                )
                _labelled_image(session_id, frame_index, label, output)
                images.append(output)
            rows.append({
                "session_id": session_id,
                "episode_index": episode_index,
                "proposed_start": "" if start is None else int(start),
                "proposed_end": "" if end is None else int(end),
                "confidence": episode["confidence"],
                "uncertain_frames": ";".join(str(value) for value in episode.get("uncertain_frames", [])),
                "human_confirmed_start": "",
                "human_confirmed_end": "",
                "review_status": "",
                "notes": episode.get("notes", ""),
            })
    csv_path = OUTPUT_ROOT / "review_items.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual-only Hook Bar candidate review artifacts.")
    parser.add_argument("--contact-sheets", action="store_true")
    parser.add_argument("--review-pack", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.contact_sheets and not args.review_pack:
        raise SystemExit("Choose --contact-sheets or --review-pack")
    if args.contact_sheets:
        outputs = create_contact_sheets()
        print(f"contact_sheets={len(outputs)} path={OUTPUT_ROOT / 'contact_sheets'}")
    if args.review_pack:
        csv_path, images = create_review_pack()
        print(f"review_csv={csv_path} images={len(images)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
