"""Extract a deliberately small, reviewed live-UI template bank from replay frames."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_roi_config, normalized_to_pixel_roi  # noqa: E402
from src.replay_session import DEFAULT_SESSION_ROOT, ReplaySession, latest_session  # noqa: E402


TEMPLATE_ROOT = PROJECT_ROOT / "assets" / "templates" / "live"


@dataclass(frozen=True)
class TemplateSpec:
    state: str
    frame_index: int
    roi_name: str

    @property
    def filename(self) -> str:
        return f"frame_{self.frame_index:03d}_{self.roi_name}.png"


# These intentionally cover only the supplied calibration frames and crop only
# configured UI ROIs, never a full replay frame.
TEMPLATE_SPECS = (
    TemplateSpec("idle", 1, "top_prompt"),
    TemplateSpec("idle", 1, "center_space"),
    TemplateSpec("idle", 500, "top_prompt"),
    TemplateSpec("waiting", 26, "top_prompt"),
    TemplateSpec("waiting", 527, "top_prompt"),
    TemplateSpec("ready", 443, "top_prompt"),
    TemplateSpec("ready", 443, "center_space"),
    TemplateSpec("hook", 459, "hook_bar"),
    TemplateSpec("press", 472, "press_sequence"),
    TemplateSpec("get", 485, "get_window"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a small live template bank from selected replay frames.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--session", type=Path)
    selection.add_argument("--latest", action="store_true")
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--template-root", type=Path, default=TEMPLATE_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="List UI crops without writing any files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_path = args.session if args.session else latest_session(args.session_root)
    session = ReplaySession.load(session_path)
    frames = session.frame_paths()
    config = load_roi_config()
    print("Planned live UI template crops (no full replay frames):")
    for spec in TEMPLATE_SPECS:
        print(f"- frame {spec.frame_index:03d} {spec.roi_name} -> {args.template_root / spec.state / spec.filename}")
    if args.dry_run:
        return 0
    for spec in TEMPLATE_SPECS:
        if spec.frame_index > len(frames):
            raise ValueError(f"Frame {spec.frame_index} is unavailable in {session.path}")
        frame = cv2.imread(str(frames[spec.frame_index - 1]), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Could not read frame {frames[spec.frame_index - 1]}")
        left, top, right, bottom = normalized_to_pixel_roi(config.rois[spec.roi_name], frame.shape[1], frame.shape[0])
        destination = args.template_root / spec.state / spec.filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), frame[top:bottom, left:right]):
            raise OSError(f"Could not write template: {destination}")
    print(f"Wrote {len(TEMPLATE_SPECS)} live UI templates under {args.template_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
