from __future__ import annotations

from pathlib import Path
import random
import threading
import time

import cv2
import numpy as np
import pytest

from src.config_loader import load_roi_config
from src.detectors.press_background_subtraction_v3 import (
    PressBackgroundModel,
    PressBackgroundSubtractionDetectorV3,
    PressForegroundExtractor,
    PressKeyStripLocator,
)
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.legacy_adapters.press_background_subtraction_v3_adapter import (
    BackgroundSubtractionPressDetectorAdapter,
)
from src.fishing_v2.live.press_v3_shadow import PressV3ShadowRunner
from src.fishing_v2.live.live_detect_only import (
    LiveDetectOnlyConfig,
    LiveDetectOnlyRuntime,
)
from src.fishing_v2.live.session_logger import LiveSessionLogger
from src.fishing_v2.perception.prompt_bundle import load_prompt_bundle
from src.fishing_v2.runtime.press_sequence_aggregator import (
    PressSequenceTemporalAggregator,
)
from tools.run_live_detect_only import parse_args


ROOT = Path(__file__).resolve().parents[2]
STRUCTURAL = ROOT / "tests" / "fixtures" / "press_structural_occupancy"
SESSIONS = ROOT / "assets" / "replay" / "sessions"
SAS = ROOT / "tests" / "fixtures" / "press_sas"


def _read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def _press_roi(session: str, frame: int) -> np.ndarray:
    image = _read(SESSIONS / session / "frames" / f"{frame:06d}.jpg")
    x1, y1, x2, y2 = load_roi_config().pixel_roi(
        "press_sequence", image.shape[1], image.shape[0]
    )
    return image[y1:y2, x1:x2]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("episode_1_shifted_wwaw.png", "WWAW"),
        ("episode_2_transparent_wsaaswa.png", "WWSAASWA"),
        ("session_20260802_145338_episode_1_dsddd.png", "DSDDD"),
        ("session_20260802_145338_episode_3_ddawasw.png", "DDAWASW"),
    ],
)
def test_v3_real_live_roi_is_exact_and_complete(filename: str, expected: str) -> None:
    result = PressBackgroundSubtractionDetectorV3().detect(_read(STRUCTURAL / filename))
    assert "".join(result["sequence_candidate"]) == expected
    assert result["occupied_slot_count"] == len(expected)
    assert result["decoded_count"] == len(expected)
    assert result["frame_complete"] is True
    assert all(
        item["occupancy"] == "EMPTY"
        for item in result["slots"][len(expected):]
    )


@pytest.mark.parametrize(
    ("session", "frame", "expected"),
    [
        ("session_20260709_192315", 472, "ASDWWDWS"),
        ("session_20260710_061220", 507, "AWSA"),
        ("session_20260710_123210", 574, "DW"),
        ("session_20260710_124419", 334, "DSWSS"),
        ("session_20260710_125441", 396, "WASASDD"),
        ("session_20260710_130308", 415, "WWDDWWSS"),
        ("session_20260710_131254", 225, "WDASADS"),
        ("session_20260710_131254", 554, "WAAASA"),
    ],
)
def test_v3_all_confirmed_replay_episodes(
    session: str, frame: int, expected: str
) -> None:
    result = PressBackgroundSubtractionDetectorV3().detect(_press_roi(session, frame))
    assert result["frame_complete"] is True
    assert "".join(result["sequence_candidate"]) == expected
    assert result["occupied_slot_count"] == len(expected)


def test_v3_sas_live_regression() -> None:
    detector = PressBackgroundSubtractionDetectorV3()
    for frame in (159, 160, 161):
        result = detector.detect(_read(SAS / f"frame_{frame:06d}.jpg"))
        assert result["frame_complete"] is True
        assert "".join(result["sequence_candidate"]) == "SAS"


@pytest.mark.parametrize(
    "colour",
    [
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
        (255, 255, 255),
    ],
)
def test_v3_recolour_invariance_uses_shape_not_hue(
    colour: tuple[int, int, int]
) -> None:
    detector = PressBackgroundSubtractionDetectorV3()
    image = _read(STRUCTURAL / "session_20260802_145338_episode_1_dsddd.png")
    baseline = detector.detect(image)
    recoloured = image.copy()
    for slot in baseline["slots"][:5]:
        x1, y1, x2, y2 = map(int, slot["inner_bbox"])
        region = recoloured[y1:y2, x1:x2]
        region[slot["binary_mask"] > 0] = colour
    result = detector.detect(recoloured)
    assert result["frame_complete"] is True
    assert "".join(result["sequence_candidate"]) == "DSDDD"


def test_v3_slot_local_background_correction_handles_small_variation() -> None:
    detector = PressBackgroundSubtractionDetectorV3()
    image = _read(STRUCTURAL / "episode_1_shifted_wwaw.png")
    baseline = detector.detect(image)
    varied = image.astype(np.int16)
    offsets = (-4, -3, -2, -1, 0, 1, 2, 3, 4, 2)
    for offset, slot in zip(offsets, baseline["slots"], strict=True):
        x1, y1, x2, y2 = map(int, slot["inner_bbox"])
        mask = slot["binary_mask"] == 0
        region = varied[y1:y2, x1:x2]
        region[mask] += offset
    result = detector.detect(np.clip(varied, 0, 255).astype(np.uint8))
    assert result["frame_complete"] is True
    assert "".join(result["sequence_candidate"]) == "WWAW"


def test_v3_locator_excludes_instruction_and_progress_and_has_ten_slots() -> None:
    image = _read(STRUCTURAL / "session_20260802_145338_episode_1_dsddd.png")
    result = PressBackgroundSubtractionDetectorV3().detect(image)
    locator = result["locator"]
    assert locator["geometry_stable"] is True
    assert len(locator["slot_bboxes"]) == 10
    assert locator["key_strip_bbox"][1] > image.shape[0] * 0.45
    assert locator["progress_baseline_y"] >= locator["key_strip_bbox"][3]
    assert all(item["foreground_pixel_count"] == 0 for item in result["slots"][5:])


def test_v3_valid_single_and_two_key_sequences_have_no_minimum_length() -> None:
    detector = PressBackgroundSubtractionDetectorV3()
    two_key = _press_roi("session_20260710_123210", 574)
    baseline = detector.detect(two_key)
    assert "".join(baseline["sequence_candidate"]) == "DW"
    single = two_key.copy()
    second = baseline["slots"][1]
    x1, y1, x2, y2 = map(int, second["inner_bbox"])
    patch = single[y1:y2, x1:x2]
    patch[second["binary_mask"] > 0] = (58, 53, 53)
    result = detector.detect(single)
    assert result["frame_complete"] is True
    assert result["sequence_candidate"] == ["D"]


def test_v3_adapter_reuses_full_frame_and_existing_temporal_consensus() -> None:
    image = _read(SESSIONS / "session_20260710_123210" / "frames" / "000574.jpg")
    adapter = BackgroundSubtractionPressDetectorAdapter()
    aggregator = PressSequenceTemporalAggregator()
    aggregate = None
    for index in range(3):
        observation = adapter.observe(
            image, FrameContext(index + 1, index * 0.2, metadata={})
        )
        aggregate = aggregator.update(observation)
    assert aggregate is not None
    assert aggregate.sequence_ready is True
    assert "".join(aggregate.sequence_candidate) == "DW"


def test_cli_defaults_to_legacy_and_shadow_is_explicit() -> None:
    default = parse_args(["--window-title", "test"])
    shadow = parse_args([
        "--window-title", "test",
        "--press-detector-mode", "background-subtraction-shadow",
    ])
    assert default.press_detector_mode == "legacy"
    assert shadow.press_detector_mode == "background-subtraction-shadow"


def test_background_subtraction_shadow_never_creates_action_intent(tmp_path: Path) -> None:
    image = _read(SESSIONS / "session_20260710_123210" / "frames" / "000574.jpg")
    runner = PressV3ShadowRunner(output_root=tmp_path)
    assert runner.observe(
        image, FrameContext(1, 0.0, metadata={}), legacy=None
    ) == []
    events, summary = runner.finish(0.1)
    assert summary["press_v3_shadow_processed_frames"] == 1
    assert events
    assert all(name.startswith("press_v3_") for name, _ in events)
    assert all("ACTION" not in name for name, _ in events)
    assert not list(tmp_path.rglob("*.mp4"))


def test_shadow_worker_is_latest_only_and_does_not_block_capture_loop(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class SlowDetector:
        def detect(self, _panel: np.ndarray) -> dict:
            started.set()
            assert release.wait(timeout=2.0)
            return {
                "detected": False,
                "confidence": 0.0,
                "panel_candidate": False,
                "panel_present": False,
                "sequence_candidate": [],
                "slots": [],
                "processing_latency_ms": 1000.0,
                "rejection_reason": "test",
            }

    image = _read(SESSIONS / "session_20260710_123210" / "frames" / "000574.jpg")
    runner = PressV3ShadowRunner(output_root=tmp_path, detector=SlowDetector())
    runner.observe(image, FrameContext(1, 0.0, metadata={}), legacy=None)
    assert started.wait(timeout=1.0)
    before = time.perf_counter()
    assert runner.observe(
        image, FrameContext(2, 0.01, metadata={}), legacy=None
    ) == []
    assert time.perf_counter() - before < 0.05
    release.set()
    _, summary = runner.finish(0.1)
    assert summary["press_v3_shadow_dropped_busy_frames"] == 1


def test_background_subtraction_live_is_explicit_and_keeps_sink_disabled(
    tmp_path: Path,
) -> None:
    runtime = LiveDetectOnlyRuntime(
        config_path=ROOT / "config" / "fishing_v2.yaml",
        prompt_bundle=load_prompt_bundle(
            ROOT / "artifacts" / "prompt_observer" / "prototype_v1"
        ),
        capture=object(),
        logger=LiveSessionLogger(tmp_path, bundle_version="v3-test"),
        live_config=LiveDetectOnlyConfig(
            duration_seconds=0.0,
            show_overlay=False,
            save_transition_frames=False,
            press_detector_mode="background-subtraction-live",
        ),
        emit_actions=False,
    )
    assert isinstance(
        runtime.press_detector, BackgroundSubtractionPressDetectorAdapter
    )
    assert runtime.action_sink is None


def test_v3_components_are_independent_and_do_not_open_video() -> None:
    assert PressKeyStripLocator is not None
    assert PressBackgroundModel is not None
    assert PressForegroundExtractor is not None
    source = (ROOT / "src" / "fishing_v2" / "live" / "press_v3_shadow.py").read_text(encoding="utf-8")
    assert "VideoWriter" not in source
    assert ".mp4" not in source


def test_v3_seeded_fixture_order_has_stable_results() -> None:
    paths = list(STRUCTURAL.glob("*.png"))
    random.Random(17).shuffle(paths)
    detector = PressBackgroundSubtractionDetectorV3()
    results = ["".join(detector.detect(_read(path))["sequence_candidate"]) for path in paths]
    assert sorted(results) == sorted(["WWAW", "WWSAASWA", "DSDDD", "DDAWASW"])
