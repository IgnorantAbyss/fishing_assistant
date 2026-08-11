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
from src.fishing_v2.domain.observations import PressObservation
from src.fishing_v2.legacy_adapters.press_background_subtraction_v3_adapter import (
    BackgroundSubtractionPressDetectorAdapter,
)
from src.fishing_v2.live.press_v3_shadow import PressV3ShadowRunner
from src.fishing_v2.live.press_v3_input_effect import (
    PressV3InputEffectTracker,
)
from src.fishing_v2.live.press_key_activity import (
    PressKeyActivityMonitor,
    WindowsAsyncKeyStateReader,
)
from src.fishing_v2.live.live_detect_only import (
    LiveDetectOnlyConfig,
    LiveDetectOnlyRuntime,
)
from src.fishing_v2.live.session_logger import LiveSessionLogger
from src.fishing_v2.perception.prompt_bundle import load_prompt_bundle
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.detector_activation import DetectorActivationMode
from src.fishing_v2.runtime.detector_evidence import DetectorEvidenceQualifier
from src.fishing_v2.runtime.press_sequence_aggregator import (
    PressSequenceTemporalAggregator,
)
from src.fishing_v2.runtime.runtime_controller import ActionExecutionMode
from tools.run_live_detect_only import parse_args


ROOT = Path(__file__).resolve().parents[2]
STRUCTURAL = ROOT / "tests" / "fixtures" / "press_structural_occupancy"
SESSIONS = ROOT / "assets" / "replay" / "sessions"
SAS = ROOT / "tests" / "fixtures" / "press_sas"
INPUT_EFFECT = ROOT / "tests" / "fixtures" / "press_v3_input_effect"


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


def _full_frame_for_press_roi(roi: np.ndarray) -> np.ndarray:
    full_frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    x1, y1, x2, y2 = load_roi_config().pixel_roi(
        "press_sequence", 2560, 1440
    )
    assert roi.shape[:2] == (y2 - y1, x2 - x1)
    full_frame[y1:y2, x1:x2] = roi
    return full_frame


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


def test_v3_single_complete_frame_is_not_ready() -> None:
    image = _read(SESSIONS / "session_20260710_123210" / "frames" / "000574.jpg")
    observation = BackgroundSubtractionPressDetectorAdapter().observe(
        image, FrameContext(1, 0.0, metadata={})
    )
    qualified, qualification = DetectorEvidenceQualifier()._press(
        observation, DetectorActivationMode.ACTIVE
    )

    assert observation.evidence["press_evidence_version"] == 3
    assert observation.evidence["frame_structurally_complete"] is True
    assert observation.confidence > 0.99
    assert qualified is not None
    assert qualified.detected is False
    assert qualified.sequence_ready is False
    assert qualified.sequence == ()
    assert qualification.sequence_qualification_reason != "legacy_ready_sequence"
    assert qualified.evidence["press_qualification_path"] == "temporal_v3"


def test_v3_consistent_clean_frames_produce_complete_temporal_sequence() -> None:
    image = _read(SESSIONS / "session_20260710_123210" / "frames" / "000574.jpg")
    adapter = BackgroundSubtractionPressDetectorAdapter()
    qualifier = DetectorEvidenceQualifier()
    qualified = None
    qualification = None
    for index in range(3):
        observation = adapter.observe(
            image, FrameContext(index + 1, index * 0.2, metadata={})
        )
        qualified, qualification = qualifier._press(
            observation, DetectorActivationMode.ACTIVE
        )

    assert qualified is not None
    assert qualification is not None
    assert qualified.detected is True
    assert qualified.sequence_ready is True
    assert qualified.sequence == tuple("DW")
    assert qualification.sequence_qualification_reason != "legacy_ready_sequence"
    certificate = qualified.evidence["press_completeness_certificate"]
    assert certificate["complete"] is True
    assert certificate["sequence"] == list("DW")
    assert qualified.evidence["frozen_sequence"] == list("DW")


def test_v3_production_post_input_frames_preserve_frozen_clean_sequence() -> None:
    adapter = BackgroundSubtractionPressDetectorAdapter()
    qualifier = DetectorEvidenceQualifier()
    clean = _full_frame_for_press_roi(
        _read(INPUT_EFFECT / "episode_002_frame_003996_clean_wssddws.png")
    )
    qualified = None
    for frame in (3996, 3997, 3998):
        observation = adapter.observe(
            clean, FrameContext(frame, frame / 20.0, metadata={})
        )
        qualified, _ = qualifier._press(
            observation, DetectorActivationMode.ACTIVE
        )
    assert qualified is not None
    assert qualified.sequence == tuple("WSSDDWS")
    frozen_certificate = qualified.evidence["press_completeness_certificate"]

    for filename, frame in (
        ("episode_002_frame_004029_green_glow.png", 4029),
        ("episode_002_frame_004030_gold_flash.png", 4030),
    ):
        post_input = _full_frame_for_press_roi(_read(INPUT_EFFECT / filename))
        observation = adapter.observe(
            post_input, FrameContext(frame, frame / 20.0, metadata={})
        )
        qualified, _ = qualifier._press(
            observation, DetectorActivationMode.ACTIVE
        )
        assert observation.evidence["episode_input_started"] is True
        assert observation.evidence["post_input_frame"] is True
        assert qualified is not None
        assert qualified.sequence == tuple("WSSDDWS")
        assert qualified.evidence["post_input_excluded"] is True
        assert (
            qualified.evidence["press_completeness_certificate"]
            == frozen_certificate
        )

    clean_after_input = adapter.observe(
        clean, FrameContext(4031, 4031 / 20.0, metadata={})
    )
    qualified, _ = qualifier._press(
        clean_after_input, DetectorActivationMode.ACTIVE
    )
    assert clean_after_input.evidence["input_effect_detected"] is False
    assert clean_after_input.evidence["episode_input_started"] is True
    assert clean_after_input.evidence["post_input_frame"] is True
    assert qualified is not None
    assert qualified.sequence == tuple("WSSDDWS")
    assert qualified.evidence["post_input_excluded"] is True

    summary = qualifier.press_qualification_summary()
    assert summary["press_v3_simple_ready_count"] == 0
    assert summary["press_v3_post_input_excluded_count"] == 3


def test_v3_panel_disappearance_resets_episode_and_allows_new_sequence() -> None:
    first = _read(SESSIONS / "session_20260710_123210" / "frames" / "000574.jpg")
    second = _read(SESSIONS / "session_20260710_061220" / "frames" / "000507.jpg")
    adapter = BackgroundSubtractionPressDetectorAdapter()
    qualifier = DetectorEvidenceQualifier()

    for index in range(3):
        observation = adapter.observe(
            first, FrameContext(index + 1, index * 0.2, metadata={})
        )
        qualified, _ = qualifier._press(
            observation, DetectorActivationMode.ACTIVE
        )
    assert qualified is not None and qualified.sequence == tuple("DW")
    first_episode = observation.evidence["press_episode_id"]

    blank = np.zeros_like(first)
    for index in range(2):
        observation = adapter.observe(
            blank, FrameContext(10 + index, 1.0 + index * 0.2, metadata={})
        )
        qualifier._press(observation, DetectorActivationMode.ACTIVE)

    for index in range(3):
        observation = adapter.observe(
            second, FrameContext(20 + index, 2.0 + index * 0.2, metadata={})
        )
        qualified, _ = qualifier._press(
            observation, DetectorActivationMode.ACTIVE
        )
    assert qualified is not None
    assert qualified.sequence == tuple("AWSA")
    assert observation.evidence["press_episode_id"] != first_episode


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


def test_shadow_worker_drops_new_frames_while_busy_without_blocking_capture(
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
    assert summary[
        "press_v3_shadow_dropped_new_while_busy_frames"
    ] == 1


def test_background_subtraction_live_is_explicit_and_keeps_sink_disabled(
    tmp_path: Path,
) -> None:
    image = _read(
        SESSIONS / "session_20260710_123210" / "frames" / "000574.jpg"
    )
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

    first = None
    final = None
    for index in range(3):
        observation = runtime.press_detector.observe(
            image, FrameContext(index + 1, index * 0.2, metadata={})
        )
        _, qualified = runtime.controller.qualify_raw_bundle(
            ObservationBundle(
                index + 1,
                index * 0.2,
                press=observation,
            ),
            action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        )
        first = first or qualified.bundle.press
        final = qualified.bundle.press
    assert first is not None and first.sequence_ready is False
    assert final is not None and final.sequence_ready is True
    assert final.sequence == tuple("DW")
    assert final.evidence["press_completeness_certificate"]["complete"] is True
    assert final.evidence["press_qualification_path"] == "temporal_v3"


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


def _effect_result(filename: str) -> dict:
    return PressBackgroundSubtractionDetectorV3().detect(
        _read(INPUT_EFFECT / filename)
    )


def test_v3_clean_baseline_is_immutable_and_clean_sequences_remain_exact() -> None:
    tracker = PressV3InputEffectTracker()
    clean = _effect_result("episode_002_frame_003996_clean_wssddws.png")
    first = tracker.evaluate(
        clean, frame_index=3996, source_capture_timestamp=171.3566
    )
    second = tracker.evaluate(
        clean, frame_index=4000, source_capture_timestamp=171.5566
    )
    assert "".join(first["sequence_candidate"]) == "WSSDDWS"
    assert first["frame_structurally_complete"] is True
    assert first["frame_clean_eligible"] is True
    assert second["frame_clean_eligible"] is True
    assert tracker.baseline_frame_index == 3996


@pytest.mark.parametrize(
    "filename",
    [
        "episode_002_frame_004029_green_glow.png",
        "episode_002_frame_004030_gold_flash.png",
    ],
)
def test_v3_episode_2_green_and_gold_input_effects_are_detected(
    filename: str,
) -> None:
    tracker = PressV3InputEffectTracker()
    tracker.evaluate(
        _effect_result("episode_002_frame_003996_clean_wssddws.png"),
        frame_index=3996,
        source_capture_timestamp=171.3566,
    )
    result = tracker.evaluate(
        _effect_result(filename),
        frame_index=int(filename.split("frame_")[1][:6]),
        source_capture_timestamp=175.8,
    )
    assert result["input_effect_detected"] is True
    assert result["episode_input_started"] is True
    assert result["frame_clean_eligible"] is False
    assert result["per_slot_halo_flash_metrics"][0][
        "diffuse_halo_or_flash"
    ] is True


@pytest.mark.parametrize(
    "filename",
    [
        "episode_003_frame_006756_green_glow.png",
        "episode_003_frame_006757_green_yellow_glow.png",
    ],
)
def test_v3_episode_3_coloured_input_effects_are_detected(
    filename: str,
) -> None:
    tracker = PressV3InputEffectTracker()
    tracker.evaluate(
        _effect_result("episode_003_frame_006724_clean_wdada.png"),
        frame_index=6724,
        source_capture_timestamp=291.1757,
    )
    result = tracker.evaluate(
        _effect_result(filename),
        frame_index=int(filename.split("frame_")[1][:6]),
        source_capture_timestamp=295.4,
    )
    assert result["input_effect_detected"] is True
    assert result["post_input_frame"] is True


def test_v3_recoloured_clean_glyph_is_not_an_input_effect() -> None:
    detector = PressBackgroundSubtractionDetectorV3()
    image = _read(
        STRUCTURAL / "session_20260802_145338_episode_1_dsddd.png"
    )
    baseline = detector.detect(image)
    tracker = PressV3InputEffectTracker()
    tracker.evaluate(
        baseline, frame_index=1, source_capture_timestamp=0.0
    )
    recoloured = image.copy()
    for slot in baseline["slots"][:5]:
        x1, y1, x2, y2 = map(int, slot["inner_bbox"])
        region = recoloured[y1:y2, x1:x2]
        region[slot["binary_mask"] > 0] = (0, 255, 0)
    result = tracker.evaluate(
        detector.detect(recoloured),
        frame_index=2,
        source_capture_timestamp=0.1,
    )
    assert "".join(result["sequence_candidate"]) == "DSDDD"
    assert result["input_effect_detected"] is False
    assert result["frame_clean_eligible"] is True


def test_v3_input_started_latches_until_next_episode_reset() -> None:
    tracker = PressV3InputEffectTracker()
    clean = _effect_result("episode_002_frame_003996_clean_wssddws.png")
    tracker.evaluate(clean, frame_index=3996, source_capture_timestamp=0.0)
    tracker.evaluate(
        _effect_result("episode_002_frame_004029_green_glow.png"),
        frame_index=4029,
        source_capture_timestamp=0.1,
    )
    latched = tracker.evaluate(
        clean, frame_index=4035, source_capture_timestamp=0.2
    )
    assert latched["input_effect_detected"] is False
    assert latched["input_started_latched"] is True
    assert latched["frame_clean_eligible"] is False
    tracker.reset()
    reset = tracker.evaluate(
        _effect_result("episode_003_frame_006724_clean_wdada.png"),
        frame_index=6724,
        source_capture_timestamp=1.0,
    )
    assert reset["episode_input_started"] is False
    assert reset["frame_clean_eligible"] is True
    assert tracker.baseline_frame_index == 6724


def _drain_shadow(
    runner: PressV3ShadowRunner,
    image: np.ndarray,
    frame_index: int,
) -> list[tuple[str, dict]]:
    if image.shape[:2] != (1440, 2560):
        full_frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        x1, y1, x2, y2 = load_roi_config().pixel_roi(
            "press_sequence", 2560, 1440
        )
        assert image.shape[:2] == (y2 - y1, x2 - x1)
        full_frame[y1:y2, x1:x2] = image
        image = full_frame
    runner.observe(
        image,
        FrameContext(frame_index, frame_index / 100.0, metadata={}),
        legacy=None,
        runtime_state="PRESS",
    )
    assert runner._future is not None
    runner._future.result(timeout=2.0)
    return runner._process_completed()


def test_v3_post_input_observation_never_enters_aggregator_or_changes_frozen(
    tmp_path: Path,
) -> None:
    class CountingAggregator:
        def __init__(self) -> None:
            self.inner = PressSequenceTemporalAggregator()
            self.calls = 0

        def reset(self) -> None:
            self.inner.reset()

        def update(self, observation):
            self.calls += 1
            return self.inner.update(observation)

    aggregator = CountingAggregator()
    runner = PressV3ShadowRunner(
        output_root=tmp_path, aggregator=aggregator
    )
    clean = _read(
        INPUT_EFFECT / "episode_002_frame_003996_clean_wssddws.png"
    )
    for frame in (3996, 3997, 3998):
        _drain_shadow(runner, clean, frame)
    before = aggregator.calls
    events = _drain_shadow(
        runner,
        _read(INPUT_EFFECT / "episode_002_frame_004029_green_glow.png"),
        4029,
    )
    assert aggregator.calls == before
    assert any(
        name == "press_v3_disagreement"
        and payload["reason"] == "post_input_frame_excluded"
        for name, payload in events
    )
    _, summary = runner.finish(41.0)
    assert summary["press_v3_episode_summaries"][0]["v3_sequence"] == list(
        "WSSDDWS"
    )


def test_read_only_key_activity_is_mockable_and_never_emits_input() -> None:
    class FakeApi:
        def __init__(self) -> None:
            self.down: set[int] = set()
            self.reads: list[int] = []

        def panic_pressed(self, virtual_key: int) -> bool:
            self.reads.append(virtual_key)
            return virtual_key in self.down

    api = FakeApi()
    reader = WindowsAsyncKeyStateReader(api=api)
    monitor = PressKeyActivityMonitor(reader)
    assert monitor.poll(
        frame_index=1,
        timestamp=0.0,
        runtime_state="PRESS",
        press_episode_id=7,
        action_sink_press_emission_active=False,
    ) == ()
    api.down.add(0x57)
    activity = monitor.poll(
        frame_index=2,
        timestamp=0.1,
        runtime_state="PRESS",
        press_episode_id=7,
        action_sink_press_emission_active=False,
    )
    assert len(activity) == 1
    assert activity[0].key == "W"
    assert activity[0].source == "external_or_manual_candidate"
    assert activity[0].payload()["telemetry_only"] is True
    assert len(api.reads) == 8


def test_runtime_press_emission_crossing_poll_interval_is_not_manual() -> None:
    class FakeReader:
        def __init__(self) -> None:
            self.down: set[str] = set()

        def is_down(self, key: str) -> bool:
            return key in self.down

    reader = FakeReader()
    monitor = PressKeyActivityMonitor(reader)
    monitor.poll(
        frame_index=576,
        timestamp=28.0,
        runtime_state="PRESS",
        press_episode_id=1,
        action_sink_press_emission_active=False,
    )
    monitor.record_runtime_emission(
        action_id="cycle:1:PRESS_SEQUENCE",
        press_episode_id=1,
        sequence=tuple("SAWSDAAA"),
        emission_started_at=27.35,
        emission_completed_at=28.8633773,
    )
    reader.down.add("W")
    activity = monitor.poll(
        frame_index=577,
        timestamp=28.8643665,
        runtime_state="RESULT_PENDING",
        press_episode_id=1,
        action_sink_press_emission_active=False,
    )
    assert len(activity) == 1
    event = activity[0].payload()
    assert event["source"] == "runtime_emission_correlated"
    assert event["source"] != "external_or_manual_candidate"
    assert event["previous_poll_timestamp"] == pytest.approx(28.0)
    assert event["current_poll_timestamp"] == pytest.approx(28.8643665)
    assert event["runtime_press_emission_overlap"] is True
    assert event["runtime_press_emission"] == {
        "action_id": "cycle:1:PRESS_SEQUENCE",
        "press_episode_id": 1,
        "sequence": list("SAWSDAAA"),
        "emission_started_at": pytest.approx(27.35),
        "emission_completed_at": pytest.approx(28.8633773),
    }
    reader.down.clear()
    monitor.poll(
        frame_index=578,
        timestamp=28.9,
        runtime_state="RESULT_PENDING",
        press_episode_id=1,
        action_sink_press_emission_active=False,
    )
    reader.down.add("D")
    later = monitor.poll(
        frame_index=579,
        timestamp=29.0,
        runtime_state="RESULT_PENDING",
        press_episode_id=1,
        action_sink_press_emission_active=False,
    )
    assert later[0].source == "external_or_manual_candidate"


def test_overlapping_runtime_emission_with_unmatched_key_is_ambiguous() -> None:
    class FakeReader:
        down: set[str] = set()

        def is_down(self, key: str) -> bool:
            return key in self.down

    reader = FakeReader()
    monitor = PressKeyActivityMonitor(reader)
    monitor.poll(
        frame_index=1,
        timestamp=10.0,
        runtime_state="PRESS",
        press_episode_id=2,
        action_sink_press_emission_active=False,
    )
    monitor.record_runtime_emission(
        action_id="cycle:2:PRESS_SEQUENCE",
        press_episode_id=2,
        sequence=("S",),
        emission_started_at=10.1,
        emission_completed_at=10.2,
    )
    reader.down.add("W")
    event = monitor.poll(
        frame_index=2,
        timestamp=10.201,
        runtime_state="RESULT_PENDING",
        press_episode_id=2,
        action_sink_press_emission_active=False,
    )[0]
    assert event.source == "runtime_emission_or_external_ambiguous"


@pytest.mark.parametrize(
    "episode_index,sequence,freeze_frame",
    [
        (1, tuple("SAWSDAAA"), 574),
        (2, tuple("SSA"), 8471),
    ],
)
def test_v3_episode_summary_latches_legacy_and_authoritative_outcome(
    tmp_path: Path,
    episode_index: int,
    sequence: tuple[str, ...],
    freeze_frame: int,
) -> None:
    runner = PressV3ShadowRunner(output_root=tmp_path)
    runner._active = True
    runner._episode = episode_index
    runner._episode_started_at = 1.0
    runner._episode_v3_sequence = sequence
    runner._episode_v3_complete = True
    certificate = {
        "complete": True,
        "occupied_count": len(sequence),
        "decoded_count": len(sequence),
        "completeness_confidence": 0.9931,
    }
    qualified = PressObservation(
        True,
        0.99,
        freeze_frame,
        2.0,
        sequence=sequence,
        panel_present=True,
        sequence_candidate=sequence,
        sequence_ready=True,
        evidence={
            "occupied_slot_count": len(sequence),
            "press_completeness_certificate": certificate,
        },
    )
    runner.record_legacy_authoritative_observation(
        qualified,
        frame_index=freeze_frame,
        timestamp=2.0,
    )
    runner.record_authoritative_action_scheduled(
        episode_index=episode_index,
        sequence=sequence,
    )
    runner.record_authoritative_action_completed(
        episode_index=episode_index,
        action_id=f"cycle:{episode_index}:PRESS_SEQUENCE",
        sequence=sequence,
        applied=True,
        terminal_outcome="completed",
    )
    runner.record_visual_ack("acknowledged")
    final_absent = PressObservation(
        False,
        0.0,
        freeze_frame + 4,
        3.0,
        panel_present=False,
        sequence_candidate=(),
        sequence_ready=False,
        evidence={"press_completeness_certificate": None},
    )
    runner.record_legacy_authoritative_observation(
        final_absent,
        frame_index=freeze_frame + 4,
        timestamp=3.0,
    )
    _, payload = runner._close_episode(
        3.0, frame_index=freeze_frame + 4
    )[0]
    runner.finish(3.1)
    assert payload["legacy_final_frame_observation"]["panel_present"] is False
    assert payload["legacy_temporal_ready"] is True
    assert payload["legacy_completeness_certificate_complete"] is True
    assert payload["legacy_sequence_candidate"] == list(sequence)
    assert payload["legacy_episode_ever_temporal_ready"] is True
    assert payload["legacy_episode_complete_certificate"]["complete"] is True
    assert payload["first_legacy_temporal_ready_frame"] == freeze_frame
    assert payload["first_legacy_complete_certificate_frame"] == freeze_frame
    assert payload["legacy_frozen_frame"] == freeze_frame
    assert payload["first_legacy_nonempty_candidate_frame"] == freeze_frame
    assert payload["legacy_frozen_sequence"] == list(sequence)
    assert payload["legacy_frozen_occupied_count"] == len(sequence)
    assert payload["legacy_frozen_decoded_count"] == len(sequence)
    assert payload["legacy_frozen_completeness_confidence"] == pytest.approx(
        0.9931
    )
    assert payload["authoritative_action_scheduled"] is True
    assert payload["authoritative_action_sequence"] == list(sequence)
    assert payload["authoritative_action_completed"] is True
    assert payload["authoritative_visual_acknowledged"] is True
    assert payload["legacy_authoritative_action_outcome"] == {
        "scheduled": True,
        "sequence": list(sequence),
        "action_id": f"cycle:{episode_index}:PRESS_SEQUENCE",
        "completed": True,
        "terminal_outcome": "completed",
        "visual_acknowledged": True,
    }
    assert payload["v3_episode_complete"] is True
    assert payload["v3_frozen_sequence"] == list(sequence)
    assert payload["legacy_v3_agreement"] == "exact"


def test_v3_debug_separates_capture_worker_and_write_timestamps(
    tmp_path: Path,
) -> None:
    runner = PressV3ShadowRunner(
        output_root=tmp_path,
        config=type("Config", (), {
            "debug_evidence": True,
            "debug_max_episodes": 1,
            "debug_max_frames_per_episode": 1,
        })(),
    )
    _drain_shadow(
        runner,
        _read(INPUT_EFFECT / "episode_002_frame_003996_clean_wssddws.png"),
        3996,
    )
    runner.finish(40.0)
    payload = __import__("json").loads(next(
        tmp_path.rglob("result.json")
    ).read_text(encoding="utf-8"))
    assert payload["source_frame_index"] == 3996
    assert payload["source_capture_timestamp"] == pytest.approx(39.96)
    assert payload["worker_completed_timestamp"] is not None
    assert payload["debug_write_completed_timestamp"] is not None
    assert payload["filesystem_time_note"] == (
        "filesystem creation time is not capture time"
    )


def test_press_v3_live_validation_reports_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "reports/fishing_v2/press_v3_validation/" in ignore
