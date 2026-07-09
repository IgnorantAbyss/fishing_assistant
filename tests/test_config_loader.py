from pathlib import Path

from src.config_loader import (
    DEFAULT_ROI_CONFIG_PATH,
    DEFAULT_THRESHOLDS_CONFIG_PATH,
    load_roi_config,
    load_thresholds_config,
    normalized_to_pixel_roi,
)


def test_roi_yaml_loads_and_converts_to_reference_pixels() -> None:
    config = load_roi_config(DEFAULT_ROI_CONFIG_PATH)

    assert config.source == DEFAULT_ROI_CONFIG_PATH
    assert config.screen_reference == (2048, 1151)
    assert config.pixel_roi("top_prompt", 2048, 1151) == (614, 23, 1434, 115)
    assert normalized_to_pixel_roi(config.rois["get_window"], 1024, 576) == (768, 317, 963, 484)


def test_missing_configuration_paths_use_built_in_defaults(tmp_path: Path) -> None:
    assert load_roi_config(tmp_path / "missing-roi.yaml").source is None
    assert load_thresholds_config(tmp_path / "missing-thresholds.yaml").source is None


def test_thresholds_yaml_loads() -> None:
    thresholds = load_thresholds_config(DEFAULT_THRESHOLDS_CONFIG_PATH)

    assert thresholds.min_confidence == 0.65
    assert thresholds.unknown_below == 0.55
    assert thresholds.stable_frames_required == 2
    assert thresholds.min_confidence_for("READY") == 0.65
    assert thresholds.debug.max_debug_images == 200
