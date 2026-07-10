"""Configuration and shared types for offline replay dataset preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.replay_ground_truth import GROUND_TRUTH_STATES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_CONFIG_PATH = PROJECT_ROOT / "config" / "dataset.yaml"


@dataclass(frozen=True)
class DatasetSettings:
    output_dir: Path
    crop_format: str
    crop_jpg_quality: int
    sample_every_n_frames: int
    materialize_crops: bool


@dataclass(frozen=True)
class SamplingRule:
    max_samples: int | None
    strategy: str


@dataclass(frozen=True)
class BoundarySampling:
    enabled: bool
    frames_before: int
    frames_after: int


@dataclass(frozen=True)
class DatasetConfig:
    dataset: DatasetSettings
    sampling: Mapping[str, SamplingRule]
    boundary: BoundarySampling
    source: Path


def _required_mapping(parent: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = parent.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"dataset config requires a {name} mapping")
    return value


def load_dataset_config(path: str | Path = DEFAULT_DATASET_CONFIG_PATH) -> DatasetConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Dataset configuration root must be a mapping: {config_path}")
    dataset = _required_mapping(data, "dataset")
    sampling = _required_mapping(data, "sampling")
    per_session = _required_mapping(sampling, "per_session")
    boundary = _required_mapping(data, "boundary_sampling")

    output_dir_value = dataset.get("output_dir")
    crop_format = dataset.get("crop_format")
    crop_quality = dataset.get("crop_jpg_quality")
    sample_every = dataset.get("sample_every_n_frames")
    materialize = dataset.get("materialize_crops")
    if not isinstance(output_dir_value, str) or not output_dir_value:
        raise ValueError("dataset.output_dir must be a non-empty string")
    if crop_format not in {"jpg", "png"}:
        raise ValueError("dataset.crop_format must be jpg or png")
    if not isinstance(crop_quality, int) or isinstance(crop_quality, bool) or not 1 <= crop_quality <= 100:
        raise ValueError("dataset.crop_jpg_quality must be between 1 and 100")
    if not isinstance(sample_every, int) or isinstance(sample_every, bool) or sample_every < 1:
        raise ValueError("dataset.sample_every_n_frames must be a positive integer")
    if not isinstance(materialize, bool):
        raise ValueError("dataset.materialize_crops must be true or false")

    rules: dict[str, SamplingRule] = {}
    for state in GROUND_TRUTH_STATES:
        raw_rule = per_session.get(state)
        if not isinstance(raw_rule, dict):
            raise ValueError(f"sampling.per_session.{state} is required")
        maximum = raw_rule.get("max_samples")
        strategy = raw_rule.get("strategy")
        if maximum is not None and (
            not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 0
        ):
            raise ValueError(f"sampling.per_session.{state}.max_samples must be null or non-negative")
        if strategy not in {"all", "uniform", "exclude"}:
            raise ValueError(f"sampling.per_session.{state}.strategy is invalid")
        rules[state] = SamplingRule(maximum, str(strategy))

    enabled = boundary.get("enabled")
    before = boundary.get("frames_before")
    after = boundary.get("frames_after")
    if not isinstance(enabled, bool):
        raise ValueError("boundary_sampling.enabled must be true or false")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (before, after)
    ):
        raise ValueError("boundary_sampling frame counts must be non-negative integers")

    output_dir = Path(output_dir_value)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return DatasetConfig(
        dataset=DatasetSettings(
            output_dir=output_dir,
            crop_format=str(crop_format),
            crop_jpg_quality=crop_quality,
            sample_every_n_frames=sample_every,
            materialize_crops=materialize,
        ),
        sampling=rules,
        boundary=BoundarySampling(enabled, before, after),
        source=config_path,
    )
