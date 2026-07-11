"""Read-only replay orchestration for architecture validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.fishing_v2.data.prompt_annotation import PromptAnnotationKind, load_prompt_ground_truth
from src.fishing_v2.domain.observations import PromptObservation, PromptObservationKind
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import FusionConfig, ObservationFusion, StateEvidence
from src.fishing_v2.legacy_adapters.get_detector_adapter import LegacyGetDetectorAdapter
from src.fishing_v2.legacy_adapters.hook_detector_adapter import LegacyHookDetectorAdapter
from src.fishing_v2.legacy_adapters.press_detector_adapter import LegacyPressDetectorAdapter
from src.fishing_v2.legacy_adapters.replay_source_adapter import LegacyReplaySourceAdapter
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.replay.v2_replay_report import write_v2_replay_report
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM
from src.fishing_v2.runtime.detector_activation import DetectorActivationConfig, DetectorActivationPolicy
from src.fishing_v2.runtime.runtime_controller import RuntimeController
from src.fishing_v2.runtime.safety_policy import SafetyConfig, SafetyPolicy
from src.fishing_v2.runtime.synchronization import SynchronizationConfig, StartupSynchronizer


@dataclass(frozen=True)
class V2ReplayRun:
    session_id: str
    mode: str
    rows: tuple[dict[str, Any], ...]
    csv_path: Path
    report_path: Path


def _config_objects(config_path: Path):
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fsm = dict(data["fsm"])
    fsm.update({
        "hook_safe_zone_start": data["hook_detector"]["safe_zone_start"],
        "hook_safe_zone_end": data["hook_detector"]["safe_zone_end"],
        "get_retry_interval_seconds": data["get_detector"]["retry_interval_seconds"],
        "get_max_attempts": data["get_detector"]["max_attempts"],
        "get_max_duration_seconds": data["get_detector"]["max_duration_seconds"],
    })
    activation = DetectorActivationConfig(
        hook_armed_fps=data["hook_detector"]["armed_fps"],
        hook_burst_fps=data["hook_detector"]["burst_fps"],
        press_armed_fps=data["press_detector"]["armed_fps"],
        press_burst_fps=data["press_detector"]["burst_fps"],
        get_armed_fps=data["get_detector"]["armed_fps"],
    )
    return (
        FusionConfig(**data["fusion"]),
        FSMConfig(**fsm),
        SynchronizationConfig(**data["sync"]),
        SafetyConfig(**data["safety"]),
        activation,
    )


def _serialize_evidence(evidence: StateEvidence) -> dict[str, Any]:
    return {
        "candidate_states": {state.value: score for state, score in evidence.candidate_states.items()},
        "scores": {state.value: score for state, score in evidence.candidate_states.items()},
        "supporting_observations": list(evidence.supporting_observations),
        "conflicting_observations": list(evidence.conflicting_observations),
        "recommended_state": evidence.recommended_state.value if evidence.recommended_state else None,
        "confidence": evidence.confidence,
        "reason": evidence.reason,
    }


class V2ReplayRunner:
    def __init__(
        self,
        config_path: str | Path,
        *,
        hook_detector: Any | None = None,
        press_detector: Any | None = None,
        get_detector: Any | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        (
            self.fusion_config,
            self.fsm_config,
            self.sync_config,
            self.safety_config,
            self.activation_config,
        ) = _config_objects(self.config_path)
        if self.safety_config.emit_actions:
            raise ValueError("Hybrid Runtime v2 replay requires safety.emit_actions=false")
        self.hook_detector = hook_detector or LegacyHookDetectorAdapter()
        self.press_detector = press_detector or LegacyPressDetectorAdapter()
        self.get_detector = get_detector or LegacyGetDetectorAdapter()

    def run(
        self,
        session_path: str | Path,
        *,
        mode: str,
        report_dir: str | Path,
        start_state: RuntimeState | None = None,
    ) -> V2ReplayRun:
        if mode not in {"scripted_prompt", "no_prompt"}:
            raise ValueError("Replay mode must be scripted_prompt or no_prompt")
        source = LegacyReplaySourceAdapter(session_path)
        prompt_labels = None
        if mode == "scripted_prompt":
            prompt_labels = load_prompt_ground_truth(
                source.session.path / "prompt_ground_truth.yaml", len(source.paths)
            )
        fsm = FishingFSM(self.fsm_config, initial_state=RuntimeState.SYNCING)
        fusion = ObservationFusion(self.fusion_config)
        safety = SafetyPolicy(self.safety_config)
        controller = RuntimeController(
            fusion,
            fsm,
            safety,
            action_sink=None,
            activation_policy=DetectorActivationPolicy(self.activation_config),
        )
        synchronizer = StartupSynchronizer(self.sync_config, started_at=0.0)
        if start_state is not None:
            fsm.force_state(start_state, 0.0, "manual_start_state_override")
        rows: list[dict[str, Any]] = []
        annotation_mapping = {
            PromptAnnotationKind.IDLE_CAST: PromptObservationKind.IDLE_CAST,
            PromptAnnotationKind.WAITING_IN_PROGRESS: PromptObservationKind.WAITING_IN_PROGRESS,
            PromptAnnotationKind.READY_BITE: PromptObservationKind.READY_BITE,
            PromptAnnotationKind.HOOK_INSTRUCTION: PromptObservationKind.HOOK_INSTRUCTION,
            PromptAnnotationKind.PRESS_INSTRUCTION: PromptObservationKind.PRESS_INSTRUCTION,
            PromptAnnotationKind.IGNORE: PromptObservationKind.UNKNOWN,
            PromptAnnotationKind.IDLE_PROMPT: PromptObservationKind.UNKNOWN,
            PromptAnnotationKind.WAITING_PROMPT: PromptObservationKind.UNKNOWN,
            PromptAnnotationKind.READY_PROMPT: PromptObservationKind.UNKNOWN,
            PromptAnnotationKind.OTHER_PROMPT: PromptObservationKind.UNKNOWN,
            PromptAnnotationKind.NO_PROMPT: PromptObservationKind.UNKNOWN,
        }
        for replay_frame in source.frames():
            frame = cv2.imread(str(replay_frame.path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(replay_frame.path)
            context = replay_frame.context
            hook = self.hook_detector.observe(frame, context)
            press = self.press_detector.observe(frame, context)
            get = self.get_detector.observe(frame, context)
            prompt = None
            if prompt_labels is not None:
                annotation = prompt_labels[context.frame_index]
                kind = annotation_mapping[annotation]
                prompt = PromptObservation(
                    kind, 1.0 if annotation != PromptAnnotationKind.IGNORE else 0.0,
                    {kind.value: 1.0}, "scripted_prompt_ground_truth",
                    context.frame_index, context.timestamp,
                    {"annotation": annotation.value, "global_state_not_used": True},
                )
            bundle = ObservationBundle(context.frame_index, context.timestamp, prompt, hook, press, get)
            sync_reason = None
            if fsm.state == RuntimeState.SYNCING:
                sync = synchronizer.observe(bundle, manual_override=start_state)
                sync_reason = sync.reason
                if sync.synchronized:
                    fsm.force_state(sync.state, context.timestamp, sync.reason)
                elif sync.state == RuntimeState.SYNC_REQUIRED:
                    fsm.force_state(RuntimeState.SYNC_REQUIRED, context.timestamp, sync.reason)
            previous = fsm.state
            result = controller.process(
                bundle,
                foreground=None,
                runtime_environment_supported=source.session.manifest.get("screen_size") == [2560, 1440],
            )
            rows.append({
                "frame_index": context.frame_index,
                "global_ground_truth": replay_frame.global_ground_truth,
                "prompt_observation": prompt.kind.value if prompt else None,
                "hook_observation": asdict(hook),
                "press_observation": asdict(press),
                "get_observation": asdict(get),
                "previous_runtime_state": previous.value,
                "state_evidence": _serialize_evidence(result.evidence),
                "next_runtime_state": result.fsm.next_state.value,
                "action_intent": result.fsm.action_request.intent.value,
                "safety_decision": result.safety.decision.value,
                "detector_activation": {
                    "hook": result.activation.hook.value,
                    "press": result.activation.press.value,
                    "get": result.activation.get.value,
                },
                "transition_reason": sync_reason or result.fsm.transition_reason,
            })
        csv_path, report_path = write_v2_replay_report(
            report_dir, source.session.path.name, rows, mode=mode
        )
        return V2ReplayRun(source.session.path.name, mode, tuple(rows), csv_path, report_path)
