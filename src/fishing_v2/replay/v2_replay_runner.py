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
from src.fishing_v2.runtime.detector_evidence import (
    DetectorEvidenceQualifier,
    EvidenceQualification,
    EvidenceQualificationConfig,
)
from src.fishing_v2.runtime.runtime_controller import ActionExecutionMode, RuntimeController
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


def _qualification_values(result: Any) -> dict[str, Any]:
    items: dict[str, EvidenceQualification] = {
        "hook": result.hook,
        "press": result.press,
        "get": result.get,
    }
    return {
        "activation": {name: item.activation_mode.value for name, item in items.items()},
        "raw_detected": {name: item.raw_detected for name, item in items.items()},
        "qualified_detected": {name: item.qualified_detected for name, item in items.items()},
        "reason": {name: item.qualification_reason for name, item in items.items()},
        "used": {name: item.used_by_fusion for name, item in items.items()},
        "diagnostic_only": {name: item.diagnostic_only for name, item in items.items()},
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
        action_mode: ActionExecutionMode | str = ActionExecutionMode.STANDARD,
        flat_report: bool = False,
    ) -> V2ReplayRun:
        if mode not in {"scripted_prompt", "no_prompt"}:
            raise ValueError("Replay mode must be scripted_prompt or no_prompt")
        source = LegacyReplaySourceAdapter(session_path)
        action_mode = ActionExecutionMode(action_mode)
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
            evidence_qualifier=DetectorEvidenceQualifier(EvidenceQualificationConfig(
                hook_strong_confidence=self.fusion_config.hook_strong_threshold,
                press_strong_confidence=self.fusion_config.press_strong_threshold,
                get_strong_confidence=self.fusion_config.get_strong_threshold,
            )),
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
                    kind, 1.0 if kind != PromptObservationKind.UNKNOWN else 0.0,
                    {kind.value: 1.0}, "scripted_prompt_ground_truth",
                    context.frame_index, context.timestamp,
                    {"annotation": annotation.value, "global_state_not_used": True},
                )
            raw_bundle = ObservationBundle(context.frame_index, context.timestamp, prompt, hook, press, get)
            sync_reason = None
            frame_start_state = fsm.state
            if fsm.state == RuntimeState.SYNCING:
                _, startup_qualified = controller.qualify_raw_bundle(
                    raw_bundle,
                    action_mode=action_mode,
                )
                sync = synchronizer.observe(startup_qualified.bundle, manual_override=start_state)
                sync_reason = sync.reason
                if sync.synchronized:
                    fsm.force_state(sync.state, context.timestamp, sync.reason)
                elif sync.state == RuntimeState.SYNC_REQUIRED:
                    fsm.force_state(RuntimeState.SYNC_REQUIRED, context.timestamp, sync.reason)
            result = controller.process(
                raw_bundle,
                foreground=None,
                runtime_environment_supported=source.session.manifest.get("screen_size") == [2560, 1440],
                action_mode=action_mode,
            )
            qualifications = _qualification_values(result.qualified)
            rows.append({
                "frame_index": context.frame_index,
                "global_ground_truth": replay_frame.global_ground_truth,
                "prompt_observation": prompt.kind.value if prompt else None,
                "action_mode": action_mode.value,
                "detector_activation_mode": qualifications["activation"],
                "raw_detected": qualifications["raw_detected"],
                "qualified_detected": qualifications["qualified_detected"],
                "qualification_reason": qualifications["reason"],
                "used_by_fusion": qualifications["used"],
                "diagnostic_only": qualifications["diagnostic_only"],
                "hook_observation": asdict(hook),
                "press_observation": asdict(press),
                "get_observation": asdict(get),
                "qualified_hook_observation": asdict(result.qualified.bundle.hook) if result.qualified.bundle.hook else None,
                "qualified_press_observation": asdict(result.qualified.bundle.press) if result.qualified.bundle.press else None,
                "qualified_get_observation": asdict(result.qualified.bundle.get) if result.qualified.bundle.get else None,
                "previous_runtime_state": frame_start_state.value,
                "state_evidence": _serialize_evidence(result.evidence),
                "next_runtime_state": result.fsm.next_state.value,
                "proposed_intent": result.fsm.action_request.intent.value,
                "action_intent": result.fsm.action_request.intent.value,
                "action_applied": result.action_applied,
                "safety_decision": result.safety.decision.value,
                "visual_acknowledgement": result.fsm.visual_acknowledgement,
                "transition_reason": sync_reason or result.fsm.transition_reason,
            })
        csv_path, report_path = write_v2_replay_report(
            report_dir,
            source.session.path.name,
            rows,
            mode=mode,
            action_mode=action_mode.value,
            flat_output=flat_report,
        )
        return V2ReplayRun(source.session.path.name, mode, tuple(rows), csv_path, report_path)
