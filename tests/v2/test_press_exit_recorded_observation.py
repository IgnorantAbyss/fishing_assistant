from pathlib import Path

from src.fishing_v2.domain.observations import (
    GetObservation,
    HookObservation,
    PressObservation,
    PromptObservation,
    PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import StateEvidence
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.replay.v2_replay_runner import V2ReplayRunner
from src.fishing_v2.runtime.detector_activation import (
    DetectorActivationMode,
    DetectorActivationSnapshot,
)
from src.fishing_v2.runtime.detector_evidence import (
    DetectorEvidenceQualifier,
    EvidenceQualificationConfig,
)
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM
from src.fishing_v2.runtime.runtime_controller import ActionExecutionMode


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "fishing_v2.yaml"
SESSION = ROOT / "assets" / "replay" / "sessions" / "session_20260710_123210"


def _press(
    frame: int,
    *,
    present: bool,
    candidate: bool = True,
    absent_frames: int = 0,
    disappeared: bool = False,
) -> PressObservation:
    return PressObservation(
        detected=present,
        confidence=0.98 if present else 0.0,
        frame_index=frame,
        timestamp=frame * 0.2,
        sequence=tuple("DW") if present else (),
        panel_candidate=candidate,
        panel_present=present,
        sequence_candidate=tuple("DW") if present else (),
        sequence_ready=present,
        sequence_confidence=0.95 if present else 0.0,
        evidence={
            "press_evidence_version": 2,
            "panel_bbox": [100, 100, 300, 200] if present else None,
            "panel_absent_frames": absent_frames,
            "panel_disappeared": disappeared,
        },
    )


def _bundle(
    press: PressObservation,
    *,
    prompt: PromptObservationKind = PromptObservationKind.UNKNOWN,
    get: bool = False,
) -> ObservationBundle:
    frame = press.frame_index
    timestamp = press.timestamp
    return ObservationBundle(
        frame,
        timestamp,
        PromptObservation(prompt, 1.0, {prompt.value: 1.0}, "synthetic", frame, timestamp),
        HookObservation(False, 0.0, frame, timestamp),
        press,
        GetObservation(get, 0.98 if get else 0.0, frame, timestamp),
    )


def _activation() -> DetectorActivationSnapshot:
    return DetectorActivationSnapshot(
        hook=DetectorActivationMode.OFF,
        press=DetectorActivationMode.ACTIVE,
        get=DetectorActivationMode.ARMED,
        hook_fps=0.0,
        press_fps=20.0,
        get_fps=5.0,
    )


def _evidence(frame: int, state: RuntimeState | None = None) -> StateEvidence:
    return StateEvidence(
        {} if state is None else {state: 0.98},
        (), (), state, 0.98 if state else 0.0, "synthetic",
        frame, frame * 0.2,
    )


def test_press_panel_presence_clears_only_after_stable_disappearance() -> None:
    qualifier = DetectorEvidenceQualifier(EvidenceQualificationConfig(
        press_panel_confirmation_frames=1,
        press_panel_disappearance_frames=2,
    ))
    qualifier.qualify(_bundle(_press(1, present=True)), _activation())
    first = qualifier.qualify(_bundle(_press(2, present=False)), _activation())
    assert first.bundle.press is not None
    assert first.bundle.press.evidence["panel_absent_frames"] == 1
    assert first.bundle.press.evidence["panel_disappeared"] is False
    second = qualifier.qualify(_bundle(_press(3, present=False)), _activation())
    assert second.bundle.press is not None
    assert second.bundle.press.evidence["panel_absent_frames"] == 2
    assert second.bundle.press.evidence["panel_disappeared"] is True
    assert second.bundle.press.evidence["selected_clean_frame"] is None


def test_absence_without_a_prior_panel_is_not_a_disappearance() -> None:
    qualifier = DetectorEvidenceQualifier(EvidenceQualificationConfig(
        press_panel_confirmation_frames=1,
        press_panel_disappearance_frames=2,
    ))
    qualifier.qualify(_bundle(_press(1, present=False)), _activation())
    result = qualifier.qualify(_bundle(_press(2, present=False)), _activation())
    assert result.bundle.press is not None
    assert result.bundle.press.evidence["panel_absent_frames"] == 2
    assert result.bundle.press.evidence["panel_disappeared"] is False


def test_recorded_press_holds_on_single_miss_then_acknowledges_stable_absence() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.PRESS)
    single = fsm.advance(
        _evidence(1), 0.2,
        _bundle(_press(1, present=False, candidate=True, absent_frames=1)),
        recorded_observation=True,
    )
    assert single.next_state == RuntimeState.PRESS
    stable = fsm.advance(
        _evidence(2), 0.4,
        _bundle(_press(2, present=False, candidate=True, absent_frames=2, disappeared=True)),
        recorded_observation=True,
    )
    assert stable.next_state == RuntimeState.RESULT_PENDING
    assert stable.visual_acknowledgement == "qualified_press_panel_disappeared"


def test_recorded_press_does_not_exit_while_panel_is_present() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.PRESS)
    result = fsm.advance(
        _evidence(1, RuntimeState.PRESS), 0.2,
        _bundle(_press(1, present=True), prompt=PromptObservationKind.IDLE_CAST),
        recorded_observation=True,
    )
    assert result.next_state == RuntimeState.PRESS


def test_recorded_result_pending_reaches_idle_after_extended_panel_absence() -> None:
    fsm = FishingFSM(
        FSMConfig(stable_frames=1, recorded_press_exit_idle_frames=4),
        initial_state=RuntimeState.RESULT_PENDING,
    )
    result = fsm.advance(
        _evidence(4), 0.8,
        _bundle(_press(4, present=False, absent_frames=4, disappeared=True)),
        recorded_observation=True,
    )
    assert result.next_state == RuntimeState.IDLE
    assert result.visual_acknowledgement == "qualified_press_panel_stably_absent"


def test_get_panel_still_has_priority_after_press() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.PRESS)
    result = fsm.advance(
        _evidence(2, RuntimeState.GET), 0.4,
        _bundle(_press(2, present=False, absent_frames=2, disappeared=True), get=True),
        recorded_observation=True,
    )
    assert result.next_state == RuntimeState.GET
    assert result.visual_acknowledgement == "qualified_get_panel_present"


def test_production_mode_keeps_existing_immediate_clear_semantics() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.PRESS)
    result = fsm.advance(
        _evidence(1), 0.2,
        _bundle(_press(1, present=False, candidate=False, absent_frames=1)),
        recorded_observation=False,
    )
    assert result.next_state == RuntimeState.RESULT_PENDING


def test_session_123210_recorded_replay_exits_press_without_applying_action(tmp_path: Path) -> None:
    run = V2ReplayRunner(CONFIG).run(
        SESSION,
        mode="scripted_prompt",
        report_dir=tmp_path,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
        flat_report=False,
    )
    assert len(run.rows) == 600
    assert run.rows[-1]["next_runtime_state"] == "IDLE"
    assert sum(row["proposed_intent"] == "PRESS_SEQUENCE" for row in run.rows) == 1
    assert not any(row["action_applied"] for row in run.rows)
    assert not any(row["next_runtime_state"] == "SYNC_REQUIRED" for row in run.rows)
    transitions = {
        int(row["frame_index"]): row["next_runtime_state"]
        for row in run.rows
        if row["previous_runtime_state"] != row["next_runtime_state"]
    }
    assert transitions[585] == "RESULT_PENDING"
    assert transitions[587] == "IDLE"
