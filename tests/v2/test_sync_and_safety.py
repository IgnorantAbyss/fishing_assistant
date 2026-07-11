from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.observations import (
    GetObservation, HookObservation, PressObservation,
    PromptObservation, PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import ObservationFusion, StateEvidence
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM
from src.fishing_v2.runtime.runtime_controller import RuntimeController
from src.fishing_v2.runtime.safety_policy import SafetyConfig, SafetyDecision, SafetyPolicy
from src.fishing_v2.runtime.synchronization import SynchronizationConfig, StartupSynchronizer


def _prompt(kind, confidence=0.95, frame=1):
    return PromptObservation(kind, confidence, {kind.value: confidence}, "fake", frame, frame * 0.2, {})


def _bundle(frame=1, prompt=None, *, press=False):
    timestamp = frame * 0.2
    return ObservationBundle(
        frame, timestamp, prompt,
        HookObservation(False, 0.0, frame, timestamp),
        PressObservation(press, 0.98 if press else 0.0, frame, timestamp),
        GetObservation(False, 0.0, frame, timestamp),
    )


def _evidence(confidence=0.95, conflict=False):
    return StateEvidence({RuntimeState.IDLE: confidence}, (), ("x",) if conflict else (), RuntimeState.IDLE, confidence, "test", 1, 0.2)


def test_prompt_consensus_synchronizes_waiting() -> None:
    sync = StartupSynchronizer(SynchronizationConfig(observation_frames=10, minimum_consensus_ratio=0.7, minimum_confidence=0.8, timeout_sec=5.0))
    result = None
    for frame in range(1, 11):
        kind = PromptObservationKind.WAITING_IN_PROGRESS if frame <= 7 else PromptObservationKind.HOOK_INSTRUCTION
        result = sync.observe(_bundle(frame, _prompt(kind, frame=frame)))
    assert result.state == RuntimeState.WAITING
    assert result.synchronized is True


def test_strong_press_synchronizes_immediately() -> None:
    result = StartupSynchronizer().observe(_bundle(1, press=True))
    assert result.state == RuntimeState.PRESS
    assert result.synchronized


def test_instruction_hint_alone_does_not_synchronize_press() -> None:
    sync = StartupSynchronizer(SynchronizationConfig(observation_frames=2, timeout_sec=5.0))
    sync.observe(_bundle(1, _prompt(PromptObservationKind.PRESS_INSTRUCTION, frame=1)))
    result = sync.observe(_bundle(2, _prompt(PromptObservationKind.PRESS_INSTRUCTION, frame=2)))
    assert result.state == RuntimeState.SYNC_REQUIRED


def test_missing_prompt_observer_never_fakes_sync() -> None:
    sync = StartupSynchronizer(SynchronizationConfig(observation_frames=2, timeout_sec=5.0))
    sync.observe(_bundle(1))
    result = sync.observe(_bundle(2))
    assert result.state == RuntimeState.SYNC_REQUIRED


def test_manual_start_state_override_is_supported() -> None:
    result = StartupSynchronizer().observe(_bundle(), manual_override=RuntimeState.GET)
    assert result.state == RuntimeState.GET
    assert result.synchronized


def test_sync_required_always_denies_actions() -> None:
    result = SafetyPolicy().evaluate(
        ActionRequest(ActionIntent.NONE, 0.0, "none"), RuntimeState.SYNC_REQUIRED,
        _evidence(), foreground=True, already_sent=False, elapsed_since_action=None,
    )
    assert result.decision == SafetyDecision.DENY


def test_low_confidence_action_waits() -> None:
    result = SafetyPolicy(SafetyConfig(emit_actions=True)).evaluate(
        ActionRequest(ActionIntent.CAST, 0.5, "cast"), RuntimeState.CAST_PENDING,
        _evidence(0.5), foreground=True, already_sent=False, elapsed_since_action=None,
    )
    assert result.decision == SafetyDecision.WAIT


def test_duplicate_action_is_denied() -> None:
    result = SafetyPolicy(SafetyConfig(emit_actions=True)).evaluate(
        ActionRequest(ActionIntent.CAST, 0.9, "cast"), RuntimeState.CAST_PENDING,
        _evidence(), foreground=True, already_sent=True, elapsed_since_action=1.0,
    )
    assert result.decision == SafetyDecision.DENY


def test_foreground_window_is_required() -> None:
    result = SafetyPolicy(SafetyConfig(emit_actions=True)).evaluate(
        ActionRequest(ActionIntent.CAST, 0.9, "cast"), RuntimeState.CAST_PENDING,
        _evidence(), foreground=False, already_sent=False, elapsed_since_action=1.0,
    )
    assert result.decision == SafetyDecision.DENY


def test_cooldown_causes_wait() -> None:
    result = SafetyPolicy(SafetyConfig(emit_actions=True, cooldown_sec=1.0)).evaluate(
        ActionRequest(ActionIntent.CAST, 0.9, "cast"), RuntimeState.CAST_PENDING,
        _evidence(), foreground=True, already_sent=False, elapsed_since_action=0.1,
    )
    assert result.decision == SafetyDecision.WAIT


def test_emit_actions_false_never_calls_sink() -> None:
    class Sink:
        def __init__(self): self.requests = []
        def emit(self, request): self.requests.append(request)

    sink = Sink()
    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.IDLE)
    controller = RuntimeController(
        ObservationFusion(), fsm, SafetyPolicy(SafetyConfig(emit_actions=False)), action_sink=sink
    )
    result = controller.process(
        _bundle(1, _prompt(PromptObservationKind.IDLE_CAST)),
        foreground=True,
        runtime_environment_supported=True,
    )
    assert result.fsm.action_request.intent == ActionIntent.CAST
    assert result.safety.decision == SafetyDecision.WAIT
    assert sink.requests == []


def test_conflicting_evidence_waits() -> None:
    result = SafetyPolicy(SafetyConfig(emit_actions=True)).evaluate(
        ActionRequest(ActionIntent.CAST, 0.9, "cast"), RuntimeState.CAST_PENDING,
        _evidence(conflict=True), foreground=True, already_sent=False, elapsed_since_action=1.0,
    )
    assert result.decision == SafetyDecision.WAIT
