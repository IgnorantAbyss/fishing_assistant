from src.fishing_v2.domain.action_intent import ActionIntent, ActionRequest
from src.fishing_v2.domain.observations import (
    GetObservation, HookObservation, PressObservation,
    PromptObservation, PromptObservationKind,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import ObservationFusion, StateEvidence
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.detector_activation import (
    DetectorActivationMode,
    DetectorActivationPolicy,
)
from src.fishing_v2.runtime.detector_evidence import DetectorEvidenceQualifier
from src.fishing_v2.runtime.fishing_fsm import FSMConfig, FishingFSM
from src.fishing_v2.runtime.runtime_controller import ActionExecutionMode, RuntimeController
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


def _recovery_bundle(
    frame: int,
    kind: PromptObservationKind,
    *,
    hook: HookObservation | None = None,
    press: PressObservation | None = None,
    get: GetObservation | None = None,
) -> ObservationBundle:
    timestamp = frame * 0.2
    return ObservationBundle(
        frame,
        timestamp,
        _prompt(kind, frame=frame),
        hook or HookObservation(False, 0.0, frame, timestamp),
        press or PressObservation(False, 0.0, frame, timestamp),
        get or GetObservation(False, 0.0, frame, timestamp),
    )


def _recovery_qualified(bundle, qualifier=None):
    qualifier = qualifier or DetectorEvidenceQualifier()
    activation = DetectorActivationPolicy().evaluate(
        RuntimeState.SYNC_REQUIRED, bundle, recorded_observation=True
    )
    return qualifier.qualify(bundle, activation)


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
        ActionRequest(ActionIntent.CAST, 0.5, "cast"), RuntimeState.IDLE,
        _evidence(0.5), foreground=True, already_sent=False, elapsed_since_action=None,
    )
    assert result.decision == SafetyDecision.WAIT


def test_duplicate_action_is_denied() -> None:
    result = SafetyPolicy(SafetyConfig(emit_actions=True)).evaluate(
        ActionRequest(ActionIntent.CAST, 0.9, "cast"), RuntimeState.IDLE,
        _evidence(), foreground=True, already_sent=True, elapsed_since_action=1.0,
    )
    assert result.decision == SafetyDecision.DENY


def test_start_hook_is_denied_outside_ready() -> None:
    policy = SafetyPolicy(SafetyConfig(emit_actions=True))
    request = ActionRequest(
        ActionIntent.START_HOOK,
        0.95,
        "ready_bite_confirmed",
    )
    for state in (
        RuntimeState.WAITING,
        RuntimeState.IDLE,
        RuntimeState.GET,
    ):
        result = policy.evaluate(
            request,
            state,
            _evidence(),
            foreground=True,
            already_sent=False,
            elapsed_since_action=1.0,
        )
        assert result.decision == SafetyDecision.DENY
        assert result.reason == "action_not_allowed_in_runtime_state"


def test_hook_action_is_denied_outside_hook_and_on_focus_loss() -> None:
    policy = SafetyPolicy(SafetyConfig(emit_actions=True))
    request = ActionRequest(
        ActionIntent.HOOK_ACTION,
        0.95,
        "hook_fill_safely_crossed_threshold",
    )
    for state in (
        RuntimeState.WAITING,
        RuntimeState.READY,
        RuntimeState.IDLE,
        RuntimeState.GET,
    ):
        result = policy.evaluate(
            request,
            state,
            _evidence(),
            foreground=True,
            already_sent=False,
            elapsed_since_action=1.0,
        )
        assert result.decision == SafetyDecision.DENY
        assert result.reason == "action_not_allowed_in_runtime_state"

    focus_loss = policy.evaluate(
        request,
        RuntimeState.HOOK,
        _evidence(),
        foreground=False,
        already_sent=False,
        elapsed_since_action=1.0,
    )
    assert focus_loss.decision == SafetyDecision.DENY
    assert focus_loss.reason == "foreground_window_not_confirmed"


def test_foreground_window_is_required() -> None:
    result = SafetyPolicy(SafetyConfig(emit_actions=True)).evaluate(
        ActionRequest(ActionIntent.CAST, 0.9, "cast"), RuntimeState.IDLE,
        _evidence(), foreground=False, already_sent=False, elapsed_since_action=1.0,
    )
    assert result.decision == SafetyDecision.DENY


def test_cooldown_causes_wait() -> None:
    result = SafetyPolicy(SafetyConfig(emit_actions=True, cooldown_sec=1.0)).evaluate(
        ActionRequest(ActionIntent.CAST, 0.9, "cast"), RuntimeState.IDLE,
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
    assert result.action_applied is False
    assert result.fsm.next_state == RuntimeState.IDLE
    assert fsm.actions_applied == frozenset()
    assert sink.requests == []


def test_conflicting_evidence_waits() -> None:
    result = SafetyPolicy(SafetyConfig(emit_actions=True)).evaluate(
        ActionRequest(ActionIntent.CAST, 0.9, "cast"), RuntimeState.IDLE,
        _evidence(conflict=True), foreground=True, already_sent=False, elapsed_since_action=1.0,
    )
    assert result.decision == SafetyDecision.WAIT


def test_live_timeline_reproduces_sync_required_lock_then_recovers_waiting() -> None:
    fsm = FishingFSM(FSMConfig(stable_frames=2, sync_lost_timeout_sec=2.0), initial_state=RuntimeState.READY)
    fusion = ObservationFusion()
    qualifier = DetectorEvidenceQualifier()
    activation = DetectorActivationPolicy()

    def advance(frame: int, timestamp: float, kind: PromptObservationKind):
        prompt = PromptObservation(kind, 0.998, {kind.value: 0.998}, "session_timeline", frame, timestamp, {})
        raw = ObservationBundle(
            frame, timestamp, prompt,
            HookObservation(False, 0.0, frame, timestamp),
            PressObservation(False, 0.0, frame, timestamp),
            GetObservation(False, 0.0, frame, timestamp),
        )
        modes = activation.evaluate(fsm.state, raw, recorded_observation=True)
        qualified = qualifier.qualify(raw, modes)
        evidence = fusion.fuse(qualified.bundle, fsm.state)
        return fsm.advance(evidence, timestamp, qualified.bundle, recorded_observation=True), qualified, evidence

    advance(193, 9.403, PromptObservationKind.WAITING_IN_PROGRESS)
    entered, _, _ = advance(236, 11.540, PromptObservationKind.WAITING_IN_PROGRESS)
    assert entered.next_state == RuntimeState.SYNC_REQUIRED
    assert entered.transition_reason == "persistent_conflicting_or_illegal_evidence"
    locked, _, _ = advance(241, 11.770, PromptObservationKind.WAITING_IN_PROGRESS)
    assert locked.next_state == RuntimeState.SYNC_REQUIRED
    assert locked.transition_reason == "sync_required_blocks_actions"

    fsm.begin_sync_recovery(11.540)
    sync = StartupSynchronizer(
        SynchronizationConfig(observation_frames=10, minimum_consensus_ratio=0.7),
        started_at=11.540,
    )
    timeline = [
        (241, 11.770),
        (246, 11.979),
        (251, 12.185),
        (256, 12.396),
        (261, 12.613),
        (266, 12.824),
        (271, 13.032),
        (276, 13.251),
        (281, 13.467),
        (286, 13.674),
    ]
    recovery = None
    for index, timestamp in timeline:
        prompt = PromptObservation(
            PromptObservationKind.WAITING_IN_PROGRESS,
            0.998,
            {"WAITING_IN_PROGRESS": 0.998},
            "session_timeline",
            index,
            timestamp,
            {},
        )
        raw = ObservationBundle(
            index, timestamp, prompt,
            HookObservation(False, 0.0, index, timestamp),
            PressObservation(False, 0.0, index, timestamp),
            GetObservation(False, 0.0, index, timestamp),
        )
        modes = activation.evaluate(RuntimeState.SYNC_REQUIRED, raw, recorded_observation=True)
        qualified = qualifier.qualify(raw, modes)
        recovery = sync.observe_recovery(qualified, has_conflict=False)
    assert recovery is not None and recovery.synchronized
    assert recovery.state == RuntimeState.WAITING
    assert recovery.observed_frames == 10
    assert index == 286
    recovered = fsm.recover_from_sync_required(recovery.state, timeline[-1][1], recovery.reason)
    assert recovered.previous_state == RuntimeState.SYNC_REQUIRED
    assert recovered.next_state == RuntimeState.WAITING
    assert recovered.action_request.intent == ActionIntent.NONE


def test_sync_required_recovers_from_stable_waiting_and_ready_prompts() -> None:
    for kind, expected in (
        (PromptObservationKind.WAITING_IN_PROGRESS, RuntimeState.WAITING),
        (PromptObservationKind.READY_BITE, RuntimeState.READY),
    ):
        sync = StartupSynchronizer(SynchronizationConfig(observation_frames=3))
        result = None
        for frame in range(1, 4):
            result = sync.observe_recovery(
                _recovery_qualified(_recovery_bundle(frame, kind)),
                has_conflict=False,
            )
        assert result is not None and result.synchronized
        assert result.state == expected


def test_sync_required_idle_recovery_requires_observed_get_absence() -> None:
    config = SynchronizationConfig(observation_frames=2)
    missing_get = StartupSynchronizer(config)
    for frame in range(1, 3):
        raw = _recovery_bundle(frame, PromptObservationKind.IDLE_CAST)
        raw = ObservationBundle(raw.frame_index, raw.timestamp, raw.prompt, raw.hook, raw.press, None)
        result = missing_get.observe_recovery(_recovery_qualified(raw), has_conflict=False)
    assert not result.synchronized
    assert result.rejection_reason == "idle_requires_qualified_get_absence"

    observed_absence = StartupSynchronizer(config)
    for frame in range(1, 3):
        result = observed_absence.observe_recovery(
            _recovery_qualified(_recovery_bundle(frame, PromptObservationKind.IDLE_CAST)),
            has_conflict=False,
        )
    assert result.synchronized and result.state == RuntimeState.IDLE


def test_sync_required_unknown_single_frame_and_conflict_do_not_recover() -> None:
    config = SynchronizationConfig(observation_frames=3)
    unknown = StartupSynchronizer(config)
    for frame in range(1, 4):
        result = unknown.observe_recovery(
            _recovery_qualified(_recovery_bundle(frame, PromptObservationKind.UNKNOWN)),
            has_conflict=False,
        )
    assert not result.synchronized

    jitter = StartupSynchronizer(config)
    first = jitter.observe_recovery(
        _recovery_qualified(_recovery_bundle(1, PromptObservationKind.READY_BITE)),
        has_conflict=False,
    )
    assert not first.synchronized

    conflict = StartupSynchronizer(config)
    for frame in range(1, 5):
        result = conflict.observe_recovery(
            _recovery_qualified(_recovery_bundle(frame, PromptObservationKind.WAITING_IN_PROGRESS)),
            has_conflict=True,
        )
    assert not result.synchronized
    assert result.rejection_reason == "conflicting_evidence_reset_window"


def test_sync_required_specialized_states_require_qualified_evidence() -> None:
    rejected = (
        _recovery_bundle(
            1,
            PromptObservationKind.HOOK_INSTRUCTION,
            hook=HookObservation(True, 0.99, 1, 0.2, fill_ratio=0.0),
        ),
        _recovery_bundle(
            1,
            PromptObservationKind.PRESS_INSTRUCTION,
            press=PressObservation(True, 0.50, 1, 0.2, sequence=("W",)),
        ),
        _recovery_bundle(
            1,
            PromptObservationKind.UNKNOWN,
            get=GetObservation(True, 0.95, 1, 0.2),
        ),
    )
    for raw in rejected:
        result = StartupSynchronizer().observe_recovery(
            _recovery_qualified(raw), has_conflict=False
        )
        assert not result.synchronized

    cases = (
        (
            RuntimeState.HOOK,
            HookObservation(True, 0.95, 1, 0.2, fill_ratio=0.8, evidence={"matched_features": ["bar_fill"]}),
            None,
            None,
        ),
        (
            RuntimeState.PRESS,
            None,
            PressObservation(True, 0.95, 1, 0.2, sequence=("W",)),
            None,
        ),
        (
            RuntimeState.GET,
            None,
            None,
            GetObservation(True, 0.95, 1, 0.2),
        ),
    )
    for expected, hook, press, get in cases:
        qualifier = DetectorEvidenceQualifier()
        sync = StartupSynchronizer(SynchronizationConfig(observation_frames=10))
        first_raw = _recovery_bundle(
            1, PromptObservationKind.UNKNOWN, hook=hook, press=press, get=get
        )
        first = sync.observe_recovery(
            _recovery_qualified(first_raw, qualifier), has_conflict=False
        )
        if expected == RuntimeState.GET:
            assert not first.synchronized
            get_second = GetObservation(True, 0.95, 2, 0.4)
            second = sync.observe_recovery(
                _recovery_qualified(
                    _recovery_bundle(2, PromptObservationKind.UNKNOWN, get=get_second),
                    qualifier,
                ),
                has_conflict=False,
            )
            assert second.synchronized and second.state == RuntimeState.GET
        else:
            assert first.synchronized and first.state == expected


def test_sync_required_activation_is_armed_and_recovery_frame_has_no_action() -> None:
    raw = _recovery_bundle(1, PromptObservationKind.READY_BITE)
    activation = DetectorActivationPolicy().evaluate(
        RuntimeState.SYNC_REQUIRED, raw, recorded_observation=True
    )
    assert activation.hook == DetectorActivationMode.ARMED
    assert activation.press == DetectorActivationMode.ARMED
    assert activation.get == DetectorActivationMode.ARMED

    fsm = FishingFSM(FSMConfig(stable_frames=1), initial_state=RuntimeState.SYNC_REQUIRED)
    controller = RuntimeController(
        ObservationFusion(), fsm, SafetyPolicy(SafetyConfig(emit_actions=False))
    )
    controller.reset_for_sync_recovery(0.0)
    recovery = fsm.recover_from_sync_required(RuntimeState.READY, 0.2, "test_recovery")
    assert recovery.action_request.intent == ActionIntent.NONE
    next_frame = controller.process(
        _recovery_bundle(2, PromptObservationKind.READY_BITE),
        foreground=True,
        runtime_environment_supported=True,
        action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
    )
    assert next_frame.fsm.action_request.intent == ActionIntent.START_HOOK
    assert next_frame.action_applied is False
    assert fsm.actions_applied == frozenset()
