from src.fishing_v2.domain.observations import (
    GetObservation,
    HookObservation,
    PressObservation,
)
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import ObservationFusion
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.detector_activation import (
    DetectorActivationMode,
    DetectorActivationSnapshot,
)
from src.fishing_v2.runtime.detector_evidence import (
    DetectorEvidenceQualifier,
    HookEvidenceKind,
)


def _hook(*, fill_ratio, features, confidence=0.95):
    return HookObservation(
        True, confidence, 1, 0.2,
        fill_ratio=fill_ratio,
        evidence={"matched_features": features},
    )


def _activation(mode: DetectorActivationMode) -> DetectorActivationSnapshot:
    return DetectorActivationSnapshot(
        mode, DetectorActivationMode.OFF, DetectorActivationMode.OFF,
        25.0 if mode != DetectorActivationMode.OFF else 0.0, 0.0, 0.0,
    )


def _bundle(hook: HookObservation) -> ObservationBundle:
    return ObservationBundle(
        1, 0.2, None, hook,
        PressObservation(False, 0.0, 1, 0.2),
        GetObservation(False, 0.0, 1, 0.2),
    )


def test_rectangle_only_ready_candidate_is_not_active_hook() -> None:
    qualified = DetectorEvidenceQualifier().qualify(
        _bundle(_hook(fill_ratio=0.0, features=["hook_bar_rect"])),
        _activation(DetectorActivationMode.BURST),
    )
    assert qualified.hook.hook_evidence_kind == HookEvidenceKind.RECTANGLE_CANDIDATE
    assert qualified.hook.qualified_detected is False
    assert qualified.bundle.hook.detected is False


def test_rectangle_only_waiting_candidate_never_enters_fusion() -> None:
    qualified = DetectorEvidenceQualifier().qualify(
        _bundle(_hook(fill_ratio=0.0, features=["hook_bar_rect"])),
        _activation(DetectorActivationMode.ARMED),
    )
    evidence = ObservationFusion().fuse(qualified.bundle, RuntimeState.WAITING)
    assert qualified.hook.used_by_fusion is False
    assert RuntimeState.HOOK not in evidence.candidate_states


def test_zero_fill_is_rejected_even_if_bar_fill_feature_is_reported() -> None:
    _, qualification = DetectorEvidenceQualifier().qualify_hook(
        _hook(fill_ratio=0.0, features=["hook_bar_rect", "bar_fill"]),
        DetectorActivationMode.BURST,
    )
    assert qualification.qualified_detected is False
    assert qualification.hook_evidence_kind == HookEvidenceKind.RECTANGLE_CANDIDATE


def test_positive_bar_fill_qualifies_without_divider_line() -> None:
    observation, qualification = DetectorEvidenceQualifier().qualify_hook(
        _hook(fill_ratio=0.42, features=["hook_bar_rect", "bar_fill"]),
        DetectorActivationMode.BURST,
    )
    assert qualification.qualified_detected is True
    assert qualification.hook_evidence_kind == HookEvidenceKind.ACTIVE_HOOK_BAR
    assert observation.detected is True


def test_off_mode_keeps_raw_result_diagnostic_only() -> None:
    observation, qualification = DetectorEvidenceQualifier().qualify_hook(
        _hook(fill_ratio=0.42, features=["hook_bar_rect", "bar_fill"]),
        DetectorActivationMode.OFF,
    )
    assert qualification.raw_detected is True
    assert qualification.qualified_detected is False
    assert qualification.diagnostic_only is True
    assert qualification.used_by_fusion is False
    assert observation.detected is False
