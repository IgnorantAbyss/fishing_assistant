"""Detector scheduling modes derived from runtime state, hints, and panel evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.fishing_v2.domain.observations import PromptObservationKind
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.perception.observation_bundle import ObservationBundle


class DetectorActivationMode(str, Enum):
    OFF = "OFF"
    ARMED = "ARMED"
    BURST = "BURST"
    ACTIVE = "ACTIVE"


@dataclass(frozen=True)
class DetectorActivationConfig:
    hook_armed_fps: float = 5.0
    hook_burst_fps: float = 25.0
    press_armed_fps: float = 5.0
    press_burst_fps: float = 20.0
    get_armed_fps: float = 5.0


@dataclass(frozen=True)
class DetectorActivationSnapshot:
    hook: DetectorActivationMode
    press: DetectorActivationMode
    get: DetectorActivationMode
    hook_fps: float
    press_fps: float
    get_fps: float


class DetectorActivationPolicy:
    """Prompt hints accelerate detectors; only specialized evidence is ACTIVE."""

    def __init__(self, config: DetectorActivationConfig | None = None) -> None:
        self.config = config or DetectorActivationConfig()

    def evaluate(
        self,
        state: RuntimeState,
        bundle: ObservationBundle,
        *,
        recorded_observation: bool = False,
    ) -> DetectorActivationSnapshot:
        prompt = bundle.prompt.kind if bundle.prompt else PromptObservationKind.UNKNOWN

        hook = DetectorActivationMode.OFF
        if state in {RuntimeState.SYNCING, RuntimeState.HOOK_PENDING}:
            hook = DetectorActivationMode.ARMED
        if (
            state in {RuntimeState.SYNCING, RuntimeState.HOOK_PENDING}
            or (recorded_observation and state == RuntimeState.READY)
        ) and prompt == PromptObservationKind.HOOK_INSTRUCTION:
            hook = DetectorActivationMode.BURST
        if state == RuntimeState.HOOK:
            hook = DetectorActivationMode.ACTIVE

        press = DetectorActivationMode.OFF
        if state in {RuntimeState.SYNCING, RuntimeState.RESULT_PENDING}:
            press = DetectorActivationMode.ARMED
        if recorded_observation and state in {RuntimeState.HOOK, RuntimeState.PRESS}:
            press = DetectorActivationMode.ARMED
        if (
            state in {RuntimeState.SYNCING, RuntimeState.RESULT_PENDING}
            or (recorded_observation and state == RuntimeState.HOOK)
        ) and prompt == PromptObservationKind.PRESS_INSTRUCTION:
            press = DetectorActivationMode.BURST
        if state == RuntimeState.PRESS:
            press = DetectorActivationMode.ACTIVE

        get = DetectorActivationMode.OFF
        if state in {RuntimeState.SYNCING, RuntimeState.RESULT_PENDING}:
            get = DetectorActivationMode.ARMED
        if recorded_observation and state in {RuntimeState.HOOK, RuntimeState.PRESS}:
            get = DetectorActivationMode.ARMED
        if state == RuntimeState.GET:
            get = DetectorActivationMode.ACTIVE

        return DetectorActivationSnapshot(
            hook=hook,
            press=press,
            get=get,
            hook_fps=self._fps(hook, self.config.hook_armed_fps, self.config.hook_burst_fps),
            press_fps=self._fps(press, self.config.press_armed_fps, self.config.press_burst_fps),
            get_fps=self.config.get_armed_fps if get != DetectorActivationMode.OFF else 0.0,
        )

    @staticmethod
    def _fps(mode: DetectorActivationMode, armed_fps: float, burst_fps: float) -> float:
        if mode == DetectorActivationMode.OFF:
            return 0.0
        if mode == DetectorActivationMode.ARMED:
            return armed_fps
        return burst_fps
