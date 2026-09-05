"""Passive-only proof: real canonical pixels, mock/recorded actions only."""
from dataclasses import asdict
from functools import partial
import json
from types import SimpleNamespace

import cv2
import pytest

from src.hook_bar_geometry import measure_bar_local_geometry, passive_right_observation
from src.hook_right_geometry import EpisodeRightTarget
from src.fishing_v2.live.hook_right_telemetry import HookRightTelemetry
from src.fishing_v2.domain.observations import HookObservation
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.legacy_adapters.hook_detector_adapter import LegacyHookDetectorAdapter
from src.detectors.hook_detector import detect_hook_bar
from src.fishing_v2.fusion.observation_fusion import ObservationFusion
from src.fishing_v2.runtime.fishing_fsm import FishingFSM, FSMConfig
from src.fishing_v2.runtime.runtime_controller import RuntimeController, ActionExecutionMode
from src.fishing_v2.runtime.safety_policy import SafetyPolicy, SafetyConfig
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from .test_hook_right_discovery import raster
from .test_hook_bar_local_geometry import canvas


def episode(n=1):
    return SimpleNamespace(hook_episode_id=f'hook:{n}',cycle_id=n,started_at=0.,start_hook_applied=True)


def observation(frame, image=None, roi=(973,468,1587,526), resolution=(2560,1440)):
    crop = raster() if image is None else image
    hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    geometry = measure_bar_local_geometry(hsv)
    passive = passive_right_observation(crop,hsv,geometry,roi,resolution)
    return HookObservation(True,.99,frame,frame*.025,evidence={'passive_right':passive})


def update(telemetry, frame, **kwargs):
    return telemetry.observe(episode=kwargs.pop('episode',episode()),
        observation=observation(frame,**kwargs),state_before='HOOK',state_after='HOOK')


def test_shared_identity_no_duplicate_constants_and_two_real_observations():
    from tools.discover_hook_right import EpisodeRightTarget as Diagnostic
    assert Diagnostic is EpisodeRightTarget
    telemetry = HookRightTelemetry()
    assert update(telemetry,1)['target_state'] == 'unconfirmed'
    assert update(telemetry,1)['observation_count'] == 1
    result = update(telemetry,2)
    assert result['target_state'] == 'confirmed'
    assert result['observation_count'] == 2
    assert result['confirmed_fillable_right_x'] == 1476
    assert result['confirmation_latency_from_START_HOOK'] == .05
    assert update(telemetry,1,episode=episode(2))['target_state'] == 'unconfirmed'


@pytest.mark.parametrize('fault',['divider','right','resolution','roi','height','drift','low_mask'])
def test_passive_conflicts_fail_closed(fault):
    t = HookRightTelemetry(); update(t,1); update(t,2)
    image = raster(); kwargs = {}
    if fault == 'divider': image = raster(shift=1,border=505)
    elif fault == 'right': image = raster(border=500)
    elif fault == 'resolution': kwargs['resolution'] = (1920,1080)
    elif fault == 'roi': kwargs['roi'] = (974,468,1588,526)
    elif fault == 'height': image[41,109:357] = (0,0,255)
    elif fault == 'drift': image[16:41,109:111] = 0
    elif fault == 'low_mask':
        image[16:41,109:357] = 0
        cv2.rectangle(image,(110,16),(356,40),(0,0,255),1)
    result = update(t,3,image=image,**kwargs)
    assert result['target_state'] == 'invalidated'
    assert result['confirmed_fillable_right_x'] is None


def test_jitter_missing_right_terminal_and_sync_reset():
    t = HookRightTelemetry();update(t,1);update(t,2)
    image = raster();image[16:41,109] = 0;image[16:41,504] = 0
    result = update(t,3,image=image)
    assert result['confirmed_fillable_right_x'] == 1476
    assert result['current_measurement_agrees'] is None
    t.observe(episode=episode(),observation=None,state_before='HOOK',state_after='SYNC_REQUIRED')
    assert t.latest == {}
    assert update(t,4)['target_state'] == 'unconfirmed'
    t.observe(episode=episode(),observation=observation(5),state_before='HOOK',state_after='RESULT_PENDING')
    assert t.active is None and t.target.confirmed is None


def test_bounded_summary_masks_not_serialized_and_no_external_io(monkeypatch):
    import builtins
    t = HookRightTelemetry(max_segments=2)
    def forbidden(*a,**k): raise AssertionError('passive I/O')
    monkeypatch.setattr(builtins,'open',forbidden)
    for n in range(10):
        obs = observation(n+1)
        t.observe(episode=episode(n),observation=obs,state_before='HOOK',state_after='RESULT_PENDING')
        serialized = json.dumps(obs.evidence)
        assert '_anchor_mask' not in serialized and len(serialized) < 2500
    summary = t.summary()
    assert len(summary['segments']) == 2
    assert summary['dropped_segments'] == 8
    assert summary['cost_sample_count'] <= 512


def strip_telemetry(value):
    if isinstance(value,dict):return {k:strip_telemetry(v) for k,v in value.items() if k!='passive_right'}
    if isinstance(value,(list,tuple)):return [strip_telemetry(v) for v in value]
    return value


@pytest.mark.parametrize('foreground',[True,False])
def test_real_detector_qualification_fsm_crossing_actions_identical(monkeypatch,foreground):
    import src.hook_bar_geometry as canonical
    original = canonical.discover_right
    outputs = []
    for mode in ('disabled','enabled','failure'):
        if mode == 'failure':
            def fail(*a,**k): raise OSError('injected passive failure')
            monkeypatch.setattr(canonical,'discover_right',fail)
        else: monkeypatch.setattr(canonical,'discover_right',original)
        adapter = LegacyHookDetectorAdapter(partial(detect_hook_bar,passive_right_enabled=mode!='disabled'))
        controller = RuntimeController(ObservationFusion(),FishingFSM(FSMConfig(stable_frames=2),
            initial_state=RuntimeState.HOOK_PENDING),SafetyPolicy(SafetyConfig(emit_actions=False)))
        t = HookRightTelemetry(enabled=mode!='disabled'); rows=[]
        for i,endpoint in enumerate([365,365,400,503,380,400],1):
            raw = adapter.observe(canvas(raster(endpoint)),FrameContext(i,i*.025))
            before = controller.fsm.state.value
            result = controller.process(ObservationBundle(i,i*.025,None,raw),foreground=foreground,
                runtime_environment_supported=True,action_mode=ActionExecutionMode.RECORDED_OBSERVATION,
                preserve_proposal=True)
            # Identical full result includes qualification, transitions, crossing
            # payload, intents, Safety, action_applied and commit result.
            rows.append(strip_telemetry({'raw':asdict(raw),'result':asdict(result)}))
            t.observe(episode=episode(),observation=raw,state_before=before,state_after=controller.fsm.state.value)
        outputs.append(rows)
    assert outputs[0] == outputs[1] == outputs[2]
    intents=[r['result']['fsm']['action_request']['intent'].value for r in outputs[0]]
    assert intents.count('HOOK_ACTION') == 1
    assert all(not r['result']['action_applied'] for r in outputs[0])


def test_adapter_reuses_single_canonical_measurement(monkeypatch):
    import src.detectors.hook_detector as detector
    import src.fishing_v2.legacy_adapters.hook_detector_adapter as adapter_module
    calls=[]; original=detector.measure_bar_local_geometry
    def measured(*a,**k):calls.append(1);return original(*a,**k)
    monkeypatch.setattr(detector,'measure_bar_local_geometry',measured)
    monkeypatch.setattr(adapter_module,'measure_hook_crossing_geometry',lambda *a:pytest.fail('rerun'))
    raw = LegacyHookDetectorAdapter().observe(canvas(raster()),FrameContext(1,.025))
    assert calls == [1]
    assert raw.evidence['passive_right']['fillable_right_x'] == 1476


def test_summary_failure_cannot_break_runtime_shutdown():
    t = HookRightTelemetry()
    t.costs.append(('corrupt',None))
    assert t.summary()['summary_reason'] == 'passive_summary_error'
