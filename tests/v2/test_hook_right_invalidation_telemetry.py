"""Precision-only telemetry: pinned pre-change implementation is the oracle."""
import copy
import json
from pathlib import Path
import subprocess
import types

import numpy as np
import pytest

from src.hook_right_geometry import EpisodeRightTarget
from src.fishing_v2.live.hook_right_telemetry import HookRightTelemetry
from .test_hook_right_discovery import observation, confirmed_target
from .test_hook_right_telemetry import observation as live_observation, episode


BASELINE = '50f8adcd0c98c04cb7dc3ee9c583ed84b3767838'


def baseline_module(path):
    source = subprocess.check_output(['git', 'show', f'{BASELINE}:{path}'],
                                     cwd=Path(__file__).resolve().parents[2], text=True)
    module = types.ModuleType('pre_precision_telemetry')
    exec(compile(source, path, 'exec'), module.__dict__)
    return module


@pytest.fixture(scope='module')
def old_target():
    return baseline_module('src/hook_right_geometry.py').EpisodeRightTarget


def fault_row(fault, frame=3):
    row = observation(frame)
    changes = {
        'resolution_changed': {'resolution': [1920,1080]},
        'roi_shape_changed': {'roi_dimensions': [615,58]},
        'roi_bounds_changed': {'roi_bounds': [974,468,1588,526]},
        'divider_changed': {'divider_x': 358},
        'bar_band_changed': {'bar_local_y_top': 17},
        'anchor_top_changed': {'anchor_bbox': (109,17,357,41)},
        'anchor_right_changed': {'anchor_bbox': (109,16,358,41)},
        'anchor_bottom_changed': {'anchor_bbox': (109,16,357,42)},
        'anchor_left_out_of_tolerance': {'anchor_bbox': (111,16,357,41)},
        'anchor_bbox_ownership_changed': {'_mask_bbox': (110,16,357,41)},
        'anchor_pixel_support_unavailable': {'anchor_bbox': (110,16,357,41), '_anchor_mask': None},
        'fixed_border_mask_changed': {'_anchor_rails': bytes(len(row['_anchor_rails']))},
        'anchor_mask_iou_low': {'_anchor_mask': bytes(len(row['_anchor_mask']))},
        'geometry_unavailable': {'divider_x': None},
        'conflicting_current_frame_geometry': {'fillable_right_x': 502},
    }
    row.update(changes[fault])
    if 'anchor_bbox' in changes[fault]:
        row['_mask_bbox'] = row['anchor_bbox']
    return row


REASONS = [
    'resolution_changed','roi_shape_changed','roi_bounds_changed','divider_changed',
    'bar_band_changed','anchor_top_changed','anchor_right_changed','anchor_bottom_changed',
    'anchor_left_out_of_tolerance','anchor_bbox_ownership_changed',
    'anchor_pixel_support_unavailable','fixed_border_mask_changed','anchor_mask_iou_low',
    'geometry_unavailable','conflicting_current_frame_geometry',
]


@pytest.mark.parametrize('reason', REASONS)
def test_precise_branch_with_identical_invalidation_lifecycle(reason, old_target):
    old, new = old_target(), EpisodeRightTarget()
    rows = [observation(1), observation(2), fault_row(reason), observation(4), observation(5)]
    for row in rows:
        assert old.update('one',copy.deepcopy(row)) == new.update('one',copy.deepcopy(row))
    detail = new.first_invalidation
    assert new.invalidation_reason == detail['reason'] == reason
    assert detail['frame_index'] == 3 and detail['confirmed_before_invalidation']
    assert detail['confirmed_target_x'] == 503
    assert detail['confirmation_count_before_invalidation'] == 1
    assert detail['reference']['fillable_right_x'] == 503
    assert '_anchor_mask' not in json.dumps(detail)
    assert detail['coordinate_space'] == 'native_roi_pixels'
    assert old.update('two',observation(1)) == new.update('two',observation(1))
    assert new.first_invalidation is None


def test_missing_masks_exact_key_fallback_and_short_circuit_priority(old_target):
    old, new = old_target(), EpisodeRightTarget()
    for frame in (1,2):
        old.update('one',observation(frame)); new.update('one',observation(frame))
    row = observation(3, _anchor_mask=None, _anchor_rails=None)
    assert old.compatible(old.key,row) == new.compatible(new.key,row) is True
    assert old.update('one',row) == new.update('one',row)
    assert not new.invalidated
    row = fault_row('roi_shape_changed',4)
    row['resolution'] = (1,1)
    old.update('one',row); new.update('one',row)
    assert new.invalidation_reason == 'roi_shape_changed'


def test_boolean_oracle_and_lifecycle_stress(old_target):
    rng = np.random.default_rng(42)
    old, new = old_target(), EpisodeRightTarget()
    base = observation(1)
    for ep in range(1000):
        for frame in range(1,7):
            row = dict(base,frame_id=frame,timestamp=frame*.025)
            if frame > 2:
                row['fillable_right_x'] = rng.choice([None,503,502])
                row['anchor_bbox'] = (int(rng.integers(109,112)),16,357,41)
                row['_mask_bbox'] = row['anchor_bbox']
                if rng.random() < .3: row['_anchor_mask'] = None
            if old.key is not None:
                key = (tuple(row['roi_dimensions']),tuple(row['resolution']),tuple(row['roi_bounds']),
                       tuple(row['anchor_bbox']),row['divider_x'],row['bar_local_y_top'],row['bar_local_y_bottom'])
                assert old.compatible(key,row) == new.compatible(key,row)
            assert old.update(ep,copy.deepcopy(row)) == new.update(ep,copy.deepcopy(row))


def test_defensive_unknown_and_diagnostics_failure_preserve_clear(monkeypatch):
    target = confirmed_target()
    monkeypatch.setattr(target,'compatible',lambda *args: False)
    target.update('one',observation(3))
    assert target.invalidation_reason == 'unknown_geometry_change'
    target = confirmed_target()
    monkeypatch.setattr(target,'_invalidation_details',lambda *args: (_ for _ in ()).throw(OSError('diagnostic failure')))
    assert target.update('one',fault_row('divider_changed'))['status'] == 'invalidated'
    assert target.confirmed is None


def feed(t, obs, ep=1, after='HOOK'):
    return t.observe(episode=episode(ep), observation=obs,state_before='HOOK',state_after=after)


def test_session_totals_survive_retention_and_representatives_are_bounded():
    t = HookRightTelemetry()
    base = live_observation(1)
    for ep in range(1000):
        for frame in range(1,4):
            obs = copy.deepcopy(base)
            obs = type(obs)(True,.99,frame,frame*.025,evidence=obs.evidence)
            if frame == 3 and ep % 3 == 0:
                obs.evidence['passive_right'].identity['divider_x'] += 1
            feed(t,obs,ep,after='RESULT_PENDING' if frame == (1 if ep%3==2 else 3) else 'HOOK')
            if ep%3==2: break
    s = t.summary()
    assert len(s['segments']) == 256 and s['dropped_segments'] == 744
    assert s['hook_right_final_state_counts'] == dict(confirmed=333,invalidated=334,unconfirmed=333)
    assert s['hook_right_invalidation_reason_counts'] == {'divider_changed':334}
    assert s['hook_right_invalidation_confirmed_before_counts'] == {'confirmed':334}
    assert len(s['hook_right_invalidation_representatives']['divider_changed']) == 3
    assert t.summary() == s  # Reading summary never counts again.
    t.finish('again')
    assert t.summary() == s


def test_near_full_formula_unchanged_and_missing_fields_are_not_new_gates():
    t = HookRightTelemetry()
    feed(t,live_observation(1)); feed(t,live_observation(2))
    obs = live_observation(3)
    row = obs.evidence['passive_right'].identity
    row.update(fillable_right_x=None,fill_endpoint_x=503)
    feed(t,obs)
    assert t.summary()['hook_right_near_full_observed_count'] == 1  # Cached right already supported.
    obs = live_observation(4)
    obs.evidence['passive_right'].identity.update(fill_endpoint_x=None,divider_x=None,fillable_right_x=None)
    feed(t,obs)
    s = t.summary()
    assert s['hook_right_near_full_missing_reason_counts'] == {
        'no_fill_endpoint':1,'no_divider':1,'no_confirmed_or_current_right':1}
    assert s['hook_right_near_full_target_invalidated_frame_count'] == 1
    assert s['hook_right_near_full_observed_count'] == 1
    # Invalidated isn't a near-full gate when a current right is present.
    t2 = HookRightTelemetry(); feed(t2,live_observation(1)); feed(t2,live_observation(2))
    obs = live_observation(3)
    obs.evidence['passive_right'].identity.update(fillable_right_x=502,fill_endpoint_x=502)
    feed(t2,obs)
    assert t2.summary()['hook_right_near_full_observed_count'] == 1
    assert t2.summary()['hook_right_near_full_missing_reason_counts'] == {}


def test_full_passive_before_after_stream_and_active_aggregate(old_target):
    old_module = baseline_module('src/fishing_v2/live/hook_right_telemetry.py')
    old_module.EpisodeRightTarget = old_target
    old, new = old_module.HookRightTelemetry(), HookRightTelemetry()
    for ep in range(1,8):
        for frame in range(1,8):
            obs = live_observation(frame)
            row = obs.evidence['passive_right'].identity
            if frame == 4:
                if ep == 1: row['fillable_right_x'] = None
                elif ep == 2: row['divider_x'] += 1
                elif ep == 3: row['_anchor_mask'] = bytes(len(row['_anchor_mask']))
                elif ep == 4: row['fill_endpoint_x'] = None
                elif ep == 5: row['fillable_right_x'] = 502
                elif ep == 6: row['bar_local_y_bottom'] += 1
            if frame >= 5: row['fill_endpoint_x'] = 503
            before = feed(old,copy.deepcopy(obs),ep)
            after = feed(new,copy.deepcopy(obs),ep)
            for key in before:
                if key not in ('update_cost_ms','invalidation_reason'):
                    assert before[key] == after[key]
            assert old.active['first_near_full_frame'] == new.active['first_near_full_frame']
            assert old.active['confirmed_before_near_full'] == new.active['confirmed_before_near_full']
        assert sum(new.summary()['hook_right_final_state_counts'].values()) == ep
        old.finish('terminal'); new.finish('terminal')
        assert sum(new.summary()['hook_right_final_state_counts'].values()) == ep


def test_unconfirmed_failure_uses_fixed_size_metadata_and_does_not_recount():
    t = HookRightTelemetry()
    feed(t,live_observation(1))
    obs = live_observation(2)
    obs.evidence['passive_right'].identity['divider_x'] += 1
    feed(t,obs)
    frozen = json.dumps(t.active['first_invalidation_deltas'],sort_keys=True)
    for f in range(3,30): feed(t,live_observation(f))
    assert json.dumps(t.active['first_invalidation_deltas'],sort_keys=True) == frozen
    assert len(frozen) < 3000
    assert t.summary()['hook_right_invalidation_confirmed_before_counts'] == {'unconfirmed':1}
    assert t.summary()['hook_right_invalidation_reason_counts'] == {'divider_changed':1}
