"""Offline prototype only; synthetic rasters are NOT Live evidence."""
from dataclasses import replace
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest

from src.hook_bar_geometry import measure_bar_local_geometry
from tools.discover_hook_right import (
    discover_right, measure_sample, summarize, EpisodeRightTarget, episode_audit, run_archive,
    geometry_audit, public_row)


def raster(endpoint=400, border=504, shift=0):
    """Synthetic thin-border topology matching the inspected native pixels."""
    image = np.zeros((58, 614, 3), np.uint8)
    image[16:41, 109+shift:357+shift] = (0, 0, 255)
    image[16:41, 357+shift] = 255
    image[16:41, 358+shift:border] = (58, 53, 53)
    image[19:38, 359+shift:endpoint+1] = (247, 202, 90)
    image[17, 358+shift:border] = (72, 68, 69)
    image[39, 358+shift:border] = (68, 64, 64)
    image[16:41, border] = (66, 63, 64)
    return image


def measured(image):
    return measure_sample(image, 1, 0.025)[0]


@pytest.mark.parametrize("border,shift", [(504, 0), (480, -7), (523, 8)])
def test_unfilled_full_and_return_are_independent_of_endpoint(border, shift):
    endpoints = [390, border-1, border-8, 405, 366]
    rows = [measured(raster(end, border, shift)) for end in endpoints]
    assert [r["fillable_right_x"] for r in rows] == [border-1]*len(rows)
    assert rows[0]["distance_to_right"] > 0
    assert rows[1]["distance_to_right"] == 0
    stats = summarize(rows, "synthetic")["right_stats"]
    assert stats == dict(min=border-1, median=border-1, max=border-1, range=0, stddev=0)


def test_fill_endpoint_not_used_as_input_or_cached_history():
    crop = raster()
    geometry = measure_bar_local_geometry(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    expected = discover_right(crop, geometry)
    for endpoint in [None, -100, 100000]:
        assert discover_right(crop, replace(geometry, fill_endpoint_x=endpoint)) == expected


def test_border_centre_is_not_fillable_or_outer_shadow_extent():
    row = measured(raster())
    assert row["structural_border_x"] == 504
    assert row["fillable_right_x"] == 503  # Inclusive inner pixel centre.
    assert row["decorative_outer_right_x"] is None  # Shadow extent not proven.
    assert row["edge_support_rows"] == row["total_rows"] == 19
    assert row["band_bottom_exclusive"]


def test_diagonal_outside_bar_and_false_vertical_edge_cannot_vote():
    crop = raster()
    expected = measured(crop)["fillable_right_x"]
    changed = crop.copy()
    for x in range(0, 614, 60):
        cv2.line(changed, (x, 0), (x+45, 57), (200, 200, 200), 3)
    changed[16:41, 109:506] = crop[16:41, 109:506]
    changed[16:41, 550:552] = (72, 68, 69)
    assert measured(changed)["fillable_right_x"] == expected


@pytest.mark.parametrize("fault", ["missing", "weak", "thick", "false_background_edge", "clipped", "one_row"])
def test_unsupported_border_abstains_without_searching_background(fault):
    crop = raster()
    if fault == "missing":
        crop[16:41, 504] = 0
    elif fault == "weak":
        crop[19:38, 504] = crop[19:38, 503]
    elif fault == "thick":
        crop[16:41, 505] = crop[16:41, 504]
    elif fault == "false_background_edge":
        crop[16:41, 504] = 0
        crop[16:41, 560] = (66, 63, 64)
    elif fault == "clipped":
        crop[:, 503:] = crop[:, 502:503]
    else:
        crop[20:38, 504] = 0
    assert measured(crop)["fillable_right_x"] is None


def test_background_contaminated_border_is_an_explicit_prototype_limit():
    crop = raster()
    # Reproduce spatial variation, not change existing detector thresholds.
    crop[25:35, 504] = (79, 76, 77)
    row = measured(crop)
    assert row["fillable_right_x"] is None
    assert row["bar_right_reason"] == "right_border_missing_weak_or_conflicting"
    assert row["edge_candidates"]  # Includes support counts for the failed gate.


def test_diagnostic_is_non_mutating_and_production_cannot_call_it(monkeypatch):
    from src.fishing_v2.runtime.hook_crossing_geometry import measure_hook_crossing_geometry
    from src.config_loader import load_roi_config
    import tools.discover_hook_right as prototype

    crop = raster()
    before = crop.copy()
    measured(crop)
    assert np.array_equal(crop, before)
    monkeypatch.setattr(prototype, "discover_right", lambda *a: pytest.fail("Production called prototype"))
    image = np.zeros((1440, 2560, 3), np.uint8)
    x1, y1, x2, y2 = load_roi_config().pixel_roi("hook_bar_precise", 2560, 1440)
    image[y1:y2, x1:x2] = crop
    geometry = measure_hook_crossing_geometry(image)
    assert geometry.fill_endpoint_x == x1+400
    # Existing canonical geometry retains sole ownership, with right unknown.
    canonical = measure_bar_local_geometry(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
    assert canonical.evidence(x1, y1)["bar_inner_right"] is None


def test_no_evidence_summary_and_missing_anchor():
    row = measured(np.zeros((58, 614, 3), np.uint8))
    summary = summarize([row], "empty")
    assert summary["accepted"] == 0
    assert summary["right_stats"] is None
    assert summary["abstain_reasons"] == {"bar_local_geometry_unavailable": 1}


def test_local_original_sep05_sequence_and_background_failure():
    archive = Path(__file__).resolve().parents[2] / "reports/fishing_v2/production_v3/session_20260905_032141.zip"
    if not archive.exists():
        pytest.skip("Original local anomaly archive is not distributed in Git")
    rows = []
    with zipfile.ZipFile(archive) as z:
        for frame in range(6129, 6140):
            name = next(n for n in z.namelist() if n.endswith(f"/episode_26/frame_{frame:06}_raw.png"))
            crop = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
            rows.append(measured(crop))
        name = next(n for n in z.namelist() if n.endswith("/episode_36/frame_010099_raw.png"))
        crop = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
        assert measured(crop)["fillable_right_x"] is None
    assert [r["fillable_right_x"] for r in rows] == [503]*11
    assert rows[0]["fill_endpoint_x"] == 396
    assert rows[1]["fill_endpoint_x"] == 503
    assert rows[-1]["fill_endpoint_x"] == 359


def observation(frame=1, **overrides):
    row = measured(raster())
    row.update(frame_id=frame, timestamp=frame*.025, resolution=[2560, 1440],
               roi_bounds=[973,468,1587,526], **overrides)
    return row


def test_two_independent_frames_and_same_x_confirm_without_fill_history():
    target = EpisodeRightTarget()
    assert target.update("one", observation(fill_endpoint_x=None))["status"] == "unconfirmed"
    result = target.update("one", observation(2, fill_endpoint_x=-100))
    assert result["status"] == "confirmed"
    assert result["confirmed_x"] == 503


def test_duplicate_or_out_of_order_capture_is_not_an_independent_detection():
    target = EpisodeRightTarget()
    row = observation()
    target.update("one", row)
    assert target.update("one", row)["status"] == "unconfirmed"
    assert target.update("one", {**row, "frame_id": 2})["status"] == "unconfirmed"
    assert target.update("one", observation(2))["status"] == "confirmed"


def test_one_frame_only_and_low_confidence_do_not_confirm():
    target = EpisodeRightTarget()
    target.update("one", observation())
    for frame in range(2,20):
        assert target.update("one", observation(frame, bar_right_confidence=.1))["status"] == "unconfirmed"


def test_duplicate_frame_cannot_hide_changed_roi_geometry():
    target = EpisodeRightTarget()
    target.update("one", observation())
    target.update("one", observation(2))
    result = target.update("one", {**observation(2), "resolution": [1920,1080]})
    assert result["status"] == "invalidated"
    assert result["event"] == "geometry_changed"
    assert result["confirmed_x"] is None


def test_conflicting_geometry_invalidates_without_reusing_previous_target():
    target = EpisodeRightTarget()
    target.update("one", observation())
    target.update("one", observation(2))
    result = target.update("one", observation(3, fillable_right_x=502))
    assert result["status"] == "invalidated"
    assert result["confirmed_x"] is None
    for frame in range(4,10):
        assert target.update("one", observation(frame))["status"] == "invalidated"


@pytest.mark.parametrize("change", [
    {"resolution": [1920,1080]}, {"roi_bounds": [974,468,1588,526]},
    {"roi_dimensions": [615,58]}, {"anchor_bbox": [110,16,357,41]},
    {"divider_x": 358}, {"bar_local_y_top": 17}, {"anchor_bbox": None},
])
def test_resolution_roi_anchor_or_divider_change_invalidates(change):
    target = EpisodeRightTarget()
    target.update("one", observation())
    target.update("one", observation(2))
    result = target.update("one", {**observation(3), **change})
    assert result["status"] == "invalidated"
    assert result["confirmed_x"] is None


def test_missing_edge_can_bridge_only_when_current_geometry_still_matches():
    target = EpisodeRightTarget()
    target.update("one", observation())
    target.update("one", observation(2))
    for frame in range(3,20):
        assert target.update("one", observation(frame, fillable_right_x=None))["confirmed_x"] == 503
    new = target.update("two", observation(1, fillable_right_x=None))
    assert new["status"] == "unconfirmed" and new["confirmed_x"] is None
    assert target.update("two", observation(2, fillable_right_x=480))["status"] == "unconfirmed"
    assert target.update("two", observation(3, fillable_right_x=480))["confirmed_x"] == 480


@pytest.mark.parametrize("border,shift", [(504,0),(480,-7),(523,8)])
def test_colour_mixed_bright_border_uses_structure_not_absolute_colour(border, shift):
    image = raster(border=border, shift=shift)
    # Keep the two horizontal corners, change the interior border substantially.
    image[24:35,border] = (150,140,145)
    assert measured(image)["fillable_right_x"] is None
    row, _ = measure_sample(image, 1, .025, method="structural")
    assert row["fillable_right_x"] == border-1
    assert row["bar_right_reason"] == "structural_fallback_joined_track_edges"
    assert row["structural_transition"]["support_rows"] == 19


@pytest.mark.parametrize("fault", ["missing", "weak", "thick", "clipped", "diagonal", "rail_disagreement", "outside_vertical", "one_row"])
def test_structural_fallback_abstains_on_weak_or_false_geometry(fault):
    image = raster()
    image[24:35,504] = (150,140,145)  # Force baseline rejection.
    if fault == "missing":
        image[16:41,504] = 0
    elif fault == "weak":
        image[19:38,504] = image[19:38,503]
    elif fault == "thick":
        image[16:41,505] = image[16:41,504]
    elif fault == "clipped":
        image[:,503:] = image[:,502:503]
    elif fault == "diagonal":
        image[19:38,504] = 0
        cv2.line(image, (501,19), (520,38), (150,140,145), 1)
    elif fault == "rail_disagreement":
        image[39,500:520] = (68,64,64)
    elif fault == "outside_vertical":
        image[16:41,504] = 0
        image[16:41,550] = (150,140,145)
    else:
        image[20:38,504] = 0
    assert measure_sample(image, 1, .025, method="structural")[0]["fillable_right_x"] is None


def test_structural_fallback_with_external_diagonal_is_fill_independent():
    image = raster()
    image[24:35,504] = (150,140,145)
    cv2.line(image, (510,0), (600,57), (255,255,255), 3)
    geometry = measure_bar_local_geometry(cv2.cvtColor(image,cv2.COLOR_BGR2HSV))
    results = [discover_right(image, replace(geometry,fill_endpoint_x=x), structural=True)
               for x in [None,400,503,600]]
    assert all(r == results[0] for r in results)
    assert results[0]["fillable_right_x"] == 503


def test_audit_does_not_count_invalidated_early_confirmation_as_available():
    target = EpisodeRightTarget()
    rows = [observation(), observation(2), observation(3, anchor_bbox=[110,16,357,41]),
            observation(4, fill_endpoint_x=503)]
    for row in rows:
        row["episode_target"] = target.update("one",row)
    summary = episode_audit(rows,"one",{})
    assert summary["first_confirmation_before_near_full"] is True
    assert summary["confirmed_before_near_full"] is False
    assert summary["invalidation_events"][0]["frame"] == 3


def test_all_original_episodes_structural_audit_and_safe_geometry_invalidation(capsys):
    archive = Path(__file__).resolve().parents[2] / "reports/fishing_v2/production_v3/session_20260905_032141.zip"
    if not archive.exists():
        pytest.skip("Original local anomaly archive is not distributed in Git")
    run_archive(archive,None,method="structural",summary_only=True)
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    episodes = [r for r in output if r["kind"] == "episode_summary"]
    assert len(episodes) == 10
    assert sum(r["samples"] for r in episodes) == 1169
    assert sum(r["accepted"] for r in episodes) == 715
    assert all(r["unique_fillable_right_x"] == [503] for r in episodes)
    assert all(r["right_stats"]["range"] == 0 for r in episodes)
    assert output[-1]["confirmed_before_near_full"] == 10
    assert output[-1]["confirmed_before_first_fill"] == 10
    assert output[-1]["conflicting_episodes"] == 0
    by_name = {r["episode"]:r for r in episodes}
    assert by_name["episode_35"]["invalidation_events"] == []
    assert by_name["episode_35"]["second_agreeing_frame"] == 9686
    assert by_name["episode_35"]["first_near_full_frame"] == 9705
    assert by_name["episode_35"]["confirmed_before_near_full"]
    assert by_name["episode_34"]["confirmed_before_near_full"]
    assert by_name["episode_36"]["confirmed_before_near_full"]
    distribution = next(r for r in output if r['kind'] == 'jitter_distribution')
    assert {r['episode'] for r in distribution['changed_anchor_frames']} == {'episode_34','episode_35'}
    assert len(distribution['changed_anchor_frames']) == 6
    assert distribution['signed_deltas']['anchor_left'] == dict(count=769,min=-1,median=0,p95=0,max=1)
    assert distribution['signed_deltas']['divider_x']['max'] == 0


def confirmed_target():
    target = EpisodeRightTarget()
    target.update('one', observation())
    assert target.update('one', observation(2))['status'] == 'confirmed'
    return target


def native_observation(image, frame):
    row = observation(frame)
    row.update(measure_sample(image,frame,frame*.025,method='structural')[0])
    return row


def test_one_pixel_quantization_with_pixel_support_keeps_fixed_reference_target():
    target = confirmed_target()
    image = raster()
    image[16:41,109] = 0
    row = native_observation(image,3)
    assert row['anchor_bbox'] == (110,16,357,41)
    for f in range(3,20):
        current = {**row,'frame_id':f,'timestamp':f*.025,'fillable_right_x':None}
        assert target.update('one',current)['confirmed_x'] == 503
    assert target.key[3][0] == 109  # No chained 1px drift.
    image[16:41,110] = 0
    assert target.update('one',native_observation(image,20))['status'] == 'invalidated'


@pytest.mark.parametrize('fault',['translation','divider','height','low_mask','low_rails','missing_pixels'])
def test_semantic_conflicts_still_invalidate(fault):
    target = confirmed_target()
    image = raster()
    if fault == 'translation':
        image = raster(border=505,shift=1)
    elif fault == 'divider':
        image[16:41,357] = (0,0,255)
        image[16:41,358] = 255
    elif fault == 'height':
        image[41,109:357] = (0,0,255)
    elif fault in ('low_mask','low_rails'):
        # Genuine thin connected perimeter retains a bbox only 1px narrower,
        # but cannot masquerade as high-overlap anchor support.
        image[16:41,109:357] = 0
        cv2.rectangle(image,(110,16),(356,40),(0,0,255),1)
        if fault == 'low_mask':
            image[[17,39],110:357] = (0,0,255)
    row = native_observation(image,3)
    if fault == 'missing_pixels':
        row = {**public_row(row),'anchor_bbox':(110,16,357,41)}
    assert target.update('one',row)['status'] == 'invalidated'
    assert target.update('one',observation(4))['confirmed_x'] is None


def test_new_episode_does_not_inherit_jitter_permission_or_confirmation():
    target = confirmed_target()
    image = raster(border=505,shift=1)
    assert target.update('two',native_observation(image,1))['status'] == 'unconfirmed'
    assert target.update('two',native_observation(image,2))['confirmed_x'] == 504


def test_raw_pixel_audit_reports_translation_without_mutation_or_mask_serialization():
    image = raster()
    moved = np.zeros_like(image)
    moved[:,1:] = image[:,:-1]
    before = moved.copy()
    row = native_observation(moved,2)
    audit = geometry_audit(row,moved,native_observation(image,1),image)
    assert audit['divider_patch_translation_px'] == [1,0]
    assert audit['divider_patch_translation_unique']
    assert audit['divider_screen_delta'] == audit['right_screen_delta'] == 1
    assert np.array_equal(moved,before)
    assert '_anchor_mask' not in json.loads(json.dumps(public_row(row)))


def test_real_9688_mask_boundary_and_9705_target(capsys):
    archive = Path(__file__).resolve().parents[2] / 'reports/fishing_v2/production_v3/session_20260905_032141.zip'
    if not archive.exists():
        pytest.skip('Original local anomaly archive is not distributed in Git')
    run_archive(archive,[35],method='structural')
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    frames = {r['frame_id']:r for r in rows if r['kind'] == 'frame'}
    a, b = frames[9687], frames[9688]
    assert b['anchor_bbox'] == [110,16,357,41]
    assert a['anchor_component_area'] == 5284 and b['anchor_component_area'] == 5263
    assert b['geometry_audit']['anchor_mask_overlap']['iou'] == pytest.approx(.9956480605487228)
    assert b['geometry_audit']['rail_mask_overlap']['changed_pixels'] == 0
    assert b['geometry_audit']['divider_patch_translation_px'] == [0,0]
    assert b['fillable_right_x'] is None  # Unknown is NOT an observed unchanged right.
    assert b['episode_target']['confirmed_x'] == frames[9705]['episode_target']['confirmed_x'] == 503
    assert frames[9705]['episode_target']['confirmation_frame'] == 9686
    assert frames[9684]['anchor_bbox'] is None  # No fabricated preappearance geometry.
