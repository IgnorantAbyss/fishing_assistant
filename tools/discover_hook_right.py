"""Offline right-edge diagnostic prototype; NOT a Production geometry source.

Reads original anomaly PNGs directly from ZIP, or an existing full reference
image. Writes JSONL to stdout only. No capture, runtime, action, or history-based
edge estimation. Coordinates are native ROI pixel centres (band bottom exclusive).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import zipfile
from collections import Counter
import math

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hook_bar_geometry import measure_bar_local_geometry
from src.config_loader import load_roi_config


def discover_right(crop, geometry, *, structural=False):
    """Prototype: follow paired rails, then require an interior vertical border.

    Coordinates are inclusive pixel centres, not half-open crop bounds. The
    one-pixel border is excluded from fillable_right; its inward AA pixel is
    included. Rail colour is learned from this frame, never an endpoint maximum.
    """
    result = dict(decorative_outer_right_x=None, structural_border_x=None,
                  decorative_edge_reason="outer_shadow_extent_not_established",
                  fillable_right_x=None,
                  bar_right_confidence=0.0, edge_support_rows=0, total_rows=0,
                  edge_candidates=[], rail_endpoints=[],
                  bar_right_reason="bar_local_geometry_unavailable")
    if geometry.anchor is None or geometry.divider_x is None:
        return result
    _, top, _, bottom = geometry.anchor
    divider = int(geometry.divider_x)
    height = bottom-top
    start = divider + max(6, height//2)
    width = crop.shape[1]
    seed_width = max(6, height//2)
    if start+seed_width >= width or height < 10:
        return result
    pixels = crop.astype(np.float32)
    rail_rows = [top+1, bottom-2]
    seeds = pixels[rail_rows, start:start+seed_width]
    references = np.median(seeds, axis=1)
    noise = float(np.max(np.abs(seeds-references[:, None, :])))
    result["rail_seed_bgr"] = references.tolist()
    result["rail_seed_noise"] = noise
    result["bar_right_reason"] = "rail_seed_not_uniform"
    if noise > 2:
        return result
    tolerance = max(2.0, float(np.max(np.abs(references[0]-references[1]))))
    result["rail_colour_tolerance"] = tolerance
    # Rail ends may differ by the observed one-pixel antialiased border.
    ends = []
    for row, reference in zip(rail_rows, references):
        matches = np.max(np.abs(pixels[row,start:]-reference), axis=1) <= tolerance
        misses = np.flatnonzero(~matches)
        if not len(misses):
            result["bar_right_reason"] = "track_clipped_or_border_missing"
            return result
        ends.append(start+int(misses[0]))
    result["bar_right_reason"] = "rail_endpoints_disagree"
    result["rail_endpoints"] = ends
    if abs(ends[0]-ends[1]) > 2 or min(ends) <= start+seed_width:
        return result
    rows = pixels[top+3:bottom-3]
    result["total_rows"] = len(rows)
    low, high = references.min(axis=0)-tolerance, references.max(axis=0)+tolerance
    candidates = []
    for x in range(max(start, min(ends)-1), min(width-1, max(ends)+1)):
        column = rows[:,x]
        border = np.all((column >= low) & (column <= high), axis=1)
        uniform = np.max(np.abs(column-np.median(column,axis=0)),axis=1) <= max(2,noise)
        inward = np.max(np.abs(column-rows[:,x-1]),axis=1) >= 3
        outward = np.max(np.abs(column-rows[:,x+1]),axis=1) >= 4
        support = int(np.count_nonzero(border & uniform & inward & outward))
        result["edge_candidates"].append(dict(
            x=x, support_rows=support, border_colour_rows=int(border.sum()),
            uniform_rows=int(uniform.sum()), inward_rows=int(inward.sum()),
            outward_rows=int(outward.sum())))
        result["edge_support_rows"] = max(result["edge_support_rows"], support)
        if support >= np.ceil(len(rows)*0.8):
            candidates.append((x,support))
    result["bar_right_reason"] = "right_border_missing_weak_or_conflicting"
    result["baseline_support_rows"] = result["edge_support_rows"]
    if len(candidates) != 1:
        if structural:
            return structural_fallback(pixels, geometry, start, rail_rows, ends, noise, result)
        return result
    edge, support = candidates[0]
    result.update(structural_border_x=edge, fillable_right_x=edge-1,
                  edge_support_rows=support, bar_right_confidence=support/len(rows),
                  bar_right_reason="paired_rails_and_single_pixel_border")
    return result


def structural_fallback(pixels, geometry, start, rail_rows, rail_ends, noise, result):
    """Offline hypothesis: joined rail corners + thin vertical transition.

    No absolute border colour or vertical border colour-uniformity gate.
    The first downward corner on each horizontal rail bounds the search: never
    jump past an earlier end to a background edge. A one-pixel AA corner can
    make the two ends differ by one pixel. The earliest corner owns the border.
    Both sides of that border must have contrast on >=80% of interior rows;
    the inward column must remain vertically coherent (opaque track support).
    """
    result["structural_reason"] = "horizontal_corners_unavailable"
    corners = []
    for row in rail_rows:
        # Mean BGR is used only for signed local contrast, never a colour class.
        values = pixels[row].mean(axis=1)
        falling = np.flatnonzero(values[start-1:-1]-values[start:] >= max(3, noise+1))
        if not len(falling):
            return result
        corners.append(start+int(falling[0]))
    result["structural_corners"] = corners
    result["structural_reason"] = "horizontal_corners_disagree"
    if abs(corners[0]-corners[1]) > 1:
        return result
    edge = min(corners)
    if any(abs(edge-end) > 1 for end in rail_ends):
        result["structural_reason"] = "corner_does_not_join_track_ends"
        return result
    _, top, _, bottom = geometry.anchor
    rows = pixels[top+3:bottom-3]
    if edge <= start or edge+1 >= pixels.shape[1]:
        return result
    inner, border, outer = rows[:,edge-1], rows[:,edge], rows[:,edge+1]
    inner_uniform = np.max(np.abs(inner-np.median(inner, axis=0)), axis=1) <= max(2, noise)
    inward = np.max(np.abs(border-inner), axis=1) >= 3
    outward = np.max(np.abs(border-outer), axis=1) >= 4
    support = int(np.count_nonzero(inner_uniform & inward & outward))
    result["structural_transition"] = dict(
        x=edge, inner_uniform_rows=int(inner_uniform.sum()),
        inward_rows=int(inward.sum()), outward_rows=int(outward.sum()), support_rows=support,
        border_bgr_min=border.min(axis=0).tolist(), border_bgr_max=border.max(axis=0).tolist())
    result["structural_reason"] = "insufficient_joined_vertical_transition"
    if support < np.ceil(len(rows)*0.8):
        return result
    result.update(structural_border_x=edge, fillable_right_x=edge-1,
                  edge_support_rows=support, bar_right_confidence=support/len(rows),
                  structural_reason="joined_corners_and_inner_transition",
                  bar_right_reason="structural_fallback_joined_track_edges")
    return result


def anchor_pixels(hsv, anchor):
    """Audit the EXACT canonical red component, not an alternative locator.

    Mirror its unchanged pixel predicate, then require its already-selected bbox.
    Packed masks are private, bounded native-coordinate comparison data. The two
    rail rows exclude the animated hatch interior; no resizing/alignment is used.
    """
    if anchor is None:
        return {}
    hue, saturation, value = cv2.split(hsv)
    red = (((hue <= 10) | (hue >= 165)) & (saturation >= 80) & (value >= 80)).astype(np.uint8)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(red, connectivity=8)
    matches = [i for i, (x,y,w,h,_) in enumerate(stats[1:], 1)
               if (x,y,x+w,y+h) == tuple(anchor)]
    if len(matches) != 1:
        return {}
    i = matches[0]
    mask = labels == i
    rails = np.zeros_like(mask)
    rail_rows = [anchor[1]+1, anchor[3]-2]
    rails[rail_rows] = mask[rail_rows]
    ys, xs = np.nonzero(rails)
    return dict(anchor_component_area=int(stats[i,4]), anchor_centroid=centroids[i].tolist(),
                anchor_width=anchor[2]-anchor[0], anchor_height=anchor[3]-anchor[1],
                track_top_edge_y=rail_rows[0], track_bottom_edge_y=rail_rows[1],
                anchor_rail_centroid=[float(xs.mean()), float(ys.mean())] if len(xs) else None,
                _mask_bbox=tuple(anchor), _anchor_mask=np.packbits(mask).tobytes(),
                _anchor_rails=np.packbits(rails).tobytes())


def mask_comparison(a, b):
    if not a or not b or len(a) != len(b):
        return None
    a, b = np.unpackbits(np.frombuffer(a,np.uint8)), np.unpackbits(np.frombuffer(b,np.uint8))
    union = int(np.count_nonzero(a | b))
    return dict(iou=float(np.count_nonzero(a & b)/union) if union else 0.0,
                changed_pixels=int(np.count_nonzero(a != b)))


def public_row(row):
    return {k:v for k,v in row.items() if not k.startswith('_')}


def measure_sample(crop, frame_id, timestamp, *, canonical_band_height=32, method="baseline"):
    if crop is None or not crop.size:
        raise ValueError("Original ROI could not be decoded")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    geometry = measure_bar_local_geometry(
        hsv,
        canonical_band_height=canonical_band_height)
    start = perf_counter()
    edge = discover_right(crop, geometry, structural=method == "structural")
    cost_ms = (perf_counter()-start)*1000
    row = dict(kind="frame", status="diagnostic_prototype", frame_id=frame_id,
               timestamp=timestamp, coordinate_space="native_roi_pixel_centres",
               roi_dimensions=[crop.shape[1], crop.shape[0]],
               anchor_bbox=geometry.anchor,
               bar_local_y_top=geometry.anchor[1] if geometry.anchor else None,
               bar_local_y_bottom=geometry.anchor[3] if geometry.anchor else None,
               band_bottom_exclusive=True, divider_x=geometry.divider_x,
               divider_confidence=geometry.divider_confidence,
               fill_endpoint_x=geometry.fill_endpoint_x, **edge)
    row.update(anchor_pixels(hsv, geometry.anchor))
    right = edge["fillable_right_x"]
    row["distance_to_right"] = (
        right-geometry.fill_endpoint_x
        if right is not None and geometry.fill_endpoint_x is not None else None)
    return row, cost_ms


def summarize(rows, episode):
    values = [r["fillable_right_x"] for r in rows if r["fillable_right_x"] is not None]
    return dict(kind="episode_summary", episode=episode, accepted=len(values),
                samples=len(rows), divider_available=sum(r["divider_x"] is not None for r in rows),
                abstain_reasons=dict(Counter(r["bar_right_reason"] for r in rows
                                            if r["fillable_right_x"] is None)),
                right_stats=dict(min=min(values), median=float(np.median(values)),
                                 max=max(values), range=max(values)-min(values),
                                 stddev=float(np.std(values))) if values else None)


def geometry_audit(row, crop, previous, previous_crop):
    """Adjacent raw-pixel diagnostics, never input to right-edge discovery.

    Translation is integer SSD matching of a divider-centred patch over +/-2
    pixels, not a whole-image/background registration or a subpixel claim.
    Missing current right remains unknown, even when a target is retained.
    """
    result = dict(anchor_bbox_iou=None, anchor_mask_overlap=None, rail_mask_overlap=None,
                  centroid_delta=None, divider_patch_translation_px=None,
                  divider_patch_translation_unique=None, deltas={})
    a, b = row.get('anchor_bbox'), previous.get('anchor_bbox') if previous else None
    if not a or not b or previous_crop.shape != crop.shape:
        return result
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    result['anchor_bbox_iou'] = intersection / (
        (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)
    result['anchor_mask_overlap'] = mask_comparison(row.get('_anchor_mask'), previous.get('_anchor_mask'))
    result['rail_mask_overlap'] = mask_comparison(row.get('_anchor_rails'), previous.get('_anchor_rails'))
    if row.get('anchor_centroid') and previous.get('anchor_centroid'):
        result['centroid_delta'] = (np.array(row['anchor_centroid'])-previous['anchor_centroid']).tolist()
    for key in ('anchor_width','anchor_height','divider_x','fillable_right_x',
                'bar_local_y_top','bar_local_y_bottom','track_top_edge_y','track_bottom_edge_y'):
        result['deltas'][key] = (row[key]-previous[key]
                                if row.get(key) is not None and previous.get(key) is not None else None)
    result['deltas'].update(anchor_left=a[0]-b[0], anchor_top=a[1]-b[1])
    result['divider_screen_delta'] = result['deltas']['divider_x']
    result['right_screen_delta'] = result['deltas']['fillable_right_x']
    d = row.get('divider_x')
    if d is not None and a[1] >= 2 and a[3]+2 <= crop.shape[0] and 6 <= d < crop.shape[1]-6:
        x = int(d)
        template = crop[a[1]:a[3],x-4:x+5].astype(np.float32)
        search = previous_crop[a[1]-2:a[3]+2,x-6:x+7].astype(np.float32)
        scores = cv2.matchTemplate(search,template,cv2.TM_SQDIFF)
        y, x = np.unravel_index(scores.argmin(),scores.shape)
        result['divider_patch_translation_px'] = [2-int(x),2-int(y)]
        result['divider_patch_translation_unique'] = int(np.count_nonzero(scores == scores.min())) == 1
        result['divider_patch_ssd'] = float(scores.min())
    return result


def jitter_distribution(rows):
    deltas = {}
    changed = []
    for row in rows:
        audit = row.get('geometry_audit', {})
        for key, value in audit.get('deltas', {}).items():
            if value is not None:
                deltas.setdefault(key, []).append(value)
        if audit.get('deltas', {}).get('anchor_left'):
            changed.append(dict(episode=row.get('episode'), frame=row['frame_id'],
                                delta=audit['deltas']['anchor_left']))
    return dict(kind='jitter_distribution', scope='adjacent_canonical_frames_no_cross_episode_pairs',
                changed_anchor_frames=changed,
                signed_deltas={k:dict(count=len(v), **dict(zip(
                    ('min','median','p95','max'),np.percentile(v,[0,50,95,100]).tolist())))
                    for k,v in deltas.items()})


class EpisodeRightTarget:
    """Bounded OFFLINE two-capture confirmation; never a runtime action latch.

    Divider, band, ROI and credible right remain exact. Sep05's 769 adjacent
    canonical pairs have left/width delta <=1, all other geometry deltas zero.
    Six left changes occur in episodes 34/35. Fixed rail masks differ by at most
    two pixels (IoU >=.9959677), including versus the first anchor, while animated
    interior masks can differ by 28.06% between captures. Therefore accept ONLY
    that left-edge quantization with fixed-reference rail XOR <=2 and full
    adjacent mask IoU >=.71 (observed minimum .719413 rounded DOWN to 2 decimals).
    These are explicit OFFLINE evidence-derived bounds, not Production tuning.
    Fixed reference bbox/rails never drift. Full centroid is telemetry only:
    hatch animation moves it by 28.76px without moving the frame.
    A credible conflict or loss/change of geometry invalidates this episode
    permanently. Missing right-edge evidence alone may bridge a gap, provided
    current anchor/divider/ROI/resolution still match. Only a NEW physical
    episode resets invalidation. Stores at most one candidate and one target.
    """
    def __init__(self):
        self.episode = None
        self._reset()

    def _reset(self):
        self.key = None
        self.candidate = None
        self.candidate_confidence = None
        self.confirmed = None
        self.confidence = None
        self.confirmation_frame = None
        self.confirmation_timestamp = None
        self.invalidated = False
        self.last_frame = None
        self.last_timestamp = None
        self.reference_rails = None
        self.previous_mask = None

    def compatible(self, key, row):
        if self.key is None:
            return True
        # All geometry except the red component's left edge is exact.
        if key[:3] != self.key[:3] or key[4:] != self.key[4:]:
            return False
        anchor, reference = key[3], self.key[3]
        if row.get('_mask_bbox') is not None and tuple(row['_mask_bbox']) != anchor:
            return False
        if anchor[1:] != reference[1:] or abs(anchor[0]-reference[0]) > 1:
            return False
        rails = mask_comparison(self.reference_rails, row.get('_anchor_rails'))
        full = mask_comparison(self.previous_mask, row.get('_anchor_mask'))
        # No pixel support => cannot excuse even a one-pixel bbox change.
        if rails is None or full is None:
            return key == self.key
        return rails['changed_pixels'] <= 2 and rails['iou'] >= .995 and full['iou'] >= .71

    def update(self, episode, row):
        if episode != self.episode:
            self._reset()
            self.episode = episode
        event = None
        frame, timestamp = row["frame_id"], row["timestamp"]
        independent = not (self.last_frame is not None and (
            frame <= self.last_frame or (
                timestamp is not None and self.last_timestamp is not None
                and timestamp <= self.last_timestamp)))
        if independent:
            self.last_frame, self.last_timestamp = frame, timestamp
        anchor, divider = row.get("anchor_bbox"), row.get("divider_x")
        if not anchor or divider is None:
            if self.key is not None and not self.invalidated:
                event = "geometry_unavailable"
        else:
            key = (tuple(row["roi_dimensions"]), tuple(row.get("resolution") or ()),
                   tuple(row.get("roi_bounds") or ()), tuple(anchor), divider,
                   row["bar_local_y_top"], row["bar_local_y_bottom"])
            if self.key is None or (self.candidate is None and not self.invalidated):
                self.key = key
                self.reference_rails = row.get('_anchor_rails')
            elif not self.compatible(key, row) and not self.invalidated:
                event = "geometry_changed"
        if independent:
            self.previous_mask = row.get('_anchor_mask')
        right, confidence = row.get("fillable_right_x"), row.get("bar_right_confidence", 0)
        credible = (right is not None and math.isfinite(right) and divider is not None
                    and divider < right < row["roi_dimensions"][0]-1
                    and math.isfinite(confidence) and 0.8 <= confidence <= 1)
        if event:
            self.invalidated = True
            self.confirmed = self.candidate = None
            self.confidence = None
        if not independent:
            # Duplicate captures cannot confirm, but must not conceal a changed
            # geometry/resolution and keep a stale target alive either.
            return self.snapshot(event or "duplicate_or_out_of_order_capture")
        if not self.invalidated and anchor and credible:
            if self.candidate is not None and right != self.candidate:
                event = "conflicting_current_frame_geometry"
                self.invalidated = True
                self.confirmed = self.candidate = None
                self.confidence = None
            elif self.candidate is None:
                self.candidate, self.candidate_confidence = right, confidence
            elif self.confirmed is None:
                self.confirmed = self.candidate
                self.confidence = min(self.candidate_confidence, confidence)
                self.confirmation_frame, self.confirmation_timestamp = frame, timestamp
                event = "two_independent_geometry_detections_agree"
        return self.snapshot(event)

    def snapshot(self, event):
        return dict(status="invalidated" if self.invalidated else (
                        "confirmed" if self.confirmed is not None else "unconfirmed"),
                    confirmed_x=self.confirmed, confidence=self.confidence,
                    confirmation_frame=self.confirmation_frame,
                    confirmation_timestamp=self.confirmation_timestamp, event=event)


def episode_audit(rows, episode, manifest):
    """Near-full is an OFFLINE observation band, not a proposed action trigger.

    Last 10% of divider-to-border span; also report equality and the stronger
    before-first-fill check so availability does not depend on this band choice.
    Retrospective scoring may use a unique border-derived x; never feeds latch.
    """
    result = summarize(rows, episode)
    valid = [r for r in rows if r["fillable_right_x"] is not None]
    unique = sorted({r["fillable_right_x"] for r in valid})
    reference = unique[0] if len(unique) == 1 else None
    confirmation = next((r for r in rows if r["episode_target"]["event"] ==
                         "two_independent_geometry_detections_agree"), None)
    filled = [r for r in rows if r["fill_endpoint_x"] is not None and r["divider_x"] is not None]
    near = next((r for r in filled if reference is not None and
                 0 <= reference-r["fill_endpoint_x"] <= .1*(reference-r["divider_x"])), None)
    equal = next((r for r in filled if r["fill_endpoint_x"] == reference), None)
    invalidations = [dict(frame=r["frame_id"], timestamp=r["timestamp"],
                          reason=r["episode_target"]["event"]) for r in rows
                     if r["episode_target"]["event"] in (
                         "geometry_changed", "geometry_unavailable", "conflicting_current_frame_geometry")]
    before = lambda r: (confirmation["timestamp"] < r["timestamp"]
                        if confirmation is not None and r is not None else None)
    available = lambda r: (bool(before(r)) and r["episode_target"]["status"] == "confirmed"
                           if r is not None else None)
    gap = None
    if confirmation:
        times = [r["timestamp"] for r in valid if r["timestamp"] >= confirmation["timestamp"]]
        times.append(rows[-1]["timestamp"])
        gap = max((b-a for a,b in zip(times,times[1:])), default=0)
    started = manifest.get("start_hook_emission_completed_at")
    result.update(
        valid_ratio=len(valid)/result["divider_available"] if result["divider_available"] else None,
        unique_fillable_right_x=unique, first_valid_frame=valid[0]["frame_id"] if valid else None,
        second_agreeing_frame=confirmation["frame_id"] if confirmation else None,
        confirmation_timestamp=confirmation["timestamp"] if confirmation else None,
        confirmation_latency_from_start_hook_completed_seconds=(
            confirmation["timestamp"]-started if confirmation and started is not None else None),
        confirmed_x=confirmation["episode_target"]["confirmed_x"] if confirmation else None,
        confirmed_confidence=confirmation["episode_target"]["confidence"] if confirmation else None,
        agreement_tolerance_px=0, near_full_definition="last_10_percent_of_divider_to_border_span_offline_only",
        first_near_full_frame=near["frame_id"] if near else None,
        first_equal_endpoint_frame=equal["frame_id"] if equal else None,
        first_fill_frame=filled[0]["frame_id"] if filled else None,
        first_confirmation_before_near_full=before(near),
        confirmed_before_near_full=available(near),
        confirmed_before_first_fill=available(filled[0] if filled else None),
        longest_observation_gap_after_confirmation_seconds=gap,
        conflicting_detections=len(unique)>1, invalidation_events=invalidations,
        final_status=rows[-1]["episode_target"]["status"] if rows else "no_samples")
    return result


def run(archive_path, episodes, references=(), *, method="baseline", summary_only=False):
    timings = []
    if archive_path:
        timings.extend(run_archive(archive_path, episodes, method=method, summary_only=summary_only))
    for path in references:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read original reference: {path}")
        height, width = image.shape[:2]
        left, top, right, bottom = load_roi_config().pixel_roi("hook_bar_precise", width, height)
        row, cost = measure_sample(image[top:bottom,left:right], path.name, None,
                                   canonical_band_height=max(16, round(height*32/1440)), method=method)
        row["reference_path"] = str(path)
        row["roi_origin"] = [left, top]
        print(json.dumps(public_row(row)))
        timings.append(cost)
    if not timings:
        raise ValueError("No matching raw evidence samples")
    print(json.dumps(dict(kind="cost_summary", samples=len(timings),
                          scope="discover_right_only_excludes_decode_canonical_geometry_and_stdout",
                          cost_ms={key:float(value) for key,value in zip(
                              ('mean','median','p95','max'),
                              (np.mean(timings),np.median(timings),np.percentile(timings,95),max(timings)))})))


def run_archive(archive_path, episodes, *, method="baseline", summary_only=False):
    timings = []
    summaries = []
    audit_rows = []
    with zipfile.ZipFile(archive_path) as archive:
        session_names = [n for n in archive.namelist() if n.endswith('/session_summary.json')]
        capture = (json.loads(archive.read(session_names[0])).get("capture_diagnostics", {})
                   if len(session_names) == 1 else {})
        resolution = capture.get("client_size")
        roi_bounds = capture.get("hook_roi_capture", {}).get("bounds")
        for name in sorted(archive.namelist()):
            if not name.endswith('/manifest.json') or '/hook_pending_anomalies/' not in name:
                continue
            episode = Path(name).parent.name
            if episodes and int(episode.split('_')[-1]) not in episodes:
                continue
            manifest = json.loads(archive.read(name))
            rows = []
            target = EpisodeRightTarget()
            previous = previous_crop = None
            for sample in manifest['samples']:
                raw = archive.read((Path(name).parent/sample['raw_image_filename']).as_posix())
                crop = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                row, cost = measure_sample(crop, sample['frame_index'], sample['monotonic_timestamp'], method=method)
                row['episode'] = episode
                row['method'] = method
                row['resolution'], row['roi_bounds'] = resolution, roi_bounds
                row['geometry_audit'] = geometry_audit(row, crop, previous, previous_crop)
                row['episode_target'] = target.update(episode, row)
                previous, previous_crop = row, crop
                rows.append(row)
                timings.append(cost)
                if not summary_only:
                    print(json.dumps(public_row(row)))
            summary = episode_audit(rows, episode, manifest)
            summaries.append(summary)
            audit_rows.extend(public_row(row) for row in rows)
            print(json.dumps(summary))
    print(json.dumps(jitter_distribution(audit_rows)))
    print(json.dumps(dict(kind="availability_summary", method=method, episodes=len(summaries),
                          confirmed_before_near_full=sum(s["confirmed_before_near_full"] is True
                                                         for s in summaries),
                          confirmed_before_first_fill=sum(s["confirmed_before_first_fill"] is True
                                                          for s in summaries),
                          conflicting_episodes=sum(s["conflicting_detections"] for s in summaries))))
    return timings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--episode', type=int, action='append')
    parser.add_argument('--reference', type=Path, action='append', default=[],
                        help='Original full frame; uses existing configured precise ROI')
    parser.add_argument('--method', choices=('baseline', 'structural'), default='baseline')
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()
    if not args.archive and not args.reference:
        parser.error('Provide --archive and/or --reference')
    run(args.archive,args.episode,args.reference,method=args.method,summary_only=args.summary_only)
