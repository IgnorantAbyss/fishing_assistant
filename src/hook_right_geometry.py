"""Shared passive Hook right-edge geometry; never an action authority."""
from __future__ import annotations
import math
import cv2
import numpy as np

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


class EpisodeRightTarget:
    """Bounded passive two-capture confirmation; never a runtime action latch.

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
        self.observation_count = 0
        self.invalidation_reason = None

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
            self.invalidation_reason = event
            self.invalidated = True
            self.confirmed = self.candidate = None
            self.confidence = None
        if not independent:
            # Duplicate captures cannot confirm, but must not conceal a changed
            # geometry/resolution and keep a stale target alive either.
            return self.snapshot(event or "duplicate_or_out_of_order_capture")
        if not self.invalidated and anchor and credible:
            self.observation_count += 1
            if self.candidate is not None and right != self.candidate:
                event = "conflicting_current_frame_geometry"
                self.invalidation_reason = event
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
