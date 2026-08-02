"""PRESS panel completeness proof, independent from sequence confidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.fishing_v2.domain.observations import PressObservation


VALID_KEYS = frozenset("WASD")


@dataclass(frozen=True)
class PressCompletenessConfig:
    occupancy_stability_frames: int = 3
    sequence_stability_frames: int = 2
    classification_min_confidence: float = 0.55
    panel_geometry_tolerance: float = 0.12


@dataclass(frozen=True)
class PressCompletenessCertificate:
    panel_confirmed: bool
    panel_phase: str
    panel_bbox_stable: bool
    slot_capacity: int
    occupied_mask: tuple[bool, ...]
    occupied_count: int
    decoded_mask: tuple[bool, ...]
    decoded_count: int
    ambiguous_mask: tuple[bool, ...]
    possible_occupied_mask: tuple[bool, ...]
    sequence: tuple[str, ...]
    per_slot_classification: tuple[Mapping[str, Any], ...]
    occupancy_stability_count: int
    sequence_stability_count: int
    observation_window_duration: float
    latest_observation_age: float
    complete: bool
    rejection_reasons: tuple[str, ...]
    consensus_frame_indices: tuple[int, ...] = ()
    selected_clean_frame: int | None = None
    completeness_confidence: float = 0.0
    classification_confidence: float = 0.0

    def payload(self) -> dict[str, Any]:
        return {
            "panel_confirmed": self.panel_confirmed,
            "panel_phase": self.panel_phase,
            "panel_bbox_stable": self.panel_bbox_stable,
            "slot_capacity": self.slot_capacity,
            "occupied_mask": list(self.occupied_mask),
            "occupied_count": self.occupied_count,
            "decoded_mask": list(self.decoded_mask),
            "decoded_count": self.decoded_count,
            "ambiguous_mask": list(self.ambiguous_mask),
            "possible_occupied_mask": list(self.possible_occupied_mask),
            "sequence": list(self.sequence),
            "per_slot_classification": [dict(item) for item in self.per_slot_classification],
            "occupancy_stability_count": self.occupancy_stability_count,
            "sequence_stability_count": self.sequence_stability_count,
            "observation_window_duration": self.observation_window_duration,
            "observation_window_duration_ms": round(self.observation_window_duration * 1000.0, 3),
            "latest_observation_age": self.latest_observation_age,
            "complete": self.complete,
            "rejection_reasons": list(self.rejection_reasons),
            "consensus_frame_indices": list(self.consensus_frame_indices),
            "selected_clean_frame": self.selected_clean_frame,
            "completeness_confidence": self.completeness_confidence,
            "classification_confidence": self.classification_confidence,
        }


def _slots(observation: PressObservation) -> list[dict[str, Any]]:
    raw = observation.evidence.get("slots")
    if isinstance(raw, (list, tuple)) and raw:
        return [dict(item) for item in raw if isinstance(item, Mapping)]

    # Compatibility for synthetic and legacy observations. A declared capacity
    # larger than the decoded boxes is deliberately represented as unknown,
    # never as confidently empty.
    boxes = observation.evidence.get("key_boxes", ())
    usable = [dict(item) for item in boxes if isinstance(item, Mapping)]
    declared = observation.evidence.get("total_slot_count")
    capacity = int(declared) if isinstance(declared, int) and not isinstance(declared, bool) else len(usable)
    slots: list[dict[str, Any]] = []
    for index in range(max(0, capacity)):
        if index < len(usable):
            box = usable[index]
            slots.append({
                "index": index,
                "bbox": box.get("bbox"),
                "occupancy": "OCCUPIED",
                "occupancy_confidence": 1.0,
                "mapped_key": box.get("key"),
                "arrow_confidence": box.get("confidence", 0.0),
                "coloured_pixel_count": 1,
            })
        else:
            slots.append({
                "index": index,
                "occupancy": "UNCERTAIN",
                "occupancy_confidence": 0.0,
                "mapped_key": None,
                "arrow_confidence": 0.0,
                "possible_occupied": True,
            })
    return slots


def _snapshot(
    observation: PressObservation,
    config: PressCompletenessConfig,
) -> dict[str, Any]:
    slots = _slots(observation)
    occupied: list[bool] = []
    decoded: list[bool] = []
    ambiguous: list[bool] = []
    possible: list[bool] = []
    sequence: list[str] = []
    per_slot: list[dict[str, Any]] = []
    confidences: list[float] = []
    candidate_sequence = tuple(observation.sequence_candidate)
    for fallback_index, slot in enumerate(slots):
        index = int(slot.get("index", fallback_index))
        occupancy = str(slot.get("occupancy", "UNCERTAIN")).upper()
        raw_key = slot.get("mapped_key", slot.get("key", ""))
        if not raw_key and fallback_index < len(candidate_sequence):
            raw_key = candidate_sequence[fallback_index]
        key = str(raw_key).upper()
        confidence = float(slot.get("arrow_confidence", slot.get("confidence", 0.0)) or 0.0)
        is_occupied = occupancy == "OCCUPIED"
        selected_letter = slot.get("selected_letter_component")
        has_component_evidence = bool(
            slot.get("possible_occupied") is True
            or (
                isinstance(selected_letter, Mapping)
                and selected_letter.get("structural", True) is True
            )
            or isinstance(slot.get("combined_input_effect_component"), Mapping)
            or int(slot.get("arrow_pixel_count", 0) or 0) > 0
            or any(
                isinstance(item, Mapping) and item.get("eligible") is True
                for item in slot.get("arrow_component_candidates", ())
            )
        )
        is_possible = bool(
            has_component_evidence
            or (is_occupied and key in VALID_KEYS)
        )
        is_decoded = bool(
            is_occupied
            and key in VALID_KEYS
            and confidence >= config.classification_min_confidence
        )
        is_ambiguous = bool(occupancy == "UNCERTAIN" or (is_occupied and not is_decoded))
        occupied.append(is_occupied)
        decoded.append(is_decoded)
        ambiguous.append(is_ambiguous)
        possible.append(is_possible)
        if is_decoded:
            sequence.append(key)
            confidences.append(confidence)
        per_slot.append({
            "index": index,
            "occupancy": occupancy,
            "occupancy_confidence": float(slot.get("occupancy_confidence", 0.0) or 0.0),
            "possible_occupied": is_possible,
            "decoded": is_decoded,
            "ambiguous": is_ambiguous,
            "key": key if key in VALID_KEYS else None,
            "classification_confidence": confidence,
            "bbox": slot.get("bbox"),
            "letter_pixel_count": int(slot.get("letter_pixel_count", 0) or 0),
            "arrow_pixel_count": int(slot.get("arrow_pixel_count", 0) or 0),
            "coloured_pixel_count": int(slot.get("coloured_pixel_count", 0) or 0),
        })
    return {
        "observation": observation,
        "slots": per_slot,
        "occupied": tuple(occupied),
        "decoded": tuple(decoded),
        "ambiguous": tuple(ambiguous),
        "possible": tuple(possible),
        "sequence": tuple(sequence),
        "classification_confidence": min(confidences) if confidences else 0.0,
    }


def _geometry_stable(
    snapshots: Sequence[dict[str, Any]],
    tolerance: float,
) -> bool:
    if not snapshots:
        return False
    observations = [item["observation"] for item in snapshots]
    panels = [item.evidence.get("panel_bbox") for item in observations]
    if any(not isinstance(panel, (list, tuple)) or len(panel) != 4 for panel in panels):
        # Synthetic observations without geometry remain testable only when
        # they do not claim a larger unseen slot capacity.
        return all(len(item["slots"]) == len(item["occupied"]) for item in snapshots)
    reference = tuple(float(value) for value in panels[-1])
    ref_w = max(1.0, reference[2] - reference[0])
    ref_h = max(1.0, reference[3] - reference[1])
    reference_centres: tuple[float, ...] | None = None
    reference_slots = snapshots[-1]["slots"]
    if all(
        isinstance(item.get("bbox"), (list, tuple))
        and len(item["bbox"]) == 4
        for item in reference_slots
    ):
        reference_centres = tuple(
            (
                (float(item["bbox"][0]) + float(item["bbox"][2])) * 0.5
                - reference[0]
            ) / ref_w
            for item in reference_slots
        )
    for snapshot, panel in zip(snapshots, panels, strict=True):
        current = tuple(float(value) for value in panel)
        deltas = (
            abs(current[0] - reference[0]) / ref_w,
            abs(current[1] - reference[1]) / ref_h,
            abs(current[2] - reference[2]) / ref_w,
            abs(current[3] - reference[3]) / ref_h,
        )
        if max(deltas) > tolerance:
            return False
        if reference_centres is not None:
            slots = snapshot["slots"]
            if len(slots) != len(reference_centres) or any(
                not isinstance(item.get("bbox"), (list, tuple))
                or len(item["bbox"]) != 4
                for item in slots
            ):
                return False
            width = max(1.0, current[2] - current[0])
            centres = tuple(
                (
                    (float(item["bbox"][0]) + float(item["bbox"][2])) * 0.5
                    - current[0]
                ) / width
                for item in slots
            )
            if max(
                (abs(actual - expected) for actual, expected in zip(
                    centres, reference_centres, strict=True
                )),
                default=0.0,
            ) > tolerance:
                return False
    return True


def evaluate_press_completeness(
    observation: PressObservation,
    history: Sequence[PressObservation],
    *,
    panel_confirmed: bool,
    config: PressCompletenessConfig | None = None,
) -> PressCompletenessCertificate:
    """Prove that every occupied slot is decoded and every trailing slot is empty."""
    active = config or PressCompletenessConfig()
    observed_current = _snapshot(observation, active)
    clean_observations = [
        item for item in history
        if item.panel_present
        and item.evidence.get("input_effect_detected") is not True
        and item.evidence.get("clean_frame_eligible", True) is True
    ]
    snapshots = [_snapshot(item, active) for item in clean_observations]
    # Preserve the earliest frame with the highest decoded coverage.  Later
    # animation/dropout frames may provide tail-empty support, but may not erase
    # a better same-episode sequence candidate.  A different sequence only
    # replaces it after that sequence itself reaches stronger stable support.
    non_empty_candidates = [item for item in snapshots if item["sequence"]]
    candidate_support: dict[tuple[tuple[str, ...], tuple[bool, ...]], int] = {}
    for item in non_empty_candidates:
        identity = (item["sequence"], item["decoded"])
        candidate_support[identity] = candidate_support.get(identity, 0) + 1
    current = max(
        non_empty_candidates,
        key=lambda item: (
            sum(item["decoded"]),
            candidate_support[(item["sequence"], item["decoded"])],
            -int(item["observation"].frame_index),
        ),
        default=observed_current,
    )
    same_occupancy = [
        item for item in snapshots
        if item["occupied"] == current["occupied"]
    ]
    occupancy_support = same_occupancy[-active.occupancy_stability_frames:]
    occupancy_stability_count = len(occupancy_support)
    same_sequence = [
        item for item in same_occupancy
        if current["sequence"]
        and item["sequence"] == current["sequence"]
        and item["decoded"] == current["decoded"]
    ]
    sequence_stability_count = len(same_sequence) if current["sequence"] else 0
    consensus_items = same_sequence
    consensus_frames = tuple(
        int(item["observation"].frame_index) for item in consensus_items
    )
    stable_geometry = _geometry_stable(occupancy_support, active.panel_geometry_tolerance)

    occupied_indices = [index for index, value in enumerate(current["occupied"]) if value]
    contiguous = occupied_indices == list(range(len(occupied_indices)))
    last_occupied = occupied_indices[-1] if occupied_indices else -1
    trailing_possible = any(current["possible"][last_occupied + 1:])
    occupied_count = sum(current["occupied"])
    decoded_count = sum(current["decoded"])
    reasons: list[str] = []
    if not panel_confirmed:
        reasons.append("panel_not_confirmed")
    if not stable_geometry:
        reasons.append("panel_or_slot_geometry_not_stable")
    if occupancy_stability_count < active.occupancy_stability_frames:
        reasons.append("occupied_mask_not_stable")
    if occupied_count <= 0:
        reasons.append("no_occupied_slots")
    if decoded_count != occupied_count:
        reasons.append("occupied_slots_not_fully_decoded")
    if any(current["ambiguous"]):
        reasons.append("ambiguous_slot_present")
    if not contiguous or observation.evidence.get("layout_conflict") is True:
        reasons.append("occupied_slots_not_contiguous")
    if trailing_possible:
        reasons.append("possible_occupied_slot_after_decoded_sequence")
    if sequence_stability_count < active.sequence_stability_frames:
        reasons.append("complete_sequence_consensus_pending")

    timestamps = [float(item["observation"].timestamp) for item in occupancy_support]
    duration = max(timestamps) - min(timestamps) if timestamps else 0.0
    occupancy_confidences = [
        float(item["occupancy_confidence"])
        for item in current["slots"]
        if item["occupancy"] == "OCCUPIED"
    ]
    occupancy_confidence = min(occupancy_confidences) if occupancy_confidences else 0.0
    stability_confidence = min(
        1.0,
        occupancy_stability_count / active.occupancy_stability_frames,
        sequence_stability_count / active.sequence_stability_frames,
    )
    decoded_coverage = (
        decoded_count / occupied_count if occupied_count > 0 else 0.0
    )
    ambiguity_quality = 0.0 if any(current["ambiguous"]) or trailing_possible else 1.0
    completeness_confidence = min(
        float(observation.confidence),
        occupancy_confidence,
        stability_confidence,
        decoded_coverage,
        ambiguity_quality,
    ) if occupied_count else 0.0
    return PressCompletenessCertificate(
        panel_confirmed=panel_confirmed,
        panel_phase=str(observation.evidence.get("panel_phase", "UNKNOWN")),
        panel_bbox_stable=stable_geometry,
        slot_capacity=len(current["slots"]),
        occupied_mask=current["occupied"],
        occupied_count=occupied_count,
        decoded_mask=current["decoded"],
        decoded_count=decoded_count,
        ambiguous_mask=current["ambiguous"],
        possible_occupied_mask=current["possible"],
        sequence=current["sequence"],
        per_slot_classification=tuple(current["slots"]),
        occupancy_stability_count=occupancy_stability_count,
        sequence_stability_count=sequence_stability_count,
        observation_window_duration=round(duration, 6),
        latest_observation_age=0.0,
        complete=not reasons,
        rejection_reasons=tuple(reasons),
        consensus_frame_indices=consensus_frames,
        selected_clean_frame=(consensus_frames[0] if consensus_frames else None),
        completeness_confidence=round(completeness_confidence, 4),
        classification_confidence=round(float(current["classification_confidence"]), 4),
    )
