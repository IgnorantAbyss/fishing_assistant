"""Read-only raw anomaly replay. No capture, ActionSink, or report writes."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_roi_config
from src.detectors.hook_detector import detect_hook_bar
from src.fishing_v2.domain.action_intent import ActionIntent
from src.fishing_v2.domain.frame_context import FrameContext
from src.fishing_v2.domain.runtime_state import RuntimeState
from src.fishing_v2.fusion.observation_fusion import ObservationFusion
from src.fishing_v2.legacy_adapters.hook_detector_adapter import LegacyHookDetectorAdapter
from src.fishing_v2.perception.observation_bundle import ObservationBundle
from src.fishing_v2.runtime.fishing_fsm import FishingFSM, FSMConfig
from src.fishing_v2.runtime.runtime_controller import RuntimeController, ActionExecutionMode
from src.fishing_v2.runtime.safety_policy import SafetyPolicy, SafetyConfig


def baseline_module(ref, relative):
    """Load selected tracked code in memory; never check out or write a file."""
    code = subprocess.check_output(["git", "show", f"{ref}:{relative}"], cwd=ROOT)
    name = "_hook_replay_baseline_" + Path(relative).stem
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(ROOT / relative)
    sys.modules[name] = module
    exec(compile(code, module.__file__, "exec"), module.__dict__)
    return module


def run(session, baseline):
    archive = zipfile.ZipFile(session) if session.is_file() else None
    try:
        manifests = (sorted(n for n in archive.namelist() if "/hook_pending_anomalies/" in n and n.endswith("/manifest.json"))
                     if archive else sorted(session.glob("hook_pending_anomalies/*/manifest.json")))
        if not manifests:
            raise ValueError("No raw hook pending anomaly manifests available")
        old = baseline_module(baseline, "src/detectors/hook_detector.py")
        old_crossing = baseline_module(baseline, "src/fishing_v2/runtime/hook_crossing_geometry.py")
        bounds = load_roi_config().pixel_roi("hook_bar_precise", 2560, 1440)
        image = np.zeros((1440, 2560, 3), np.uint8)
        adapter = LegacyHookDetectorAdapter()
        for manifest_path in manifests:
            manifest = json.loads(archive.read(manifest_path) if archive else manifest_path.read_text(encoding="utf-8"))
            samples = manifest["samples"]
            controller = RuntimeController(ObservationFusion(), FishingFSM(FSMConfig(stable_frames=2), initial_state=RuntimeState.HOOK_PENDING, initial_timestamp=samples[0]["monotonic_timestamp"]), SafetyPolicy(SafetyConfig(emit_actions=False)))
            rows = []
            elapsed_old, elapsed_new = [], []
            for sample in samples:
                raw_path = Path(manifest_path).parent / sample["raw_image_filename"]
                crop = (cv2.imdecode(np.frombuffer(archive.read(raw_path.as_posix()), np.uint8), cv2.IMREAD_COLOR) if archive else cv2.imread(str(raw_path)))
                x1,y1,x2,y2 = bounds
                if crop.shape != (y2-y1,x2-x1,3):
                    raise ValueError(f"Raw ROI dimension mismatch: {raw_path}")
                image[y1:y2,x1:x2] = crop
                start = perf_counter()
                previous = old.detect_hook_bar(image, save_debug=False)
                previous_geometry = old_crossing.measure_hook_crossing_geometry(image)
                elapsed_old.append((perf_counter()-start)*1000)
                index, timestamp = sample["frame_index"], sample["monotonic_timestamp"]
                start = perf_counter()
                raw = adapter.observe(image, FrameContext(index,timestamp))
                elapsed_new.append((perf_counter()-start)*1000)
                result = controller.process(ObservationBundle(index,timestamp,None,raw), foreground=True, runtime_environment_supported=True, action_mode=ActionExecutionMode.RECORDED_OBSERVATION, preserve_proposal=True)
                rows.append({"frame":index,"timestamp":timestamp,
                             "old_raw":previous["raw_detected"], "old_reason":previous["structural_reason"],
                             "old_divider":previous_geometry.divider_line_x,"old_endpoint":previous_geometry.fill_endpoint_x,
                             "new_raw":raw.detected,"new_structural":raw.evidence["structural_candidate"],
                             "divider":raw.evidence.get("divider_line_x"),"endpoint":raw.evidence.get("fill_endpoint_x"),
                             "divider_roi":raw.evidence.get("divider_measurement_roi"),
                             "fill_roi":raw.evidence.get("fill_measurement_roi"),
                             "state":result.fsm.next_state.value,"transition":result.fsm.transition_reason,
                             "intent":result.fsm.action_request.intent.value,"safety":result.safety.reason})
            def timing(values):
                return {"mean_ms":round(float(np.mean(values)),3),"p95_ms":round(float(np.percentile(values,95)),3),"max_ms":round(max(values),3)}
            summary = {"episode":Path(manifest_path).parent.name,"samples":len(rows),
                       "old_candidates":sum(r["old_raw"] for r in rows),"new_candidates":sum(r["new_raw"] for r in rows),
                       "new_structural_candidates":sum(r["new_structural"] for r in rows),
                       "first_hook_frame":next((r["frame"] for r in rows if r["state"]=="HOOK"),None),
                       "hook_intent_frames":[r["frame"] for r in rows if r["intent"]==ActionIntent.HOOK_ACTION.value],
                       "old_detector_plus_crossing":timing(elapsed_old),"new_adapter":timing(elapsed_new),
                       "actions_applied":0,
                       "representative_frames":[r for r in rows if 6129<=r["frame"]<=6139] or [r for r in rows if r["new_structural"]][:2]}
            print(json.dumps(summary, ensure_ascii=False))
    finally:
        if archive:
            archive.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline", required=True, help="Unmodified Git revision to compare")
    args = parser.parse_args()
    if not args.session.exists():
        parser.error("Session evidence does not exist")
    run(args.session, args.baseline)
