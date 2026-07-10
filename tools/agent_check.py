"""Run offline project health checks and write a concise Markdown report."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import (  # noqa: E402
    DEFAULT_CAPTURE_CONFIG_PATH,
    DEFAULT_ROI_CONFIG_PATH,
    DEFAULT_THRESHOLDS_CONFIG_PATH,
    load_capture_config,
    load_roi_config,
    load_thresholds_config,
)
from src.state_detector import STATE_BY_FILENAME  # noqa: E402


REPORT_PATH = PROJECT_ROOT / "reports" / "agent_check_latest.md"
REFERENCE_DIR = PROJECT_ROOT / "assets" / "reference"
PYTEST_TEMP_ROOT = PROJECT_ROOT / "tmp" / "agent_check_pytest"


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)


def parse_visual_report(output: str) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for line in output.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and "image" in item and "state" in item:
            results.append(item)
    return results


def build_report(
    *,
    references: list[tuple[str, str]],
    roi_status: str,
    thresholds_status: str,
    capture_status: str,
    visual: subprocess.CompletedProcess[str],
    detections: list[dict[str, object]],
    tests: subprocess.CompletedProcess[str],
    replay_tests: subprocess.CompletedProcess[str],
    failures: list[str],
) -> str:
    detected_by_name = {Path(str(item["image"])).name: item for item in detections}
    lines = [
        "# Fishing Assistant Agent Check",
        "",
        f"- Generated (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}",
        f"- Python: {sys.version.splitlines()[0]}",
        f"- OS: {platform.platform()}",
        f"- Reference image count: {len(references)}",
        f"- ROI config: {roi_status}",
        f"- Threshold config: {thresholds_status}",
        f"- Capture config: {capture_status}",
        "",
        "## Static state report",
        "",
        "| Image | Expected state | Detected state | Confidence |",
        "| --- | --- | --- | ---: |",
    ]
    debug_paths: list[str] = []
    for filename, expected in references:
        item = detected_by_name.get(filename)
        if item is None:
            lines.append(f"| {filename} | {expected} | _missing_ | — |")
            continue
        confidence = float(item.get("confidence", 0.0))
        detected = str(item.get("state", "_missing_"))
        lines.append(f"| {filename} | {expected} | {detected} | {confidence:.2%} |")
        debug_path = item.get("debug_image_path")
        if debug_path:
            debug_paths.append(str(debug_path))

    lines.extend(
        [
            "",
            "## Command results",
            "",
            f"- `visual_state_report.py --all --json`: exit code {visual.returncode}",
            f"- `pytest -q`: exit code {tests.returncode}",
            f"- replay session/detector tests: exit code {replay_tests.returncode}",
            "",
            "## Pytest output",
            "",
            "```text",
            (tests.stdout + tests.stderr).strip() or "(no output)",
            "```",
            "",
            "## Failure summary",
            "",
    ]
    )
    lines.extend([f"- {failure}" for failure in failures] or ["- None"])
    lines.extend(
        [
            "",
            "## Replay session/detector test output",
            "",
            "```text",
            (replay_tests.stdout + replay_tests.stderr).strip() or "(no output)",
            "```",
        ]
    )
    lines.extend(["", "## Debug output paths", ""])
    lines.extend([f"- {path}" for path in debug_paths] or ["- None"])
    return "\n".join(lines) + "\n"


def main() -> int:
    references = [(filename, state) for filename, state in STATE_BY_FILENAME.items()]
    failures: list[str] = []
    missing = [filename for filename, _ in references if not (REFERENCE_DIR / filename).is_file()]
    if missing:
        failures.append(f"Missing reference images: {', '.join(missing)}")

    try:
        load_roi_config(DEFAULT_ROI_CONFIG_PATH)
        roi_status = f"OK ({DEFAULT_ROI_CONFIG_PATH})"
    except Exception as exc:  # Report configuration faults instead of stopping diagnostics.
        roi_status = f"FAILED: {exc}"
        failures.append(f"ROI configuration: {exc}")
    try:
        load_thresholds_config(DEFAULT_THRESHOLDS_CONFIG_PATH)
        thresholds_status = f"OK ({DEFAULT_THRESHOLDS_CONFIG_PATH})"
    except Exception as exc:
        thresholds_status = f"FAILED: {exc}"
        failures.append(f"Threshold configuration: {exc}")
    try:
        load_capture_config(DEFAULT_CAPTURE_CONFIG_PATH)
        capture_status = f"OK ({DEFAULT_CAPTURE_CONFIG_PATH})"
    except Exception as exc:
        capture_status = f"FAILED: {exc}"
        failures.append(f"Capture configuration: {exc}")

    visual = run_command([sys.executable, "tools/visual_state_report.py", "--all", "--json"])
    detections = parse_visual_report(visual.stdout)
    if visual.returncode != 0:
        failures.append(f"visual_state_report failed (exit {visual.returncode}): {(visual.stderr or visual.stdout).strip()}")
    detected_by_name = {Path(str(item["image"])).name: item for item in detections}
    for filename, expected in references:
        result = detected_by_name.get(filename)
        if result is None:
            failures.append(f"No report result for {filename}")
        elif result.get("state") != expected:
            failures.append(f"{filename}: expected {expected}, detected {result.get('state')}")

    PYTEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    tests = run_command(
        [sys.executable, "-m", "pytest", "-q", "--basetemp", str(PYTEST_TEMP_ROOT / "all")]
    )
    if tests.returncode != 0:
        failures.append(f"pytest failed (exit {tests.returncode})")
    replay_tests = run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_replay_session.py",
            "tests/test_replay_detector.py",
            "-q",
            "--basetemp",
            str(PYTEST_TEMP_ROOT / "replay"),
        ]
    )
    if replay_tests.returncode != 0:
        failures.append(f"replay session/detector tests failed (exit {replay_tests.returncode})")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        build_report(
            references=references,
            roi_status=roi_status,
            thresholds_status=thresholds_status,
            capture_status=capture_status,
            visual=visual,
            detections=detections,
            tests=tests,
            replay_tests=replay_tests,
            failures=failures,
        ),
        encoding="utf-8",
    )
    print(REPORT_PATH)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
