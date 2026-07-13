from pathlib import Path

from src.fishing_v2.runtime.hook_crossing_geometry import measure_hook_crossing_geometry


ROOT = Path(__file__).resolve().parents[2]
SESSIONS = ROOT / "assets" / "replay" / "sessions"


def _frame(session: str, frame: int) -> Path:
    return SESSIONS / session / "frames" / f"{frame:06d}.jpg"


def test_crossing_geometry_uses_white_divider_and_cyan_endpoint() -> None:
    before = measure_hook_crossing_geometry(_frame("session_20260710_124419", 314))
    after = measure_hook_crossing_geometry(_frame("session_20260710_124419", 316))

    assert before.divider_line_detected is True
    assert before.fill_endpoint_x is None
    assert after.divider_line_x == 1330.0
    assert after.fill_endpoint_x is not None
    assert after.fill_endpoint_x >= after.divider_line_x + 10


def test_margin_distinguishes_first_cyan_pixel_from_safe_crossing() -> None:
    shallow = measure_hook_crossing_geometry(_frame("session_20260710_125441", 368))
    safe = measure_hook_crossing_geometry(_frame("session_20260710_125441", 369))

    assert shallow.fill_endpoint_x is not None
    assert shallow.fill_endpoint_x < shallow.divider_line_x + 10
    assert safe.fill_endpoint_x is not None
    assert safe.fill_endpoint_x >= safe.divider_line_x + 10
