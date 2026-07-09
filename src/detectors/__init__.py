"""Offline component detectors for fishing mini-game UI elements."""

from src.detectors.hook_detector import detect_hook_bar
from src.detectors.press_detector import detect_press_sequence

__all__ = ["detect_hook_bar", "detect_press_sequence"]
