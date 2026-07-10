def clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def is_strong(value: float, threshold: float) -> bool:
    return clamp_confidence(value) >= float(threshold)
