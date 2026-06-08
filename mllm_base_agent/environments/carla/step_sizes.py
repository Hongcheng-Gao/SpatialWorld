"""Discrete CARLA step-size and turn labels shared by vehicle and walker executors."""

from __future__ import annotations

from typing import Any

# Only these three distances are accepted (meters).
STEP_DISTANCES: tuple[float, float, float] = (2.5, 5.0, 10.0)

STEP_SIZE_LABEL_TO_METERS: dict[str, float] = {
    "small": 2.5,
    "medium": 5.0,
    "large": 10.0,
}

# Walker turns: medium / large only (no small tier for turns).
TURN_DEGREES: tuple[float, float] = (30.0, 90.0)

TURN_SIZE_LABEL_TO_DEGREES: dict[str, float] = {
    "medium": 30.0,
    "large": 90.0,
    "small": 30.0,  # legacy alias → medium
}


def resolve_step_distance(value: Any, *, default: float = 10.0) -> float:
    """Map model output to one of {2.5, 5, 10}.

    Accepts:
      - labels: small / medium / large (case-insensitive)
      - numeric literals: 2.5, 5, 10 (snapped to nearest allowed)
    """
    if isinstance(value, str):
        label = value.strip().lower()
        if label in STEP_SIZE_LABEL_TO_METERS:
            return STEP_SIZE_LABEL_TO_METERS[label]

    try:
        v = float(value)
    except (TypeError, ValueError):
        return default

    return min(STEP_DISTANCES, key=lambda a: abs(a - v))


def turn_label_for_degrees(degrees: float) -> str:
    """Map snapped turn angle to prompt label (medium or large)."""
    d = resolve_turn_degrees(degrees)
    return "large" if d >= 60.0 else "medium"


def resolve_turn_degrees(value: Any, *, default: float = 30.0) -> float:
    """Map model output to one of {30, 90}.

    Accepts:
      - labels: medium / large (case-insensitive; small is a legacy alias for medium)
      - numeric literals: 30, 90 (snapped to nearest allowed)
    """
    if isinstance(value, str):
        label = value.strip().lower()
        if label in TURN_SIZE_LABEL_TO_DEGREES:
            return TURN_SIZE_LABEL_TO_DEGREES[label]

    try:
        v = float(value)
    except (TypeError, ValueError):
        return default

    return min(TURN_DEGREES, key=lambda a: abs(a - v))
