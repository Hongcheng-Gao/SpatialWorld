"""Utilities for deriving the episode step budget.

SpatialWorld uses the paper-style rule: max_steps = 10 + 2n, where n is
normally the number of executable golden actions, excluding terminal DONE/FAIL.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

_TERMINAL_ACTIONS = {"DONE", "FAIL"}


def _safe_len(value: Any) -> Optional[int]:
    if isinstance(value, (str, bytes)) or value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return None


def _non_terminal_action_count(actions: Any) -> Optional[int]:
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
        return None
    count = 0
    for action in actions:
        if str(action).strip().upper() not in _TERMINAL_ACTIONS:
            count += 1
    return count if count > 0 else None


def derive_task_n(task_config: Mapping[str, Any] | None) -> Optional[int]:
    """Infer n for max_steps = 10 + 2n from common task fields."""
    if not isinstance(task_config, Mapping):
        return None

    golden = task_config.get("golden_actions") or {}
    if isinstance(golden, Mapping):
        count = _non_terminal_action_count(golden.get("actions"))
        if count is not None:
            return count
        steps = golden.get("steps")
        try:
            if steps is not None:
                return max(0, int(steps))
        except (TypeError, ValueError):
            pass

    for key in ("n", "num_steps", "golden_steps", "expert_steps"):
        try:
            if task_config.get(key) is not None:
                return max(0, int(task_config[key]))
        except (TypeError, ValueError):
            pass

    for key in ("init_actions", "actions"):
        count = _non_terminal_action_count(task_config.get(key))
        if count is not None:
            return count

    for key in ("success_conditions", "target_object_types", "target_objects"):
        count = _safe_len(task_config.get(key))
        if count is not None and count > 0:
            return count

    return None


def compute_max_steps_from_n(n: int) -> int:
    return 10 + 2 * max(0, int(n))


def derive_dual_golden_steps(task_config: Mapping[str, Any] | None) -> Optional[int]:
    """Read n from ``golden_actions.steps`` for multi-agent tasks."""
    if not isinstance(task_config, Mapping):
        return None
    golden = task_config.get("golden_actions") or {}
    if not isinstance(golden, Mapping):
        return None
    steps = golden.get("steps")
    try:
        if steps is not None:
            return max(0, int(steps))
    except (TypeError, ValueError):
        return None
    return None


def compute_dual_agent_max_steps_from_steps(steps: int) -> int:
    """Per-agent step budget for multi-agent tasks: max_steps = 10 + n."""
    return 10 + max(0, int(steps))


def resolve_dual_agent_max_steps_from_task(
    task_config: Mapping[str, Any] | None, default: int = 30
) -> int:
    """Resolve per-agent max_steps for multi-agent tasks using golden_actions.steps."""
    steps = derive_dual_golden_steps(task_config)
    if steps is not None:
        return compute_dual_agent_max_steps_from_steps(steps)
    if isinstance(task_config, Mapping) and task_config.get("max_steps") is not None:
        try:
            return int(task_config["max_steps"])
        except (TypeError, ValueError):
            pass
    return int(default)


def resolve_max_steps_from_task(task_config: Mapping[str, Any] | None, default: int = 30) -> int:
    """Resolve max_steps using 10 + 2n, falling back only when n is unknown."""
    n = derive_task_n(task_config)
    if n is not None:
        return compute_max_steps_from_n(n)
    if isinstance(task_config, Mapping) and task_config.get("max_steps") is not None:
        try:
            return int(task_config["max_steps"])
        except (TypeError, ValueError):
            pass
    return int(default)


__all__ = [
    "compute_dual_agent_max_steps_from_steps",
    "compute_max_steps_from_n",
    "derive_dual_golden_steps",
    "derive_task_n",
    "resolve_dual_agent_max_steps_from_task",
    "resolve_max_steps_from_task",
]
