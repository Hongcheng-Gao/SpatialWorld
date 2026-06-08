"""Evaluator registry and compatibility imports."""

from __future__ import annotations

from typing import Any, Dict


def create_evaluator_from_config(task_config: Dict[str, Any]):
    """Create an evaluator using the first available legacy implementation."""
    last_error = None
    for module_name in (
        'evaluation.virtualhome.base',
        'evaluation.ai2thor.base',
        'evaluation.carla.base',
        'evaluation.procthor.base',
    ):
        try:
            module = __import__(module_name, fromlist=['create_evaluator_from_config'])
            return module.create_evaluator_from_config(task_config)
        except Exception as exc:
            last_error = exc
    raise ImportError(f'No evaluator implementation available: {last_error}')
