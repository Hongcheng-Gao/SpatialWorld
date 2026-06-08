"""Prompt registry for all SpatialWorld environments."""

from __future__ import annotations

from typing import Optional

from mllm_base_agent.prompts.ai2thor import get_ai2thor_prompt
try:
    from mllm_base_agent.prompts.ai2thor_continuous import get_ai2thor_continuous_prompt
except Exception:  # pragma: no cover
    get_ai2thor_continuous_prompt = None
from mllm_base_agent.prompts.carla import (
    CARLA_MAP_THINK_SYSTEM_PROMPT,
    CARLA_VEHICLE_THINK_SYSTEM_PROMPT,
    CARLA_WALKER_THINK_SYSTEM_PROMPT,
)
from mllm_base_agent.prompts.procthor import get_procthor_prompt
from mllm_base_agent.prompts.procthor_continuous import get_procthor_continuous_prompt
from mllm_base_agent.prompts.virtualhome import get_virtualhome_prompt


def get_system_prompt(
    env_type: str,
    enable_summary: bool = False,
    executor_type: Optional[str] = None,
    input_modality: Optional[str] = None,
    navigation_mode: str = 'discrete',
    virtualhome_interactable_object_types: Optional[list[str]] = None,
) -> str:
    env_type = (env_type or 'ai2thor').lower()
    if env_type == 'carla':
        if input_modality == 'map_goal':
            return CARLA_MAP_THINK_SYSTEM_PROMPT
        if executor_type == 'walker':
            return CARLA_WALKER_THINK_SYSTEM_PROMPT
        return CARLA_VEHICLE_THINK_SYSTEM_PROMPT
    if env_type == 'virtualhome':
        return get_virtualhome_prompt(interactable_object_types=virtualhome_interactable_object_types)
    if env_type == 'procthor':
        if navigation_mode == 'continuous':
            return get_procthor_continuous_prompt()
        return get_procthor_prompt()
    if env_type == 'ai2thor' and navigation_mode == 'continuous' and get_ai2thor_continuous_prompt is not None:
        return get_ai2thor_continuous_prompt()
    return get_ai2thor_prompt()
