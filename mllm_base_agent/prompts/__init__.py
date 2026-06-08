"""System prompts for all environments."""

from mllm_base_agent.prompts.ai2thor import (
    AI2THOR_THINK_SYSTEM_PROMPT,
    AI2THOR_THINK_SYSTEM_PROMPT_NO_SUMMARY,
    get_ai2thor_prompt,
)

try:
    from mllm_base_agent.prompts.ai2thor_continuous import (
        AI2THOR_THINK_SYSTEM_PROMPT_CONTINUOUS,
        AI2THOR_THINK_SYSTEM_PROMPT_NO_SUMMARY_CONTINUOUS,
        get_ai2thor_continuous_prompt,
    )
except Exception:
    pass

from mllm_base_agent.prompts.carla import (
    CARLA_MAP_THINK_SYSTEM_PROMPT,
    CARLA_THINK_SYSTEM_PROMPT,
    CARLA_VEHICLE_THINK_SYSTEM_PROMPT,
    CARLA_WALKER_THINK_SYSTEM_PROMPT,
)
from mllm_base_agent.prompts.procthor import (
    PROCTHOR_THINK_SYSTEM_PROMPT,
    PROCTHOR_THINK_SYSTEM_PROMPT_NO_SUMMARY,
    get_procthor_prompt,
)
from mllm_base_agent.prompts.procthor_continuous import (
    PROCTHOR_THINK_SYSTEM_PROMPT_CONTINUOUS,
    PROCTHOR_THINK_SYSTEM_PROMPT_NO_SUMMARY_CONTINUOUS,
    get_procthor_continuous_prompt,
)
from mllm_base_agent.prompts.virtualhome import (
    VIRTUALHOME_THINK_SYSTEM_PROMPT,
    VIRTUALHOME_THINK_SYSTEM_PROMPT_NO_SUMMARY,
    get_virtualhome_prompt,
)
from mllm_base_agent.prompts.registry import get_system_prompt
