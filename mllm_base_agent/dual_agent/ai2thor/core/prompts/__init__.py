"""
Dual Agent Prompts Module
Contains system prompts for the dual-agent collaboration system
Both agents have equal status and collaborate as partners.

Note: Legacy role-based exports (EXPLORER_*, EXECUTOR_*) are kept for backward compatibility
but are deprecated. Use get_collaborative_system_prompt() for new code.
"""
from .dual_agent import (
    #       （  ）
    get_collaborative_system_prompt,
    get_communication_prompt,
    #       （      ）
    EXPLORER_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
    get_explorer_system_prompt,
    get_executor_system_prompt,
)

__all__ = [
    #       
    "get_collaborative_system_prompt",
    "get_communication_prompt",
    #       （      ）
    "EXPLORER_SYSTEM_PROMPT",
    "EXECUTOR_SYSTEM_PROMPT",
    "get_explorer_system_prompt",
    "get_executor_system_prompt",
]
