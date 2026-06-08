"""Compatibility wrapper for the unified mllm_base_agent package."""

from mllm_base_agent.agent.graph import *
from mllm_base_agent.agent.graph import create_agent_graph


def _thor_agent_id_for_logical(agent_id):
    """Map logical dual-agent IDs to AI2-THOR agent indices."""
    if isinstance(agent_id, int):
        return agent_id
    text = str(agent_id).strip().lower()
    if text in {"agent_2", "agent2", "2", "second"}:
        return 1
    return 0


def create_dual_agent_graph():
    """Return the shared agent runner used by legacy dual-agent entry points."""
    return create_agent_graph()
