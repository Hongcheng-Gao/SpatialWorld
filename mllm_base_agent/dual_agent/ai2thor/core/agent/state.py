"""Compatibility wrapper for the unified mllm_base_agent package."""

from mllm_base_agent.agent.state import *


def initialize_dual_agent_state(
    *,
    task_prompt,
    env,
    vlm,
    agent_vlms=None,
    config=None,
    max_global_steps=None,
    collaboration_mode="alternating",
    run_output_dir=None,
    skip_env_reset=False,
):
    """Build an initial state compatible with the shared runner."""
    max_steps = max_global_steps or (config or {}).get("max_steps", 50)
    return {
        "task_prompt": task_prompt,
        "observation": None,
        "step_count": 0,
        "max_steps": max_steps,
        "max_steps_override": max_steps,
        "success": None,
        "fail_reason": None,
        "failure_type": None,
        "short_term_history": [],
        "long_term_summary": "",
        "structured_trajectory": [],
        "conversation_history": [],
        "next_action": None,
        "should_continue": True,
        "task_done_by_model": False,
        "task_fail_by_model": False,
        "think_failed": False,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "vlm": vlm,
        "agent_vlms": agent_vlms or {"agent_1": vlm, "agent_2": vlm},
        "env": env,
        "config": config or {},
        "executor_type": (config or {}).get("executor", {}).get("type"),
        "goal_image_path": None,
        "run_output_dir": run_output_dir,
        "skip_env_reset": skip_env_reset,
        "collaboration_mode": collaboration_mode,
        "global_step_count": 0,
        "global_action_count": 0,
        "global_success": None,
        "communication_history": [],
        "agent_1": {"step_count": 0},
        "agent_2": {"step_count": 0},
    }
