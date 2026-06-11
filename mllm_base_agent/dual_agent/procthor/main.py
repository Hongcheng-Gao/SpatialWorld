#!/usr/bin/env python3
"""
ProcTHOR dual-agent main entry.

Design goals:
- Reuse SpatialWorld single-agent provider / parser / environment stack.
- Support one shared model or two per-agent models via --agent1 / --agent2.
- Keep output artifacts compatible with benchmark usage (log.json + dual_episode_*.json).
"""


"""
python -m mllm_base_agent.dual_agent.procthor.run_csv_benchmark \
  --csv "./experiments/csv/procthor/dual/Spatial-Annotation-procthor.csv" \
  --config "experiments/configs/procthor/dual/config_close_gpt-5.yaml"
"""

import argparse
import base64
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from actions.max_steps import derive_dual_golden_steps
from configs.procthor.load_config import load_config
from actions.parser import parse_action_string
from mllm_base_agent.llm.provider import get_vlm
from mllm_base_agent.environments.procthor.wrapper import ProcTHOREnvWrapper
from evaluation.procthor.base import create_evaluator_from_config
from scripts.evaluate_actions_procthor import load_init_actions_for_task

from .prompts import get_dual_procthor_prompt


LOCAL_RETRY_CONFIG = {
    "max_retries": 3,
    "api_max_retries": 5,
    "retry_delay": 2,
    "api_retry_delay": 5,
}
MODEL_HISTORY_TURNS = 29


# Map logical agent id ("agent_1" / "agent_2") to AI2-THOR embodied agentId (0 / 1).
AGENT_TO_THOR_ID = {"agent_1": 0, "agent_2": 1}


class APIRetryError(Exception):
    """API retries exhausted."""


def load_config_dict(config_path: str) -> dict:
    """Load config file into a plain dict."""
    config = load_config(config_path)
    return config.get_all() if hasattr(config, "get_all") else config.config


def extract_model_config_from_file(config_path: str) -> dict:
    """Extract model.vlm from a single-agent config file."""
    config_dict = load_config_dict(config_path)
    model_config = config_dict.get("model", {}).get("vlm", {})
    if not model_config:
        raise ValueError(f"        model.vlm: {config_path}")
    return dict(model_config)


def apply_agent_model_overrides(
    config_dict: dict,
    agent1_config_path: Optional[str] = None,
    agent2_config_path: Optional[str] = None,
) -> dict:
    """Apply agent-specific model config overrides from single-agent config files."""
    if not agent1_config_path and not agent2_config_path:
        config_dict.setdefault("dual_agent", {})
        config_dict["dual_agent"].setdefault("use_separate_models", False)
        return config_dict

    if not agent1_config_path and agent2_config_path:
        agent1_config_path = agent2_config_path
    if not agent2_config_path and agent1_config_path:
        agent2_config_path = agent1_config_path

    agent1_path = str(Path(agent1_config_path).resolve())
    agent2_path = str(Path(agent2_config_path).resolve())

    for path in [agent1_path, agent2_path]:
        if not Path(path).exists():
            raise FileNotFoundError(f"         : {path}")

    agent1_model_config = extract_model_config_from_file(agent1_path)
    agent2_model_config = extract_model_config_from_file(agent2_path)

    config_dict.setdefault("model", {})
    config_dict.setdefault("dual_agent", {})

    if agent1_path == agent2_path:
        config_dict["model"]["vlm"] = agent1_model_config
        config_dict["dual_agent"]["use_separate_models"] = False
        print("✓         ")
        print(f"  Shared model config: {agent1_path}")
        print(f"     : {agent1_model_config.get('model_name')}")
    else:
        config_dict["model"]["vlm"] = agent1_model_config
        config_dict["dual_agent"]["use_separate_models"] = True
        config_dict["dual_agent"]["agent_1"] = agent1_model_config
        config_dict["dual_agent"]["agent_2"] = agent2_model_config
        print("✓        ")
        print(f"  Agent 1 config: {agent1_path}")
        print(f"  Agent 1    : {agent1_model_config.get('model_name')}")
        print(f"  Agent 2 config: {agent2_path}")
        print(f"  Agent 2    : {agent2_model_config.get('model_name')}")

    return config_dict


def create_vlm_from_config(vlm_config: dict):
    """Create a VLM instance from a config dict."""
    return get_vlm(
        provider=vlm_config.get("provider", "openai"),
        model_name=vlm_config.get("model_name", "gpt-4o"),
        temperature=vlm_config.get("temperature", 0.2),
        top_p=vlm_config.get("top_p"),
        max_tokens=vlm_config.get("max_tokens", 2000),
        base_url=vlm_config.get("base_url"),
        api_key=vlm_config.get("api_key"),
        proxy_url=vlm_config.get("proxy_url"),
    )


def get_agent_vlms(config_dict: dict) -> Tuple[Dict[str, Any], bool]:
    """Create either one shared model or two separate models."""
    model_config = dict(config_dict.get("model", {}).get("vlm", {}))
    dual_config = config_dict.get("dual_agent", {})
    use_separate_models = bool(dual_config.get("use_separate_models", False))

    def _vlm_display_name(vlm, cfg: dict) -> str:
        for attr in ("model_name", "_model", "model"):
            val = getattr(vlm, attr, None)
            if val:
                return str(val)
        return str(cfg.get("model_name") or cfg.get("model") or type(vlm).__name__)

    if not use_separate_models:
        shared = create_vlm_from_config(model_config)
        print(f"✓     : {_vlm_display_name(shared, model_config)}")
        return {"agent_1": shared, "agent_2": shared}, False

    agent_1_config = {**model_config, **(dual_config.get("agent_1", {}) or {})}
    agent_2_config = {**model_config, **(dual_config.get("agent_2", {}) or {})}
    agent_1_vlm = create_vlm_from_config(agent_1_config)
    agent_2_vlm = create_vlm_from_config(agent_2_config)
    print("✓         ")
    print(f"  Agent 1   : {_vlm_display_name(agent_1_vlm, agent_1_config)}")
    print(f"  Agent 2   : {_vlm_display_name(agent_2_vlm, agent_2_config)}")
    return {"agent_1": agent_1_vlm, "agent_2": agent_2_vlm}, True


def load_init_actions_from_task_folder(task_folder_path: str):
    """Load init actions from tasks/<task_id>/task.json."""
    task_file = os.path.join(task_folder_path, "task.json")
    if not os.path.isfile(task_file):
        return None
    return load_init_actions_for_task(task_file)


def perform_final_evaluation(env, task_config: dict, observation=None) -> tuple:
    """Run evaluator on the final environment state."""
    if not task_config:
        return False, 0.0

    if (not observation or not getattr(observation, "metadata", None)) and hasattr(env, "_get_current_observation"):
        try:
            observation = env._get_current_observation()
        except Exception:
            observation = observation

    if not observation or not getattr(observation, "metadata", None):
        return False, 0.0

    try:
        evaluator = create_evaluator_from_config(task_config)
        score = evaluator.evaluate(env, observation.metadata)
        return score >= 1.0, score
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return False, 0.0


def _extract_tag_block(text: str, tag: str) -> Optional[str]:
    start_token = f"<{tag}>"
    end_token = f"</{tag}>"
    start = text.find(start_token)
    end = text.find(end_token)
    if start == -1 or end == -1 or end <= start:
        return None
    start += len(start_token)
    return text[start:end]


def parse_dual_agent_response(response_text: str, enable_summary: bool = False) -> Dict[str, Any]:
    """Parse THINK/ACTION/(COMMUNICATE)/(SUMMARY) from a model response."""
    think_block = _extract_tag_block(response_text, "THINK")
    action_block = _extract_tag_block(response_text, "ACTION")
    communicate_block = _extract_tag_block(response_text, "COMMUNICATE")
    summary_block = _extract_tag_block(response_text, "SUMMARY") if enable_summary else None

    if not action_block:
        raise ValueError("Missing <ACTION> tag")
    action_string = action_block.strip()
    if not action_string:
        raise ValueError("ACTION tag content is empty")

    thinking_text = (
        think_block.strip()
        if think_block and think_block.strip()
        else "(No <THINK> block; action will still be executed)"
    )
    communication_text = communicate_block.strip() if communicate_block and communicate_block.strip() else ""
    updated_summary = summary_block.strip() if summary_block and summary_block.strip() else ""
    parsed_action = parse_action_string(action_string)
    if parsed_action.get("action_type") == "communication":
        communication_text = parsed_action.get("message", communication_text)
    return {
        "thinking_text": thinking_text,
        "action_string": action_string,
        "parsed_action": parsed_action,
        "communication_text": communication_text,
        "updated_summary": updated_summary,
    }


def initialize_agent_state(
    agent_id: str,
    vlm: Any,
    observation: Any,
    max_steps: int,
) -> dict:
    return {
        "agent_id": agent_id,
        "vlm": vlm,
        "observation": observation,
        "step_count": 0,
        "max_steps": max_steps,
        "short_term_history": [],
        "long_term_summary": "",
        "structured_trajectory": [],
        "should_continue": True,
        "failure_type": None,
        "fail_reason": None,
        "consecutive_failures": 0,
        "last_error_message": None,
    }


def build_shared_context(state: dict, current_agent_id: str) -> str:
    """Build shared context for the current agent from recent communications."""
    communications = state.get("communication_history", [])
    relevant = []
    for msg in communications[-8:]:
        sender = msg.get("sender", "unknown")
        receiver = msg.get("receiver", "unknown")
        message = msg.get("message", "")
        step = msg.get("global_step", "?")
        if receiver == current_agent_id or sender != current_agent_id:
            relevant.append(f"[Step {step}] {sender} -> {receiver}: {message}")
    return "\n".join(relevant) if relevant else "No messages from partner yet."


def consume_pending_messages(state: dict, current_agent_id: str) -> List[dict]:
    """Pop pending messages for the current agent."""
    pending = []
    remaining = []
    for msg in state.get("message_queue", []):
        if msg.get("receiver") == current_agent_id:
            pending.append(msg)
        else:
            remaining.append(msg)
    state["message_queue"] = remaining
    return pending


def agent_can_continue(state: dict, agent_id: str) -> bool:
    agent_state = state[agent_id]
    return agent_state.get("should_continue", True) and agent_state.get("step_count", 0) < agent_state.get("max_steps", 0)


def ensure_system_step_started(state: dict, current_agent_id: str) -> None:
    """Initialize expected agents for the current system step."""
    if state.get("system_step_expected_agents"):
        return

    other_agent_id = "agent_2" if current_agent_id == "agent_1" else "agent_1"
    expected = [current_agent_id]
    if agent_can_continue(state, other_agent_id):
        expected.append(other_agent_id)

    state["system_step_expected_agents"] = expected
    state["system_step_completed_agents"] = []


def finalize_system_step_if_complete(state: dict) -> None:
    """Increase system step count when all expected active agents have acted."""
    expected = state.get("system_step_expected_agents", [])
    completed = state.get("system_step_completed_agents", [])

    if expected and all(agent_id in completed for agent_id in expected):
        state["global_step_count"] = state.get("global_step_count", 0) + 1
        state["system_step_expected_agents"] = []
        state["system_step_completed_agents"] = []


def refresh_system_step_expected_agents(state: dict) -> None:
    """Drop agents that died mid-step and close the step if all remaining agents have acted."""
    expected = state.get("system_step_expected_agents", [])
    if not expected:
        return

    completed = state.get("system_step_completed_agents", [])
    refreshed = []
    for agent_id in expected:
        if agent_id in completed or agent_can_continue(state, agent_id):
            refreshed.append(agent_id)

    state["system_step_expected_agents"] = refreshed
    finalize_system_step_if_complete(state)


def record_system_step_progress(state: dict, current_agent_id: str) -> None:
    """Record one agent action toward the current system step."""
    ensure_system_step_started(state, current_agent_id)

    completed = state.get("system_step_completed_agents", [])
    if current_agent_id not in completed:
        completed.append(current_agent_id)
    state["system_step_completed_agents"] = completed

    finalize_system_step_if_complete(state)


def finalize_partial_system_step(state: dict) -> None:
    """Count the last partial system step when execution terminates mid-round."""
    expected = state.get("system_step_expected_agents", [])
    completed = state.get("system_step_completed_agents", [])

    if expected and completed:
        state["global_step_count"] = state.get("global_step_count", 0) + 1

    state["system_step_expected_agents"] = []
    state["system_step_completed_agents"] = []


def maybe_switch_agent(state: dict, switch_interval: int) -> Optional[str]:
    """Choose current agent according to collaboration mode and availability."""
    current_agent = state["current_agent"]
    other_agent = "agent_2" if current_agent == "agent_1" else "agent_1"
    mode = state.get("collaboration_mode", "alternating")

    if mode == "alternating" and state.get("current_turn_steps", 0) >= switch_interval and agent_can_continue(state, other_agent):
        state["current_agent"] = other_agent
        state["current_turn_steps"] = 0
        state["turn_count"] += 1
        current_agent = other_agent

    if not agent_can_continue(state, current_agent):
        if agent_can_continue(state, other_agent):
            state["current_agent"] = other_agent
            state["current_turn_steps"] = 0
            state["turn_count"] += 1
            return other_agent
        return None

    if mode == "sequential" and current_agent == "agent_1" and not agent_can_continue(state, "agent_1"):
        if agent_can_continue(state, "agent_2"):
            state["current_agent"] = "agent_2"
            state["current_turn_steps"] = 0
            state["turn_count"] += 1
            return "agent_2"
        return None

    return current_agent


def handoff_agent_or_finish(
    state: dict,
    current_agent_id: str,
    reason: str,
    failure_type: Optional[str] = None,
) -> bool:
    """Deactivate current agent and hand off to the partner if possible."""
    other_agent_id = "agent_2" if current_agent_id == "agent_1" else "agent_1"
    current_agent = state[current_agent_id]

    current_agent["should_continue"] = False
    current_agent["fail_reason"] = reason
    current_agent["last_error_message"] = reason
    if failure_type is not None:
        current_agent["failure_type"] = failure_type

    if agent_can_continue(state, other_agent_id):
        state["current_agent"] = other_agent_id
        state["current_turn_steps"] = 0
        state["turn_count"] += 1
        print(f"🔄     {other_agent_id}: {reason}")
        return True

    if not state.get("fail_reason"):
        state["fail_reason"] = reason
    if failure_type is not None and not state.get("failure_type"):
        state["failure_type"] = failure_type
    print(f"⚠️      ：{other_agent_id}         : {reason}")
    return False


def mark_agent_failure(state: dict, agent_id: str, failure_type: str, fail_reason: str):
    """Mark current agent as failed and maybe update global failure hints."""
    agent_state = state[agent_id]
    agent_state["should_continue"] = False
    agent_state["failure_type"] = failure_type
    agent_state["fail_reason"] = fail_reason
    if not state.get("failure_type"):
        state["failure_type"] = failure_type
    if not state.get("fail_reason"):
        state["fail_reason"] = fail_reason


def save_dual_conversation_log(state: dict, output_dir: str):
    """Save log.json for dual-agent runs."""
    log_file = os.path.join(output_dir, "log.json")
    conversation_json = {
        "metadata": {
            "task_description": state.get("task_prompt", ""),
            "task_result": "success" if state.get("success") else "failure",
            "fail_reason": state.get("fail_reason"),
            "failure_type": state.get("failure_type"),
            "total_steps": state.get("global_step_count", 0),
            "max_steps": state.get("max_global_steps", 0),
            "agent_1_steps": state.get("agent_1", {}).get("step_count", 0),
            "agent_2_steps": state.get("agent_2", {}).get("step_count", 0),
            "communication_events": len(state.get("communication_history", [])),
            "mode": "dual_agent",
        },
        "messages": [],
        "images": [],
    }

    trajectory = []
    for agent_id in ["agent_1", "agent_2"]:
        for entry in state.get(agent_id, {}).get("structured_trajectory", []):
            copied = dict(entry)
            copied["agent_id"] = agent_id
            trajectory.append(copied)
    trajectory.sort(key=lambda x: (x.get("global_step", 0), x.get("agent_id", "")))

    for entry in trajectory:
        step_id = entry.get("global_step", entry.get("step", 0))
        image_path = entry.get("image_path", "")
        raw_response = entry.get("raw_response", "")
        action_string = entry.get("action_string", "")
        reward = entry.get("reward", 0)
        error_message = entry.get("error_message")
        agent_id = entry.get("agent_id", "agent")
        conversation_json["messages"].append(
            {
                "role": "user",
                "content": f"[{agent_id}] Step {step_id}" + ("\n<image>" if image_path else ""),
                "step": step_id,
                "image_path": image_path,
            }
        )
        conversation_json["messages"].append(
            {
                "role": "assistant",
                "content": raw_response,
                "step": step_id,
                "agent_id": agent_id,
                "action_executed": action_string,
                "reward": reward,
                "error_message": error_message,
                "communication": entry.get("communication", ""),
            }
        )
        if image_path:
            conversation_json["images"].append(image_path)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(conversation_json, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Conversation log saved: {log_file}")


def save_dual_episode_log(state: dict, output_dir: str, env) -> None:
    """Save a detailed dual-agent episode json."""
    scene_name = "ProcTHOR"
    if hasattr(env, "scene") and env.scene is not None:
        scene_name = str(env.scene) if not isinstance(env.scene, dict) else env.scene.get("sceneName", "ProcTHOR")
    scene_short = scene_name.replace(" ", "_").replace("/", "_")[:50]
    task_name = (state.get("config") or {}).get("task", {}).get("name", "task") or "task"
    task_short = task_name.replace(" ", "_").replace("/", "_")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dual_episode_{scene_short}_{task_short}_{timestamp_str}.json"
    filepath = os.path.join(output_dir, filename)

    trajectory = []
    for agent_id in ["agent_1", "agent_2"]:
        for entry in state.get(agent_id, {}).get("structured_trajectory", []):
            copied = dict(entry)
            copied["agent_id"] = agent_id
            trajectory.append(copied)
    trajectory.sort(key=lambda x: (x.get("global_step", 0), x.get("agent_id", "")))

    episode_log = {
        "task": state.get("task_prompt", ""),
        "scene": scene_name,
        "success": state.get("success", False),
        "fail_reason": state.get("fail_reason"),
        "failure_type": state.get("failure_type"),
        "global_step_count": state.get("global_step_count", 0),
        "max_global_steps": state.get("max_global_steps", 0),
        "agent_1_steps": state.get("agent_1", {}).get("step_count", 0),
        "agent_2_steps": state.get("agent_2", {}).get("step_count", 0),
        "turn_count": state.get("turn_count", 0),
        "communication_history": state.get("communication_history", []),
        "action_sequence": env.get_action_sequence() if hasattr(env, "get_action_sequence") else "(no action records)",
        "trajectory": trajectory,
        "timestamp": datetime.now().isoformat(),
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(episode_log, f, ensure_ascii=False, indent=2)
    print(f"✓ Dual episode log saved: {filepath}")


def run_dual_agent_loop(
    env,
    agent_vlms: Dict[str, Any],
    task_config: dict,
    task_output_dir: str,
    config: dict,
    collaboration_mode: str = "alternating",
    switch_interval: int = 1,
):
    """Run a lightweight dual-agent collaboration loop on one shared ProcTHOR env."""
    task_prompt = task_config.get("instruction") or task_config.get("description") or "Complete the task."
    per_agent_steps = int(task_config.get("max_steps", config.get("max_steps", 30)))
    max_global_steps = 2 * per_agent_steps
    golden_steps = derive_dual_golden_steps(task_config)
    if golden_steps is not None:
        print(
            f"✓ Per-agent max steps (10+n, n=golden_actions.steps={int(golden_steps)}): "
            f"{per_agent_steps} | Global cap: {max_global_steps}"
        )
    else:
        print(f"✓ Per-agent max steps: {per_agent_steps} | Global cap: {max_global_steps}")
    enable_summary = config.get("context_management", {}).get("enable_long_term_summary", False)
    configured_history = int(config.get("context_management", {}).get("short_term_history_window_size", MODEL_HISTORY_TURNS))
    max_history = min(MODEL_HISTORY_TURNS, max(0, configured_history))

    initial_observation = env.reset(task_prompt)

    #   ：  agent2     agent1   ；                （    agent1/   agent2/）
    agent_1_observation = initial_observation
    agent_2_observation = initial_observation
    if getattr(env, "agent_count", 1) > 1:
        if hasattr(env, "relocate_second_agent_near_agent1"):
            env.relocate_second_agent_near_agent1()
        if hasattr(env, "get_observation_for_agent"):
            try:
                agent_1_observation = env.get_observation_for_agent(0)
                agent_2_observation = env.get_observation_for_agent(1)
                print(
                    f"✓     : agent1={agent_1_observation.image_path} | "
                    f"agent2={agent_2_observation.image_path}"
                )
            except Exception as e:
                print(f"⚠️            ，       : {e}")

    state = {
        "task_prompt": task_prompt,
        "config": config,
        "collaboration_mode": collaboration_mode,
        "switch_interval": switch_interval,
        "global_step_count": 0,
        "max_global_steps": max_global_steps,
        "system_step_expected_agents": [],
        "system_step_completed_agents": [],
        "current_agent": "agent_1",
        "current_turn_steps": 0,
        "turn_count": 0,
        "communication_history": [],
        "message_queue": [],
        "success": False,
        "fail_reason": None,
        "failure_type": None,
        "agent_1": initialize_agent_state("agent_1", agent_vlms["agent_1"], agent_1_observation, per_agent_steps),
        "agent_2": initialize_agent_state("agent_2", agent_vlms["agent_2"], agent_2_observation, per_agent_steps),
    }

    while state["global_step_count"] < state["max_global_steps"]:
        refresh_system_step_expected_agents(state)
        if state["global_step_count"] >= state["max_global_steps"]:
            break

        current_agent_id = maybe_switch_agent(state, switch_interval)
        if current_agent_id is None:
            if not state.get("fail_reason"):
                state["fail_reason"] = "Both agents stopped before success"
            break

        current_agent = state[current_agent_id]
        other_agent_id = "agent_2" if current_agent_id == "agent_1" else "agent_1"

        #   ：    "   agent"           （  agent      ，
        #     step_with_action_dict            frame，      Pass） 
        if getattr(env, "agent_count", 1) > 1 and hasattr(env, "get_observation_for_agent"):
            last_acting = state.get("last_acting_agent")
            needs_refresh = (
                current_agent.get("observation") is None
                or (last_acting is not None and last_acting != current_agent_id)
            )
            if needs_refresh:
                try:
                    thor_aid = AGENT_TO_THOR_ID.get(current_agent_id, 0)
                    current_agent["observation"] = env.get_observation_for_agent(thor_aid)
                except Exception as e:
                    print(f"⚠️  Refresh obs for {current_agent_id} failed: {e}")

        observation = current_agent.get("observation") or state["agent_1"].get("observation")

        print(f"\n{'=' * 60}\n🧠 {current_agent_id.upper()} Step {current_agent['step_count'] + 1}\n{'=' * 60}")

        from mllm_base_agent.llm.messages import AIMessage, HumanMessage, SystemMessage

        system_prompt = get_dual_procthor_prompt(enable_summary=enable_summary).format(
            task_prompt=task_prompt,
            shared_context=build_shared_context(state, current_agent_id),
        )
        messages = [SystemMessage(content=system_prompt)]

        for entry in current_agent.get("short_term_history", [])[-max_history:]:
            content = []
            img_path = entry.get("image_path")
            if img_path and os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
            content.append({"type": "text", "text": f"Step {entry.get('step', 0)}"})
            messages.append(HumanMessage(content=content))
            messages.append(AIMessage(content=entry.get("raw_response", "")))

        pending_messages = consume_pending_messages(state, current_agent_id)
        current_content = []
        if pending_messages:
            current_content.append(
                {
                    "type": "text",
                    "text": "**Messages from Partner:**\n" + "\n".join(f"- {msg.get('message', '')}" for msg in pending_messages),
                }
            )

        last_error = current_agent.get("last_error_message")
        if last_error:
            current_content.append(
                {
                    "type": "text",
                    "text": f"**Last action error:** {last_error}\nAdjust your plan before repeating the same action.",
                }
            )

        image_path = observation.image_path if observation else None
        if not image_path or not os.path.exists(image_path):
            mark_agent_failure(state, current_agent_id, "env_error", "Missing observation image")
            continue

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        current_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
        current_content.append(
            {
                "type": "text",
                "text": f"Current step is Step {current_agent['step_count'] + 1}. "
                "Please output <THINK> and <ACTION> using the unified action space.",
            }
        )
        messages.append(HumanMessage(content=current_content))

        response_text = None
        last_api_error = None
        for api_attempt in range(LOCAL_RETRY_CONFIG["api_max_retries"]):
            try:
                print(f"📡 Calling VLM... (API attempt {api_attempt + 1}/{LOCAL_RETRY_CONFIG['api_max_retries']})")
                response = current_agent["vlm"].invoke(messages)
                response_text = response.content if hasattr(response, "content") else str(response)
                break
            except Exception as api_error:
                err_str = str(api_error)
                is_api = any(k in err_str.lower() for k in ["api", "request failed", "connection", "timeout", "timed out", "http", "429", "500", "400"])
                if is_api and api_attempt < LOCAL_RETRY_CONFIG["api_max_retries"] - 1:
                    delay = LOCAL_RETRY_CONFIG["api_retry_delay"] if "400" in err_str else LOCAL_RETRY_CONFIG["retry_delay"]
                    print(f"⚠️  API Error (attempt {api_attempt + 1}/{LOCAL_RETRY_CONFIG['api_max_retries']}): {err_str[:200]}")
                    print(f"   Waiting {delay}s before retry...")
                    time.sleep(delay)
                    continue
                last_api_error = api_error
                break

        if response_text is None:
            reason = f"API error after retries: {last_api_error}"
            current_agent["structured_trajectory"].append(
                {
                    "step": current_agent["step_count"] + 1,
                    "global_step": state["global_step_count"] + 1,
                    "thinking": "",
                    "action_string": "",
                    "raw_response": "",
                    "parse_error": str(last_api_error),
                    "failure_type": "api_error",
                    "image_path": image_path,
                    "communication": "",
                }
            )
            mark_agent_failure(state, current_agent_id, "api_error", reason)
            continue

        parsed = None
        parse_error = None
        for parse_attempt in range(LOCAL_RETRY_CONFIG["max_retries"]):
            try:
                parsed = parse_dual_agent_response(response_text, enable_summary=enable_summary)
                if parse_attempt > 0:
                    print(f"✓ Success after {parse_attempt + 1} parse attempts")
                break
            except ValueError as e:
                parse_error = e
                print(f"⚠️  Parse Error (parse attempt {parse_attempt + 1}/{LOCAL_RETRY_CONFIG['max_retries']}): {e}")
                if parse_attempt < LOCAL_RETRY_CONFIG["max_retries"] - 1:
                    print(f"   Waiting {LOCAL_RETRY_CONFIG['retry_delay']}s before re-calling VLM...")
                    time.sleep(LOCAL_RETRY_CONFIG["retry_delay"])
                    try:
                        response = current_agent["vlm"].invoke(messages)
                        response_text = response.content if hasattr(response, "content") else str(response)
                    except Exception as recall_error:
                        parse_error = recall_error
                    continue
                break

        if parsed is None:
            parse_error_text = str(parse_error)
            reason = f"Parse error: {parse_error}"
            current_agent["structured_trajectory"].append(
                {
                    "step": current_agent["step_count"] + 1,
                    "global_step": state["global_step_count"] + 1,
                    "thinking": "",
                    "action_string": "",
                    "raw_response": response_text[:2000],
                    "parse_error": str(parse_error),
                    "failure_type": "parse_error",
                    "image_path": image_path,
                    "communication": "",
                }
            )

            missing_action = (
                "Missing <ACTION> tag" in parse_error_text
                or "ACTION tag content is empty" in parse_error_text
            )
            if missing_action:
                handoff_agent_or_finish(
                    state,
                    current_agent_id,
                    "Missing ACTION after 3 retries; hand off to partner",
                    failure_type="model_error",
                )
            else:
                mark_agent_failure(state, current_agent_id, "parse_error", reason)
            continue

        communication_text = parsed["communication_text"]
        if communication_text:
            comm = {
                "sender": current_agent_id,
                "receiver": other_agent_id,
                "message": communication_text,
                "global_step": state["global_step_count"] + 1,
            }
            state["communication_history"].append(comm)
            state["message_queue"].append(comm)
            print(f"📨 Communication sent to {other_agent_id}: {communication_text[:120]}...")

        action_dict = parsed["parsed_action"]
        action_string = parsed["action_string"]
        thinking_text = parsed["thinking_text"]

        print(f"✓ Thinking: {thinking_text[:200]}{'...' if len(thinking_text) > 200 else ''}")
        print(f"✓ Action String: {action_string}")
        print(f"✓ Parsed Action: {action_dict}")

        trajectory_entry = {
            "step": current_agent["step_count"] + 1,
            "global_step": state["global_step_count"] + 1,
            "thinking": thinking_text,
            "action_string": action_string,
            "action": action_dict,
            "raw_response": response_text,
            "image_path": image_path,
            "communication": communication_text,
            "updated_summary": parsed.get("updated_summary", ""),
        }

        current_agent["step_count"] += 1
        record_system_step_progress(state, current_agent_id)
        state["current_turn_steps"] += 1

        if enable_summary and parsed.get("updated_summary"):
            current_agent["long_term_summary"] = parsed["updated_summary"]

        if action_dict.get("action_type") == "communication":
            trajectory_entry["reward"] = 0.0
            trajectory_entry["error_message"] = None
            current_agent["structured_trajectory"].append(trajectory_entry)
            current_agent["short_term_history"].append(
                {
                    "step": current_agent["step_count"],
                    "image_path": image_path,
                    "raw_response": response_text,
                }
            )
            current_agent["short_term_history"] = current_agent["short_term_history"][-max_history:]
            handoff_agent_or_finish(state, current_agent_id, "communication action")
            continue

        if action_dict.get("action_type") == "task_completion":
            trajectory_entry["reward"] = 0.0
            trajectory_entry["error_message"] = None
            current_agent["structured_trajectory"].append(trajectory_entry)
            current_agent["short_term_history"].append(
                {
                    "step": current_agent["step_count"],
                    "image_path": image_path,
                    "raw_response": response_text,
                }
            )

            if action_dict.get("action_name") == "DONE":
                success, score = perform_final_evaluation(env, task_config, observation)
                if success:
                    state["success"] = True
                    state["fail_reason"] = None
                    state["failure_type"] = None
                    print(f"✅ DONE verified (score={score:.2f})")
                    break
                current_agent["last_error_message"] = "DONE was rejected by evaluator"
                print(f"❌ DONE rejected by evaluator (score={score:.2f})")
                handoff_agent_or_finish(
                    state,
                    current_agent_id,
                    "Model claimed DONE but success conditions were not met",
                    failure_type="model_error",
                )
                continue

            handoff_agent_or_finish(
                state,
                current_agent_id,
                "Model indicated FAIL before task completion",
                failure_type="model_error",
            )
            continue

        thor_agent_id = AGENT_TO_THOR_ID.get(current_agent_id, 0)
        try:
            #       thor_agent_id   ；            
            if getattr(env, "agent_count", 1) > 1:
                observation, error_message = env.step_with_action_dict(
                    action_dict, thor_agent_id=thor_agent_id
                )
            else:
                observation, error_message = env.step_with_action_dict(action_dict)
        except Exception as e:
            observation = None
            error_message = str(e)

        reward = 0.0 if error_message else 0.1
        trajectory_entry["reward"] = reward
        trajectory_entry["error_message"] = error_message
        trajectory_entry["thor_agent_id"] = thor_agent_id
        current_agent["structured_trajectory"].append(trajectory_entry)
        current_agent["short_term_history"].append(
            {
                "step": current_agent["step_count"],
                "image_path": image_path,
                "raw_response": response_text,
            }
        )
        current_agent["short_term_history"] = current_agent["short_term_history"][-max_history:]

        #   ：         agent      ；         
        if observation is not None:
            if getattr(env, "agent_count", 1) > 1:
                current_agent["observation"] = observation
            else:
                state["agent_1"]["observation"] = observation
                state["agent_2"]["observation"] = observation

        #           agent，               
        state["last_acting_agent"] = current_agent_id

        if error_message:
            current_agent["consecutive_failures"] += 1
            current_agent["last_error_message"] = error_message
            print(f"  ⚠️  Action failed: {error_message}")
        else:
            current_agent["consecutive_failures"] = 0
            current_agent["last_error_message"] = None
            print("✓ Action executed successfully")

        if current_agent["consecutive_failures"] >= 4:
            mark_agent_failure(
                state,
                current_agent_id,
                "action_error",
                f"Consecutive {current_agent['consecutive_failures']} action failures (early stop)",
            )
            print(f"🛑 Early stop for {current_agent_id}: consecutive action failures")
            continue

        if observation is None:
            mark_agent_failure(state, current_agent_id, "env_error", "Environment step returned None")
            continue

        if current_agent["step_count"] >= current_agent["max_steps"] and not state["success"]:
            handoff_agent_or_finish(
                state,
                current_agent_id,
                "Reached max steps before task completion",
            )

    finalize_partial_system_step(state)

    final_observation = (
        state.get(state.get("current_agent", "agent_1"), {}).get("observation")
        or state.get("agent_1", {}).get("observation")
        or state.get("agent_2", {}).get("observation")
    )
    final_success, final_score = perform_final_evaluation(env, task_config, final_observation)
    if final_success:
        state["success"] = True
        state["fail_reason"] = None
        state["failure_type"] = None
        print(f"✅ Final terminal-state evaluation succeeded (score={final_score:.2f})")
    else:
        state["success"] = False
        if not state.get("fail_reason"):
            state["fail_reason"] = f"Final evaluation failed (score={final_score:.2f})"
        print(f"❌ Final terminal-state evaluation failed (score={final_score:.2f})")

    return state


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="ProcTHOR dual-agent main entry")
    parser.add_argument("--config", type=str, default="experiments/configs/procthor/dual/config_close_gpt-5.yaml", help="Base config file path")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Task ID(s), e.g. procthor00001")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--headless", action="store_true", help="Headless mode")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--switch-interval", type=int, default=1, help="Agent switch interval")
    parser.add_argument("--collaboration-mode", type=str, default="alternating", choices=["alternating", "sequential"], help="Dual-agent collaboration mode")
    parser.add_argument("--agent1", type=str, default=None, help="Agent 1 single-agent config path")
    parser.add_argument("--agent2", type=str, default=None, help="Agent 2 single-agent config path")
    parser.add_argument("--print-config", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    print(f"\n{'=' * 60}\n🔧         \n{'=' * 60}\n    : {args.config}")
    config = load_config(args.config)
    config_dict = config.get_all() if hasattr(config, "get_all") else config.config
    config_dict = apply_agent_model_overrides(config_dict, args.agent1, args.agent2)

    if args.print_config:
        print(json.dumps(config_dict, ensure_ascii=False, indent=2))
        return

    task_names = args.tasks or config.get_all_task_names()
    if not task_names:
        print("❌ No tasks specified and no task names from config")
        return

    dual_config = config_dict.setdefault("dual_agent", {})
    dual_config.setdefault("equal_collaboration", True)
    dual_config["collaboration_mode"] = args.collaboration_mode

    output_dir = args.output_dir or config.get("experiment.output_dir", "outputs")
    if args.output_dir and len(task_names) == 1:
        run_output_dir = args.output_dir
    else:
        run_output_dir = os.path.join(output_dir, f"dual_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_output_dir, exist_ok=True)

    agent_vlms, separate_models = get_agent_vlms(config_dict)

    all_results = []
    for task_idx, task_name in enumerate(task_names, 1):
        print(f"\n{'=' * 60}\n📋 Task {task_idx}/{len(task_names)}: {task_name}\n{'=' * 60}")
        task_config = config.apply_task_by_name(task_name, dual_agent=True)
        if args.max_steps is not None:
            task_config["max_steps"] = args.max_steps
            print(f"✓ max_steps overridden: {args.max_steps}")

        task_output_dir = os.path.join(run_output_dir, task_name) if len(task_names) > 1 else run_output_dir
        os.makedirs(task_output_dir, exist_ok=True)

        task_folder_path = task_config.get("task_folder_path") or os.path.join("tasks", task_name)
        init_actions = load_init_actions_from_task_folder(task_folder_path)

        full_config = deepcopy(config_dict)
        full_config["task"] = task_config
        full_config["init_actions"] = init_actions or []
        #      two-bodies：   agentCount=2；relocate agent2 near agent1
        full_config.setdefault("env", {})
        full_config["env"].setdefault("agent_count", 2)
        full_config.setdefault("dual_agent", {})
        full_config["dual_agent"].setdefault("relocate_agent2_near_agent1", True)
        full_config["dual_agent"].setdefault("second_agent_spawn_offset_m", 0.75)

        #    agent1/   agent2/      （AI2-THOR   ）
        os.makedirs(os.path.join(task_output_dir, "agent1"), exist_ok=True)
        os.makedirs(os.path.join(task_output_dir, "agent2"), exist_ok=True)

        try:
            env = ProcTHOREnvWrapper(
                scene_index=task_config.get("scene_index", 0),
                output_dir=task_output_dir,
                config=full_config,
                headless=args.headless,
            )
        except Exception as e:
            print(f"❌ Failed to create environment: {e}")
            import traceback

            traceback.print_exc()
            all_results.append({"task_name": task_name, "success": False, "step_count": 0, "fail_reason": str(e)})
            continue

        try:
            state = run_dual_agent_loop(
                env=env,
                agent_vlms=agent_vlms,
                task_config=task_config,
                task_output_dir=task_output_dir,
                config=full_config,
                collaboration_mode=args.collaboration_mode,
                switch_interval=args.switch_interval,
            )
            save_dual_conversation_log(state, task_output_dir)
            save_dual_episode_log(state, task_output_dir, env)
            all_results.append(
                {
                    "task_name": task_name,
                    "success": state.get("success", False),
                    "step_count": state.get("global_step_count", 0),
                    "fail_reason": state.get("fail_reason"),
                }
            )
            print(
                f"\n📊 Result: {'✅ Success' if state['success'] else '❌ Failure'}"
                f" | Steps: {state['global_step_count']}/{state['max_global_steps']}"
                f" | Agent1: {state['agent_1']['step_count']}"
                f" | Agent2: {state['agent_2']['step_count']}"
                f" | Separate models: {separate_models}"
            )
        except Exception as e:
            print(f"❌ Task error: {e}")
            import traceback

            traceback.print_exc()
            all_results.append({"task_name": task_name, "success": False, "step_count": 0, "fail_reason": str(e)})
        finally:
            env.close()

    success_count = sum(1 for r in all_results if r["success"])
    print(f"\n{'=' * 80}\n🎉 All Dual-Agent Tasks Completed\n{'=' * 80}")
    print(f"Total: {len(all_results)} | Success: {success_count} | Failure: {len(all_results) - success_count}")
    if all_results:
        print(f"Success Rate: {success_count / len(all_results) * 100:.1f}%")
    print(f"Output: {run_output_dir}\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
