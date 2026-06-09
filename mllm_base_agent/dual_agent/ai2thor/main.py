#!/usr/bin/env python3
"""
          -       
                 

    :
  #       
  python mllm_base_agent/dual_agent/ai2thor/main.py --task ai2thor05002
  
  #       
  python mllm_base_agent/dual_agent/ai2thor/main.py --task ai2thor05001 ai2thor05002 ai2thor05003
  
  #            
  python mllm_base_agent/dual_agent/ai2thor/main.py --task ai2thor05002 --max-steps 40 --switch-interval 3
  
  #          
  python mllm_base_agent/dual_agent/ai2thor/main.py --task ai2thor05002 --config experiments/configs/ai2thor/dual/config_close_gpt-5.yaml
"""


"""
python "./dual_agent/run_benchmark.py" \
  --csv "./experiments/csv/ai2thor/Spatial-Annotation-ai2thor-Kimi-K2.5.csv" \
  --config "./experiments/configs/ai2thor/dual/config_close_kimi-k25.yaml"
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

_AI2THOR_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (str(_AI2THOR_ROOT), str(_REPO_ROOT)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(_REPO_ROOT))
sys.path.append(str(_AI2THOR_ROOT))

from actions.max_steps import compute_max_steps_from_n

from .config import load_config
from .core.llm import get_dual_agent_vlms, get_vlm, use_separate_agent_models
from .core.agent.graph import create_dual_agent_graph
from .core.agent.state import initialize_dual_agent_state


def get_vlm_display_name(vlm) -> str:
    """Get a human-readable model name for different chat model implementations."""
    return (
        getattr(vlm, "model_name", None)
        or getattr(vlm, "_model", None)
        or getattr(vlm, "model", None)
        or type(vlm).__name__
    )


def resolve_task_dir(task_id: str) -> Path:
    """Resolve a dual-agent task id or explicit task directory."""
    raw_path = Path(task_id).expanduser()
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([Path.cwd() / raw_path, _REPO_ROOT / raw_path])

    task_names = [raw_path.name]
    if "_" in raw_path.name:
        task_names.append(raw_path.name.replace("_", ""))
    elif raw_path.name.lower().startswith("ai2thor"):
        task_names.append(raw_path.name.replace("ai2thor", "ai2thor_", 1))

    for root in (
        _AI2THOR_ROOT / "tasks",
        _REPO_ROOT / "data" / "ai2thor" / "tasks",
    ):
        for name in task_names:
            candidates.append(root / name)

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_dir() and (candidate / "task.json").exists():
            return candidate

    return _AI2THOR_ROOT / "tasks" / raw_path.name


def load_task_info(task_id: str) -> dict:
    """      
    
    Args:
        task_id:    ID，  ai2thor05002
        
    Returns:
              
    """
    task_dir = resolve_task_dir(task_id)
    task_file = task_dir / "task.json"
    init_file = task_dir / "init.json"
    
    if not task_file.exists():
        raise FileNotFoundError(f"       : {task_file}")
    
    with open(task_file, 'r', encoding='utf-8') as f:
        task_info = json.load(f)
    
    #        （   ）
    init_actions = []
    scene_name = None
    if init_file.exists():
        with open(init_file, 'r', encoding='utf-8') as f:
            init_data = json.load(f)
            if isinstance(init_data, list):
                init_actions = init_data
            elif isinstance(init_data, dict):
                init_actions = init_data.get('actions', [])
                scene_name = init_data.get('scene')
    
    #    DONE   
    if init_actions and init_actions[-1].strip().upper() == "DONE":
        init_actions = init_actions[:-1]
    
    golden_actions = task_info.get("golden_actions", {}) or {}
    golden_action_list = golden_actions.get("actions", []) or []
    counted_actions = [
        action
        for action in golden_action_list
        if str(action).strip().upper() != "DONE"
    ]
    # n   10+2n     ：             （   Done） 
    # task.json   golden_actions.steps    len(actions)-1   1（    Done   ），
    #     actions   ，          10+2n        
    if counted_actions:
        golden_action_steps = len(counted_actions)
    else:
        steps_field = golden_actions.get("steps")
        golden_action_steps = int(steps_field) if isinstance(steps_field, int) else None

    recommended_max_steps = (
        compute_max_steps_from_n(golden_action_steps)
        if isinstance(golden_action_steps, int)
        else None
    )

    return {
        "task_id": task_id,
        "instruction": task_info.get("instruction", ""),
        "scene": scene_name or task_info.get("scene", "FloorPlan1"),
        "init_actions": init_actions,
        "task_info": task_info,
        "golden_action_steps": golden_action_steps,
        "recommended_max_steps": recommended_max_steps,
    }


def execute_init_actions(env, init_actions: list):
    """       """
    if not init_actions:
        return 0, None
    
    print(f"\n{'='*60}")
    print(f"📁         ({len(init_actions)}  )")
    print(f"{'='*60}")
    
    from .core.agent.graph import parse_action_string
    
    init_count = 0
    last_observation = None
    for i, action_str in enumerate(init_actions, 1):
        action_str = action_str.strip()
        if not action_str or action_str.upper() == "DONE":
            break
        
        print(f"  {i}. {action_str}")
        try:
            action_dict = parse_action_string(action_str)
            #    init         ：    agentId=0（Agent 1    ）  
            observation, error = env.step_with_action_dict(
                action_dict, thor_agent_id=0
            )
            last_observation = observation
            init_count += 1
            
            if error:
                print(f"     ⚠️  {error}")
        except Exception as e:
            print(f"     ❌   : {e}")
    
    print(f"✓    {init_count}       \n")
    return init_count, last_observation


def load_config_dict(config_path: str) -> dict:
    """            """
    config = load_config(config_path)
    return config.get_all() if hasattr(config, "get_all") else config.config


def extract_model_config_from_file(config_path: str) -> dict:
    """             model.vlm    """
    config_dict = load_config_dict(config_path)
    model_config = config_dict.get("model", {}).get("vlm", {})
    if not model_config:
        raise ValueError(f"        model.vlm: {config_path}")
    return dict(model_config)


def apply_agent_model_overrides(
    config_dict: dict,
    agent1_config_path: str = None,
    agent2_config_path: str = None,
) -> dict:
    """   --agent1/--agent2                   """
    if not agent1_config_path and not agent2_config_path:
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

    if "model" not in config_dict:
        config_dict["model"] = {}
    if "dual_agent" not in config_dict:
        config_dict["dual_agent"] = {}

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


def compute_recursion_limit(per_agent_max_steps: int) -> int:
    """            coordinator/think/act/evaluate    

       200    10+2n             ，       
         dual_episode_*.json（benchmark      null） 
    """
    total_action_cap = max(1, 2 * int(per_agent_max_steps))
    #         4–6       ，      Pass/  
    return max(500, 15 * total_action_cap)


def _json_safe(value):
    """Convert runner state fragments to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    return str(value)


def save_dual_episode_log(final_state: dict, task_id: str, task_data: dict, output_dir: str) -> Path:
    """Persist the dual-agent result format expected by the benchmark wrapper."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene = task_data.get("scene", "UnknownScene")
    safe_scene = str(scene).replace(" ", "_").replace("/", "_")[:80]
    safe_task = str(task_id).replace(" ", "_").replace("/", "_")[:80]
    path = Path(output_dir) / f"dual_episode_{safe_scene}_{safe_task}_{timestamp}.json"

    success = final_state.get("global_success")
    if success is None:
        success = final_state.get("success", False)
    fail_reason = final_state.get("global_fail_reason") or final_state.get("fail_reason")
    failure_type = final_state.get("failure_type")
    if not success and not failure_type:
        failure_type = "model_error"
    trajectory = final_state.get("structured_trajectory", []) or []
    fallback_steps = final_state.get("step_count") or len(trajectory)
    global_step_count = final_state.get("global_step_count") or fallback_steps or 0
    global_action_count = final_state.get("global_action_count") or len(trajectory) or global_step_count
    agent_1_steps = final_state.get("agent_1", {}).get("step_count", 0) or global_action_count
    agent_2_steps = final_state.get("agent_2", {}).get("step_count", 0) or 0

    episode = {
        "task_id": task_id,
        "task": task_data.get("instruction", ""),
        "scene": scene,
        "mode": "dual_agent",
        "success": bool(success),
        "failure_type": failure_type,
        "fail_reason": fail_reason,
        "global_step_count": global_step_count,
        "global_action_count": global_action_count,
        "agent_1_steps": agent_1_steps,
        "agent_2_steps": agent_2_steps,
        "turn_count": final_state.get("turn_count") or global_step_count,
        "communication_history": _json_safe(final_state.get("communication_history", [])),
        "trajectory": _json_safe(trajectory),
        "action_sequence": (
            final_state.get("env").get_action_sequence()
            if hasattr(final_state.get("env"), "get_action_sequence")
            else None
        ),
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "token_usage": _json_safe(final_state.get("token_usage", {})),
            "max_steps": final_state.get("max_steps"),
            "max_global_steps": final_state.get("max_global_steps"),
        },
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(episode, handle, ensure_ascii=False, indent=2)
    return path


def main():
    #            ：      /    .env，
    #           dual_agent/.env       
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    
    parser = argparse.ArgumentParser(
        description="         -       ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  :
  #       
  python python -m mllm_base_agent.dual_agent.ai2thor.main --task ai2thor05002
  
  #       
  python python -m mllm_base_agent.dual_agent.ai2thor.main --task ai2thor05001 ai2thor05002
  
  #         40
  python python -m mllm_base_agent.dual_agent.ai2thor.main --task ai2thor05002 --max-steps 40
  
  #            5  
  python python -m mllm_base_agent.dual_agent.ai2thor.main --task ai2thor05002 --switch-interval 5
  
  #               
  python mllm_base_agent/dual_agent/ai2thor/main.py --task ai2thor05002 --config experiments/configs/ai2thor/dual/config_close_gpt-5.yaml --agent1 experiments/configs/ai2thor/config_close_gpt-5.yaml

  #       agent       
  python mllm_base_agent/dual_agent/ai2thor/main.py --task ai2thor05002 --config experiments/configs/ai2thor/dual/config_close_gpt-5.yaml --agent1 experiments/configs/ai2thor/config_close_Gemini-3.1-Pro-Preview.yaml --agent2 experiments/configs/ai2thor/config_kimi-a3b.yaml
        """
    )
    
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        required=True,
        help="   ID（  ai2thor05002）"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/ai2thor/dual/config_close_gpt-5.yaml",
        help="        （  /  /      ）"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="     （       ）"
    )
    
    parser.add_argument(
        "--switch-interval",
        type=int,
        default=None,
        help="       （  ）"
    )

    parser.add_argument(
        "--agent1",
        type=str,
        default=None,
        help="Agent 1            ；      agent config，       "
    )

    parser.add_argument(
        "--agent2",
        type=str,
        default=None,
        help="Agent 2            ；   agent1   ，       "
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="    ；           ，           task_id    ",
    )

    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=None,
        help="    ；    10+2n     （  15×     ）",
    )
    
    args = parser.parse_args()
    
    #     
    print(f"\n{'='*60}")
    print("🔧     ")
    print(f"{'='*60}")
    print(f"    : {args.config}")
    
    config = load_config(args.config)
    
    #    config.get_all()         （     runner   ）
    config_dict = config.get_all() if hasattr(config, 'get_all') else config.config
    config_dict = apply_agent_model_overrides(config_dict, args.agent1, args.agent2)
    dual_config = config_dict.get("dual_agent", {})
    
    #    equal_collaboration   
    if not dual_config.get("equal_collaboration", False):
        print("\n⚠️    :     equal_collaboration = false")
        print("                   ")
    
    #        （     golden_actions，     10+2n     ）
    default_max_steps = dual_config.get("max_global_steps", 60)
    switch_interval = dual_config.get("switch_interval", 1)
    
    #          
    if args.max_steps:
        default_max_steps = args.max_steps
        if "dual_agent" not in config_dict:
            config_dict["dual_agent"] = {}
        config_dict["dual_agent"]["max_global_steps"] = default_max_steps
        print(f"✓        : {default_max_steps}")
    
    if args.switch_interval:
        switch_interval = args.switch_interval
        if "dual_agent" not in config_dict:
            config_dict["dual_agent"] = {}
        config_dict["dual_agent"]["switch_interval"] = switch_interval
        print(f"✓        : {switch_interval}")
    
    print("\n     :      (Equal Collaboration)")
    print(f"          : {default_max_steps}")
    print(f"    : {switch_interval}  ")
    
    #       
    for task_id in args.task:
        print(f"\n{'='*60}")
        print(f"📋     : {task_id}")
        print(f"{'='*60}")
        
        try:
            task_data = load_task_info(task_id)
        except FileNotFoundError as e:
            print(f"❌ {e}")
            continue
        
        task_prompt = task_data.get('instruction', '')
        golden_action_steps = task_data.get("golden_action_steps")
        recommended_max_steps = task_data.get("recommended_max_steps")
        task_max_steps = args.max_steps or recommended_max_steps or default_max_steps
        print(f"  : {task_prompt}")
        print(f"  : {task_data['scene']}")
        print(f"     : {len(task_data['init_actions'])}  ")
        if golden_action_steps is not None:
            print(f"golden n (       ，   Done): {golden_action_steps}")
            print(f"         (10+2n): {task_max_steps}（        {2 * task_max_steps}）")
        else:
            print(f"        : {task_max_steps}（        {2 * task_max_steps}）")

        recursion_limit = args.recursion_limit or compute_recursion_limit(task_max_steps)
        print(f"    : {recursion_limit}")
        
        #            （    ）
        config_dict["task"] = task_data.get('task_info', {})
        config_dict.setdefault("dual_agent", {})
        config_dict["dual_agent"]["max_global_steps"] = task_max_steps
        
        #     
        print(f"\n{'='*60}")
        print("🌍      ")
        print(f"{'='*60}")
        
        from envs.ai2thor import AI2ThorEnvWrapper
        
        if args.output_dir:
            if len(args.task) == 1:
                output_dir = args.output_dir
            else:
                output_dir = str(Path(args.output_dir) / task_id)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"dual_agent/outputs/task_{task_id}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(Path(output_dir) / "agent1", exist_ok=True)
        os.makedirs(Path(output_dir) / "agent2", exist_ok=True)
        
        env_config = config.get("env", {})
        config_dict.setdefault("env", env_config)
        #     ：           （AI2-THOR agentCount=2）；   YAML   env.agent_count   
        config_dict["env"].setdefault("agent_count", 2)

        env = AI2ThorEnvWrapper(
            scene=task_data['scene'],
            grid_size=env_config.get("grid_size", 0.25),
            width=env_config.get("width", 800),
            height=env_config.get("height", 600),
            output_dir=output_dir,
            config=config_dict,
        )
        
        print(f"✓        : {task_data['scene']}")
        
        try:
            #   reset     ，   agent 1    init（   init    reset   ）
            observation = env.reset(task_description=task_prompt)
            _, init_observation = execute_init_actions(env, task_data['init_actions'])
            if init_observation is not None:
                observation = init_observation

            if getattr(env, "agent_count", 1) > 1 and hasattr(
                env, "relocate_second_agent_near_agent1"
            ):
                env.relocate_second_agent_near_agent1()

            #    VLM
            print(f"{'='*60}")
            print("🤖     VLM")
            print(f"{'='*60}")
            
            model_config = config_dict.get("model", {}).get("vlm", {})
            separate_models = use_separate_agent_models(config_dict)

            if separate_models:
                agent_1_vlm, agent_2_vlm = get_dual_agent_vlms(config_dict)
                vlm = agent_1_vlm
                agent_vlms = {
                    "agent_1": agent_1_vlm,
                    "agent_2": agent_2_vlm,
                }
                print("✓         ")
                print(f"  Agent 1   : {get_vlm_display_name(agent_1_vlm)}")
                print(f"  Agent 2   : {get_vlm_display_name(agent_2_vlm)}")
            else:
                vlm = get_vlm(agent_config=model_config)
                agent_vlms = {
                    "agent_1": vlm,
                    "agent_2": vlm,
                }
                print(f"✓     : {get_vlm_display_name(vlm)}")
            
            #          
            initial_state = initialize_dual_agent_state(
                task_prompt=task_prompt,
                env=env,
                vlm=vlm,
                agent_vlms=agent_vlms,
                config=config_dict,
                max_global_steps=task_max_steps,
                collaboration_mode=dual_config.get("collaboration_mode", "alternating"),
                run_output_dir=output_dir,
                skip_env_reset=True,
            )
            initial_state["observation"] = observation
            
            #         
            print(f"\n{'='*60}")
            print("🚀           ")
            print(f"{'='*60}")
            print("Agent 1   Agent 2         ")
            print(f"    :   {switch_interval}  ")
            print(f"        : {task_max_steps} |        : {2 * task_max_steps}")
            print(f"{'='*60}\n")
            
            app = create_dual_agent_graph()
            final_state = app.invoke(
                initial_state, config={"recursion_limit": recursion_limit}
            )
            dual_episode_path = save_dual_episode_log(
                final_state, task_id, task_data, output_dir
            )
            print(f"✓ Dual episode saved: {dual_episode_path}")
            
            #     
            print(f"\n{'='*60}")
            print(f"📊     : {task_id}")
            print(f"{'='*60}")
            
            success = final_state.get("global_success", False)
            result_symbol = "✅" if success else "❌"
            result_text = "  " if success else "  "
            
            print(f"{result_symbol}   : {result_text}")
            print(f"     (global_step_count): {final_state.get('global_step_count', 0)}")
            print(
                f"      : {final_state.get('global_action_count', 0)}"
                f" / {2 * task_max_steps}"
            )
            print(f"Agent 1   : {final_state.get('agent_1', {}).get('step_count', 0)}")
            print(f"Agent 2   : {final_state.get('agent_2', {}).get('step_count', 0)}")
            print(f"    : {len(final_state.get('communication_history', []))}")
            
            if not success:
                fail_reason = final_state.get("global_fail_reason", "  ")
                print(f"    : {fail_reason}")
            
            print(f"\n    : {output_dir}")
            print(f"{'='*60}\n")
            
        finally:
            env.close()


if __name__ == "__main__":
    main()
