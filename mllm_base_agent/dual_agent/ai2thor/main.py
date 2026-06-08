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
for _path in (_REPO_ROOT, _AI2THOR_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

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


def load_task_info(task_id: str) -> dict:
    """      
    
    Args:
        task_id:    ID，  ai2thor05002
        
    Returns:
              
    """
    task_dir = Path(__file__).resolve().parent / "tasks" / task_id
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
        return 0
    
    print(f"\n{'='*60}")
    print(f"📁         ({len(init_actions)}  )")
    print(f"{'='*60}")
    
    from .core.agent.graph import parse_action_string
    
    init_count = 0
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
            init_count += 1
            
            if error:
                print(f"     ⚠️  {error}")
        except Exception as e:
            print(f"     ❌   : {e}")
    
    print(f"✓    {init_count}       \n")
    return init_count


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
            env.reset(task_description=task_prompt)
            execute_init_actions(env, task_data['init_actions'])

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
