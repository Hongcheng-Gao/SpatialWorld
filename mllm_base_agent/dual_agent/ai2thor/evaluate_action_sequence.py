"""
            
                 

        ：
1.      (--headless):    CloudRendering   
2.      (--simulate):         
3.     :       

    ：
  #         golden_actions
  python -m dual_agent.evaluate_action_sequence --headless --task ai2thor001
  
  #           
  python -m dual_agent.evaluate_action_sequence --headless --task ai2thor001 --actions "MoveAhead,RotateLeft,DONE"
  
  #           
  python -m dual_agent.evaluate_action_sequence --headless --task ai2thor001 --collaboration-mode alternating
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

# Make both the ai2thor project and repository root importable when run directly.
_AI2THOR_ROOT = os.path.dirname(os.path.dirname(__file__))
_REPO_ROOT = os.path.dirname(_AI2THOR_ROOT)
for _path in (_REPO_ROOT, _AI2THOR_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from .config import load_config
from .core.agent.graph import (
    parse_action_string,
    perform_final_evaluation,
    _thor_agent_id_for_logical,
)


def parse_action_sequence(action_str: str) -> List[Dict[str, Any]]:
    """         """
    actions = []
    for action in action_str.split(','):
        action = action.strip()
        if not action:
            continue
        try:
            action_dict = parse_action_string(action)
            actions.append(action_dict)
        except ValueError as e:
            print(f"⚠️       '{action}'   : {e}")
            continue
    return actions


def extract_golden_actions_from_task(task_config: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """         golden_actions"""
    golden_actions = task_config.get('golden_actions')
    
    if not golden_actions:
        return None
    
    if isinstance(golden_actions, dict):
        actions = golden_actions.get('actions')
        if isinstance(actions, list) and len(actions) > 0:
            if all(isinstance(a, dict) and 'action_name' in a for a in actions):
                print(f"✓    golden_actions (     , {len(actions)}    )")
                return actions
            elif all(isinstance(a, str) for a in actions):
                print(f"✓    golden_actions (     , {len(actions)}    )")
                parsed_actions = []
                for action_str in actions:
                    if action_str.strip().upper() == "DONE":
                        parsed_actions.append({"action_type": "task_completion", "action_name": "DONE"})
                        break
                    try:
                        parsed_actions.append(parse_action_string(action_str))
                    except ValueError as e:
                        print(f"⚠️      : {action_str},   : {e}")
                return parsed_actions
    
    elif isinstance(golden_actions, str):
        print("✓    golden_actions (       )")
        return parse_action_sequence(golden_actions)
    
    return None


def create_env(scene: str, config: dict, output_dir: str, headless: bool = False):
    """      """
    from envs.ai2thor import AI2ThorEnvWrapper
    
    env_config = dict(config.get("env", {}))
    env_config.setdefault("agent_count", 2)

    if headless:
        env_config["platform"] = "CloudRendering"

    config_with_platform = config.copy()
    config_with_platform["env"] = env_config
    
    return AI2ThorEnvWrapper(
        scene=scene,
        grid_size=env_config.get("grid_size", 0.25),
        render_depth_image=env_config.get("render_depth", False),
        render_instance_segmentation=env_config.get("render_instance_segmentation", False),
        width=env_config.get("width", 800),
        height=env_config.get("height", 600),
        output_dir=output_dir,
        config=config_with_platform,
    )


def execute_dual_agent_actions(
    env, 
    actions: List[Dict[str, Any]], 
    collaboration_mode: str = "alternating",
    switch_interval: int = 1
) -> Dict[str, Any]:
    """          
    
    Args:
        env:     
        actions:     
        collaboration_mode:     
        switch_interval:     
        
    Returns:
              
    """
    results = {
        "agent_1_actions": [],
        "agent_2_actions": [],
        "global_step_count": 0,
        "agent_1_step_count": 0,
        "agent_2_step_count": 0,
        "errors": [],
        "observations": [],
    }
    
    current_agent = "agent_1"
    agent_step_counts = {"agent_1": 0, "agent_2": 0}
    last_obs: Dict[str, Any] = {"agent_1": None, "agent_2": None}
    if getattr(env, "agent_count", 1) > 1 and hasattr(env, "get_observation_for_agent"):
        last_obs["agent_1"] = env.get_observation_for_agent(0)
        last_obs["agent_2"] = env.get_observation_for_agent(1)
    
    print(f"\n{'='*60}")
    print(f"🤝            ({len(actions)}    )")
    print(f"    : {collaboration_mode}")
    print(f"{'='*60}\n")
    
    for i, action_dict in enumerate(actions):
        action_type = action_dict.get("action_type")
        action_name = action_dict.get("action_name")
        
        #       
        if action_type == "task_completion":
            print(f"Step {i+1}:        - {action_name}")
            break
        
        #              
        if collaboration_mode == "alternating":
            if agent_step_counts[current_agent] >= switch_interval:
                current_agent = "agent_2" if current_agent == "agent_1" else "agent_1"
                print(f"\n🔄     {current_agent}\n")
        
        #     
        print(f"Step {i+1} [{current_agent}]: {action_name}", end="")
        if action_dict.get("object_type"):
            print(f"({action_dict['object_type']})", end="")
        print()
        
        try:
            thor_id = _thor_agent_id_for_logical(current_agent)
            prior = last_obs.get(current_agent)
            vision_meta = (
                prior.metadata
                if prior is not None and getattr(prior, "metadata", None)
                else None
            )
            observation, error_message = env.step_with_action_dict(
                action_dict,
                thor_agent_id=thor_id,
                vision_metadata=vision_meta,
            )
            if observation is not None:
                last_obs[current_agent] = observation
            
            #     
            action_record = {
                "step": i + 1,
                "action": action_dict,
                "success": observation.reward > 0 if observation else False,
                "error": error_message,
            }
            
            if current_agent == "agent_1":
                results["agent_1_actions"].append(action_record)
                results["agent_1_step_count"] += 1
            else:
                results["agent_2_actions"].append(action_record)
                results["agent_2_step_count"] += 1
            
            results["global_step_count"] += 1
            agent_step_counts[current_agent] += 1
            
            if error_message:
                print(f"  ⚠️  {error_message}")
                results["errors"].append({"step": i+1, "error": error_message})
            else:
                print("  ✓   ")
                
        except Exception as e:
            print(f"  ❌     : {e}")
            results["errors"].append({"step": i+1, "error": str(e)})
    
    return results


def evaluate_dual_agent_sequence(
    task_name: str,
    actions: List[Dict[str, Any]],
    config,
    headless: bool = False,
    simulate: bool = False,
    collaboration_mode: str = "alternating",
) -> Dict[str, Any]:
    """          
    
    Args:
        task_name:     
        actions:     
        config:   
        headless:       
        simulate:       
        collaboration_mode:     
        
    Returns:
            
    """
    #       
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/dual_eval_{task_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    #       
    task_config = config.apply_task_by_name(task_name)
    scene = task_config.get("scene", "FloorPlan1")
    task_description = task_config.get("instruction") or task_config.get("description", "")
    
    print(f"\n{'='*60}")
    print("📋           ")
    print(f"{'='*60}")
    print(f"  : {task_name}")
    print(f"  : {scene}")
    print(f"  : {task_description}")
    print(f"   : {len(actions)}")
    print(f"    : {collaboration_mode}")
    print(f"{'='*60}\n")
    
    if simulate:
        print("⚠️       -          ")
        return {
            "task_name": task_name,
            "mode": "simulate",
            "actions_count": len(actions),
            "success": None,
        }
    
    #     
    try:
        env = create_env(scene, config.get_all(), output_dir, headless)
    except Exception as e:
        print(f"❌       : {e}")
        return {"task_name": task_name, "error": str(e)}
    
    try:
        #     
        env.reset(task_description, scene=scene)
        
        #        
        task_folder = os.path.join("tasks", task_name)
        init_file = os.path.join(task_folder, "init.json")
        
        if os.path.exists(init_file):
            print(f"📁        : {init_file}")
            with open(init_file, 'r', encoding='utf-8') as f:
                init_data = json.load(f)
            
            if isinstance(init_data, list):
                init_actions = init_data
            elif isinstance(init_data, dict):
                init_actions = init_data.get('actions', [])
                if init_data.get('scene'):
                    scene = init_data['scene']
                    env.reset(task_description, scene=scene)
            else:
                init_actions = []
            
            for action_str in init_actions:
                if action_str.strip().upper() == "DONE":
                    break
                try:
                    action_dict = parse_action_string(action_str)
                    env.step_with_action_dict(action_dict, thor_agent_id=0)
                except Exception as e:
                    print(f"  ⚠️         : {e}")

        if getattr(env, "agent_count", 1) > 1 and hasattr(
            env, "relocate_second_agent_near_agent1"
        ):
            env.relocate_second_agent_near_agent1()

        #           
        exec_results = execute_dual_agent_actions(
            env, actions, collaboration_mode
        )
        
        #     
        
        #            
        final_observation = env._get_observation()
        
        eval_state = {
            "config": config.get_all(),
            "env": env,
            "current_agent": "agent_1",
            "agent_1": {"observation": final_observation},
            "agent_2": {"observation": final_observation},
        }
        
        success, score = perform_final_evaluation(eval_state)
        
        #     
        result = {
            "task_name": task_name,
            "scene": scene,
            "collaboration_mode": collaboration_mode,
            "success": success,
            "score": score,
            "global_step_count": exec_results["global_step_count"],
            "agent_1_step_count": exec_results["agent_1_step_count"],
            "agent_2_step_count": exec_results["agent_2_step_count"],
            "errors_count": len(exec_results["errors"]),
            "output_dir": output_dir,
        }
        
        #     
        print(f"\n{'='*60}")
        print("📊     ")
        print(f"{'='*60}")
        print(f"  : {'✅  ' if success else '❌  '}")
        print(f"  : {score:.2f}")
        print(f"   : {exec_results['global_step_count']}")
        print(f"Agent 1   : {exec_results['agent_1_step_count']}")
        print(f"Agent 2   : {exec_results['agent_2_step_count']}")
        print(f"   : {len(exec_results['errors'])}")
        print(f"{'='*60}\n")
        
        #     
        result_file = os.path.join(output_dir, "dual_eval_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾      : {result_file}")
        
        return result
        
    except Exception as e:
        print(f"❌       : {e}")
        import traceback
        traceback.print_exc()
        return {"task_name": task_name, "error": str(e)}
        
    finally:
        env.close()


def main():
    """   """
    parser = argparse.ArgumentParser(
        description="          ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="      ",
    )
    
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="    ",
    )
    
    parser.add_argument(
        "--actions",
        type=str,
        default=None,
        help="       （    ）",
    )
    
    parser.add_argument(
        "--action-file",
        type=str,
        default=None,
        help="        ",
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="    ",
    )
    
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="    ",
    )
    
    parser.add_argument(
        "--collaboration-mode",
        type=str,
        default="alternating",
        choices=["alternating", "sequential"],
        help="    ",
    )
    
    args = parser.parse_args()
    
    #     
    config = load_config(args.config)
    
    #       
    actions = None
    
    if args.actions:
        actions = parse_action_sequence(args.actions)
    elif args.action_file:
        with open(args.action_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        actions = parse_action_sequence(content)
    else:
        #         
        task_config = config.apply_task_by_name(args.task)
        actions = extract_golden_actions_from_task(task_config)
        
        if not actions:
            #     task.json     
            task_folder = os.path.join("tasks", args.task)
            task_file = os.path.join(task_folder, "task.json")
            
            if os.path.exists(task_file):
                with open(task_file, 'r', encoding='utf-8') as f:
                    task_json = json.load(f)
                actions = extract_golden_actions_from_task(task_json)
    
    if not actions:
        print("❌        ")
        print("    --actions   --action-file   ，           golden_actions")
        return
    
    #     
    evaluate_dual_agent_sequence(
        task_name=args.task,
        actions=actions,
        config=config,
        headless=args.headless,
        simulate=args.simulate,
        collaboration_mode=args.collaboration_mode,
    )
    
    print("\n✓     ")


if __name__ == "__main__":
    main()
