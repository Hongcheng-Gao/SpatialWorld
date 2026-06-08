"""
         
               ，      

   spatial-planning/scripts/ai2thor/evaluate_action_sequence.py

    :
python scripts/evaluate_action_sequence.py --task procthor00001 --actions "MoveAhead,RotateLeft,DONE"
"""

import os
import platform
import sys

#     ：   prior/git   ~/.git-credentials prior      ProcTHOREnvWrapper     （     ） 
if not os.environ.get("GIT_CONFIG_GLOBAL"):
    try:
        import tempfile
        _fd, _path = tempfile.mkstemp(suffix=".gitconfig", prefix="procthor_")
        os.close(_fd)
        with open(_path, "w") as _f:
            _f.write("[credential]\n\thelper = \n")
        os.environ["GIT_CONFIG_GLOBAL"] = _path
    except Exception:
        pass

import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Make both the ProcTHOR subproject and the merged repo package importable.
PROCTHOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROCTHOR_ROOT.parent
for import_root in (str(REPO_ROOT), str(PROCTHOR_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from mllm_base_agent.console import configure_utf8_stdio

configure_utf8_stdio()

from core.action_parser import parse_action_string
from envs.procthor_wrapper import ProcTHOREnvWrapper
from evaluation.procthor.base import create_evaluator_from_config


def _parse_action_sequence_with_logging(action_str: str) -> List[Dict[str, Any]]:
    """         ，             （  spatial-planning     ） """
    actions = []
    for action in action_str.split(","):
        action = action.strip()
        if not action:
            continue
        try:
            actions.append(parse_action_string(action))
        except ValueError as e:
            print(f"⚠️  Failed to parse action '{action}': {e}")
            continue
    return actions


def parse_action_sequence(action_str: str) -> List[Dict[str, Any]]:
    """         （    ），           """
    return _parse_action_sequence_with_logging(action_str)


def load_init_actions_for_task(task_file: str) -> Optional[List[Dict[str, Any]]]:
    """        init.json            
      task.json      init.json          ，          
    
    Args:
        task_file:       ，  tasks/procthor109/task.json
        
    Returns:
                  ，  init.json          None
    """
    task_dir = os.path.dirname(os.path.abspath(task_file))
    init_path = os.path.join(task_dir, "init.json")
    if not os.path.isfile(init_path):
        return None
    try:
        with open(init_path, "r", encoding="utf-8") as f:
            init_data = json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load init.json: {init_path} -> {e}")
        return None
    if isinstance(init_data, list):
        action_strings = init_data
    elif isinstance(init_data, dict):
        action_strings = init_data.get("actions", [])
    else:
        return None
    if not action_strings:
        return None
    parsed = []
    for s in action_strings:
        s = (s or "").strip()
        if s.upper() == "DONE":
            parsed.append({"action_type": "task_completion", "action_name": "DONE"})
            break
        try:
            parsed.append(parse_action_string(s))
        except (ValueError, TypeError):
            continue
    if not parsed:
        return None
    print(f"✓ Loaded init actions: {init_path} ({len(parsed)} actions)")
    return parsed


def get_task_and_init_config(task_path_or_id: str) -> tuple:
    """               init   ，       agent      
    
        （agent   ）:
        task_config, init_actions = get_task_and_init_config("procthor109")
        env = ProcTHOREnvWrapper(..., config={"task": task_config, "init_actions": init_actions or []})
        obs = env.reset(instruction)  # reset       init_actions，     
        #    agent   post-init        ，    create_evaluator_from_config(task_config)   
    
    Args:
        task_path_or_id:           id，  "procthor109"   "tasks/procthor109/task.json"
        
    Returns:
        (task_config, init_actions):       ；init     （   None）
    """
    task_file = None
    if os.path.isfile(task_path_or_id):
        task_file = task_path_or_id
    elif os.path.isfile(f"tasks/{task_path_or_id}/task.json"):
        task_file = f"tasks/{task_path_or_id}/task.json"
    if not task_file:
        return {}, None
    try:
        with open(task_file, "r", encoding="utf-8") as f:
            task_config = json.load(f)
    except Exception:
        return {}, None
    init_actions = load_init_actions_for_task(task_file)
    # max_steps = 10 + 2 * n（n=    golden actions  ）
    _apply_max_steps_from_golden(task_config)
    return task_config, init_actions


def _apply_max_steps_from_golden(task_config: Dict[str, Any]) -> None:
    """      golden_actions  ，  max_steps = 10 + 2 * n    
    n     actions      Done     ，      steps         task_config 
    """
    ga = task_config.get("golden_actions")
    if not ga:
        return
    n: Optional[int] = None
    if isinstance(ga, dict):
        actions = ga.get("actions") or []
        counted = [a for a in actions if str(a).strip().upper() != "DONE"]
        if counted:
            n = len(counted)
        else:
            steps_field = ga.get("steps")
            if isinstance(steps_field, int):
                n = int(steps_field)
    if n is not None:
        try:
            task_config["max_steps"] = 10 + 2 * int(n)
        except (TypeError, ValueError):
            pass


def extract_golden_actions_from_task(task_config: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """         golden_actions
    
    Args:
        task_config:       
        
    Returns:
              ，         None
    """
    golden_actions = task_config.get("golden_actions")
    
    if not golden_actions:
        return None
    
    #   1:     （   steps   actions）
    if isinstance(golden_actions, dict):
        actions = golden_actions.get("actions")
        if isinstance(actions, list) and len(actions) > 0:
            #           （    ）
            if all(isinstance(a, str) for a in actions):
                steps_field = golden_actions.get("steps", len(actions))
                if steps_field != len(actions):
                    print(
                        f"⚠️  Step count mismatch: steps={steps_field}, "
                        f"actions={len(actions)}"
                    )
                    print(f"   Using actions count: {len(actions)}")
                print(f"✓ Loaded golden_actions from task config ({len(actions)} actions)")
                if len(actions) <= 10:
                    print(f"   Actions: {actions}")
                else:
                    print(f"   First 5 actions: {actions[:5]}")
                    print(f"   Last 5 actions: {actions[-5:]}")
                    print(f"   Total actions: {len(actions)}")
                parsed_actions = []
                for idx, action_str in enumerate(actions, 1):
                    if action_str.strip().upper() == "DONE":
                        parsed_actions.append({
                            "action_type": "task_completion",
                            "action_name": "DONE",
                        })
                        break
                    
                    try:
                        action_dict = parse_action_string(action_str)
                        parsed_actions.append(action_dict)
                    except ValueError as e:
                        print(
                            f"⚠️  Failed to parse golden action "
                            f"{idx}/{len(actions)} '{action_str}': {e}"
                        )
                        continue
                return parsed_actions if parsed_actions else None
    
    #   2:          
    elif isinstance(golden_actions, str):
        print("✓ Loaded golden_actions from task config string")
        return parse_action_sequence(golden_actions)
    
    return None


def perform_final_evaluation(
    env, task_config: dict, observation
) -> tuple:
    """      
    
    Args:
        env:     
        task_config:     
        observation:      
        
    Returns:
        Tuple of (success: bool, score: float)
    """
    try:
        if not task_config:
            print("⚠️  No task configuration found, cannot evaluate")
            return False, 0.0
        
        #      
        evaluator = create_evaluator_from_config(task_config)
        
        #      
        if not observation or not observation.metadata:
            print("⚠️  No observation metadata available for evaluation")
            return False, 0.0
        
        metadata = observation.metadata
        
        #     
        score = evaluator.evaluate(env, metadata)
        
        #          
        print(f"\n📊 Final evaluation:")
        print(f"   Score: {score:.2f}")
        
        #          ，         
        if hasattr(evaluator, 'conditions'):
            print(f"   Conditions: {len(evaluator.conditions)}")
            print(f"   Logic: {evaluator.logic}")
            for i, condition in enumerate(evaluator.conditions, 1):
                print(
                    f"   Condition {i}: {condition.get('type')} - "
                    f"{condition.get('description', condition)}"
                )
        
        #      score >= 1.0
        return score >= 1.0, score
    
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return False, 0.0


def copy_task_info_to_clipboard(task_config: Dict[str, Any]) -> None:
    """           （     ，      Excel）
    
    Args:
        task_config:       
    """
    #     
    task_id = task_config.get("task_id", "")
    task_name = task_config.get("task_name", "")
    instruction = task_config.get("instruction", "").replace("\n", " ").replace("\t", " ")  #             
    
    #    Golden Action
    golden_actions = task_config.get("golden_actions", {})
    golden_action_str = ""
    step_number = ""
    
    if isinstance(golden_actions, dict):
        actions = golden_actions.get("actions", [])
        if isinstance(actions, list) and len(actions) > 0:
            #            ，     
            golden_action_str = ",".join(str(action) for action in actions)
        step_number = str(golden_actions.get("steps", ""))
    elif isinstance(golden_actions, str):
        golden_action_str = golden_actions
        step_number = str(len(golden_actions.split(",")))
    
    #     
    evaluation = ""  #  
    category = task_config.get("Category", "")
    level = task_config.get("Level", "")
    evaluation_type = task_config.get("Evaluation_Type", "")
    plan = ""  #  
    env = "procthor"
    scene = str(task_config.get("scene_index", ""))
    
    #            
    clipboard_text = "\t".join([
        task_id,
        task_name,
        instruction,
        golden_action_str,
        step_number,
        evaluation,
        category,
        level,
        evaluation_type,
        plan,
        env,
        scene
    ])
    
    #       ：macOS   pbcopy，Windows   clip
    if platform.system() == "Darwin":
        copy_cmd = ["/usr/bin/pbcopy"]  #       ，   PATH   
    elif platform.system() == "Windows":
        copy_cmd = ["clip"]
    else:
    #     WSL (Windows Subsystem for Linux)     
        if "microsoft" in platform.uname().release.lower(): 
            copy_cmd = ["clip.exe"] 
        else:
            copy_cmd = None
    try:
        if copy_cmd is None:
            print("⚠️  Clipboard copy is not available on this platform")
            return
        process = subprocess.Popen(
            copy_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        process.communicate(input=clipboard_text)
        
        if process.returncode == 0:
            print(f"\n📋 Task info copied to clipboard:")
            print(f"   Task ID: {task_id}")
            print(f"   Task Name: {task_name}")
            print(f"   Step Number: {step_number}")
            print(f"   Category: {category}")
            print(f"   Level: {level}")
            print("   Ready to paste into Excel")
        else:
            print(f"⚠️  Clipboard copy failed (exit code: {process.returncode})")
    except FileNotFoundError:
        print("⚠️  Clipboard command not found")
    except Exception as e:
        raise e


def execute_action_sequence(
    env: ProcTHOREnvWrapper,
    action_sequence: List[Dict[str, Any]],
    task_config: Dict[str, Any],
    output_dir: str,
    step_by_step: bool = False,
) -> Dict[str, Any]:
    """              
    
    Args:
        env:     
        action_sequence:     
        task_config:     
        output_dir:     
        step_by_step:           
        
    Returns:
              
    """
    print(f"\n{'=' * 60}")
    print("Executing action sequence")
    print(f"{'=' * 60}")
    
    #      
    result = {
        "task_name": task_config.get("task_name", "custom_sequence"),
        "action_sequence": action_sequence,
        "executed_actions": [],
        "success": False,
        "evaluation_score": 0.0,
        "step_count": 0,
        "error_messages": [],
        "final_state": None,
    }
    
    #     
    observation = env.reset(task_config.get("instruction", "") or task_config.get("description", "Execute action sequence"))
    
    #       
    for i, action in enumerate(action_sequence, 1):
        if step_by_step:
            print(f"\n🛑 [Step-by-step] Next action: {action}")
            input(f"⌨️  Press Enter to execute step {i}/{len(action_sequence)}...")
        
        print(f"\nCurrent action {i}/{len(action_sequence)}: {action}")
        
        #         
        if action.get("action_type") == "task_completion":
            action_name = action.get("action_name", "").upper()
            print(f"🏁 Task completion action: {action_name}")
            
            action_result = {
                "step": i,
                "action": action,
                "success": True,
                "error_message": None,
                "reward": 0,
                "done": True,
                "is_task_completion": True,
            }
            result["executed_actions"].append(action_result)
            result["step_count"] = i
            break
        
        #     
        try:
            observation, error_message = env.step_with_action_dict(action)
            
            if observation and hasattr(observation, "image_path") and observation.image_path:
                print(f"🖼️  Image saved: {observation.image_path}")
            print(f"Step result: reward={observation.reward}, done={observation.done}")
            if error_message:
                print(f"Error message: {error_message}")
            
            #     PickupObject，        
            if action.get("action_name") == "PickupObject" and observation.metadata:
                inventory = observation.metadata.get("inventoryObjects", [])
                if inventory:
                    print(f"  ✓ Inventory: {[obj.get('objectType') for obj in inventory]}")
                else:
                    print("  ⚠️  Inventory: empty")
                    if observation.metadata.get("lastActionSuccess"):
                        print(
                            "     Note: lastActionSuccess=True, but no inventory "
                            "object was reported"
                        )
            
            #       
            last_action_success = (
                observation.metadata.get("lastActionSuccess", True)
                if observation.metadata
                else True
            )
            action_result = {
                "step": i,
                "action": action,
                "success": last_action_success,
                "error_message": error_message,
                "reward": observation.reward,
                "done": observation.done,
            }
            result["executed_actions"].append(action_result)
            result["step_count"] = i
            
            if error_message:
                result["error_messages"].append(f"Step {i}: {error_message}")
                print(f"⚠️  Action warning: {error_message}")
            
            #         
            if observation.done:
                print("✅ Task completed by environment")
                result["success"] = True
                break
        
        except Exception as e:
            print(f"❌ Execution error: {e}")
            result["error_messages"].append(f"Step {i}: {str(e)}")
            continue
    
    #       
    try:
        success, score = perform_final_evaluation(
            env=env, task_config=task_config, observation=observation
        )
        result["evaluation_score"] = score
        result["success"] = success
        print(f"Evaluation score: {score:.2f}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        result["error_messages"].append(f"Evaluation failed: {e}")
    
    #       
    final_last_action_success = (
        observation.metadata.get("lastActionSuccess", True)
        if observation.metadata
        else True
    )
    result["final_state"] = {
        "done": observation.done,
        "reward": observation.reward,
        "text_state": observation.text_state,
        "last_action_success": final_last_action_success,
    }
    
    #        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(output_dir, f"action_sequence_result_{timestamp}.json")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Result saved: {result_file}")
    
    return result


def main():
    """   """
    parser = argparse.ArgumentParser(
        description="Evaluate a ProcTHOR action sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task ID or task.json path (e.g. procthor00001 or tasks/procthor00001/task.json)",
    )
    
    parser.add_argument(
        "--scene-index", type=int, default=0, help="ProcTHOR scene index"
    )
    
    parser.add_argument(
        "--actions", type=str, default=None, help="Comma-separated action sequence"
    )
    
    parser.add_argument(
        "--action-file", type=str, default=None, help="File containing an action sequence"
    )
    
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run AI2-THOR in headless mode",
    )
    parser.add_argument(
        "--x-display",
        type=str,
        default=None,
        help="X display for xvfb or a physical X server, e.g. :99",
    )
    
    parser.add_argument(
        "--step-by-step",
        action="store_true",
        help="Pause before each action and wait for Enter",
    )
    
    args = parser.parse_args()
    
    #     
    if not args.task and not args.actions and not args.action_file:
        print("❌ Missing input. Provide one of:")
        print("   1. --task <task_id_or_task_json>")
        print("   2. --actions <comma_separated_actions>")
        print("   3. --action-file <action_sequence_file>")
        sys.exit(1)
    
    #       
    task_config = {}
    task_file = None
    if args.task:
        #          
        if os.path.isfile(args.task):
            task_file = args.task
        elif os.path.isfile(f"tasks/{args.task}/task.json"):
            task_file = f"tasks/{args.task}/task.json"
        
        if task_file:
            try:
                with open(task_file, "r", encoding="utf-8") as f:
                    task_config = json.load(f)
                print(f"✓ Loaded task file: {task_file}")
            except Exception as e:
                print(f"⚠️  Failed to load task file: {e}")
                print("   Continuing with custom action sequence")
        else:
            print(f"⚠️  Task not found: {args.task}")
            print("   Continuing with custom action sequence")
    
    #           init.json，    init_actions（  reset          ）
    init_actions = load_init_actions_for_task(task_file) if task_file else None
    
    #       
    action_sequence = None
    
    #    1：           golden_actions
    if args.task and task_config:
        action_sequence = extract_golden_actions_from_task(task_config)
    
    #    2：    --actions   
    if action_sequence is None and args.actions:
        print("✓ Loaded actions from --actions")
        action_sequence = parse_action_sequence(args.actions)
    
    #    3：  --action-file     
    if action_sequence is None and args.action_file:
        print(f"✓ Loading actions from file: {args.action_file}")
        try:
            with open(args.action_file, "r", encoding="utf-8") as f:
                action_str = f.read().strip()
            action_sequence = parse_action_sequence(action_str)
        except Exception as e:
            print(f"❌ Failed to read action file: {e}")
            sys.exit(1)
    
    if not action_sequence:
        print("❌ No valid actions found")
        sys.exit(1)
    
    print(f"Action sequence ({len(action_sequence)} actions):")
    for i, action in enumerate(action_sequence, 1):
        action_name = action["action_name"]
        action_type = action["action_type"]
        object_type = action.get("object_type")
        if object_type:
            print(f"  {i}. [{action_type}] {action_name}({object_type})")
        else:
            print(f"  {i}. [{action_type}] {action_name}")
    
    #       
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(Path(args.output_dir).expanduser().resolve() / f"action_eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    #     
    try:
        #            （   ）
        scene_index = task_config.get("scene_index", args.scene_index)
        
        env_config = {"task": task_config} if task_config else {}
        env_config.setdefault("env", {})
        if args.x_display:
            env_config["env"]["x_display"] = args.x_display
        if init_actions is not None:
            env_config["init_actions"] = init_actions
        env = ProcTHOREnvWrapper(
            scene_index=scene_index,
            output_dir=output_dir,
            headless=args.headless,
            config=env_config,
        )
    except Exception as e:
        print(f"❌ Failed to initialize ProcTHOR environment: {e}")
        sys.exit(1)
    
    try:
        #          
        result = execute_action_sequence(
            env,
            action_sequence,
            task_config,
            output_dir,
            step_by_step=args.step_by_step,
        )
    finally:
        env.close()
    
    #       
    print(f"\n{'=' * 60}")
    print("Final Summary")
    print(f"{'=' * 60}")
    print(f"Task name: {result['task_name']}")
    print(f"Steps executed: {result['step_count']}")
    print(f"Success: {'✅ Yes' if result['success'] else '❌ No'}")
    print(f"Evaluation score: {result['evaluation_score']:.2f}")
    
    if result["error_messages"]:
        print(f"Errors ({len(result['error_messages'])}):")
        for msg in result["error_messages"]:
            print(f"  - {msg}")
    
    print(f"{'=' * 60}")
    
    #           
    if args.task and task_config:
        try:
            copy_task_info_to_clipboard(task_config)
        except Exception as e:
            print(f"⚠️  Failed to copy task info: {e}")


if __name__ == "__main__":
    main()
