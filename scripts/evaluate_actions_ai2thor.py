"""
         
               ，      

        ：
1.      (--headless):    CloudRendering   ，   X11      
2.      (--simulate):         ，       

#     

##   1：    task.json      golden_actions（  ）
python scripts/ai2thor/evaluate_action_sequence.py --headless --task ai2thor001
#      tasks/ai2thor001/task.json     golden_actions

##   2：             （  --task      ，    task.json   golden_actions）
python scripts/ai2thor/evaluate_action_sequence.py --headless --task ai2thor001 --actions "MoveAhead,RotateLeft,DONE"

##   3：         
python evaluate_action_sequence.py --headless --task ai2thor001 --action-file actions.txt

##   4：            （yaml task_presets）
python evaluate_action_sequence.py --headless --task open_fridge --actions "MoveAhead,RotateLeft,DONE"

##   5：    （       ，       ）
python evaluate_action_sequence.py --simulate --task ai2thor001

      :
- --headless:    CloudRendering   ，   X11      ，    AI2-THOR     
- --simulate:         ，                

      :
-     : MoveAhead,RotateLeft,MoveBack,RotateRight,LookUp,LookDown
-      : PickupObject(Apple),OpenObject(Fridge),PutObject(CounterTop)
"""

import os
import sys
import json
import argparse
import platform
import subprocess
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional

#           
from pathlib import Path

AI2THOR_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = AI2THOR_ROOT.parent
for import_root in (str(REPO_ROOT), str(AI2THOR_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from mllm_base_agent.console import configure_utf8_stdio

configure_utf8_stdio()

from config import load_config

#    graph.py       
from core.agent.graph import (
    parse_action_string,
    perform_final_evaluation,
    execute_action,
)


def parse_action_sequence(action_str: str) -> List[Dict[str, Any]]:
    """
             

       graph.py    parse_action_string()           

        ：
    - MoveAhead,RotateLeft,MoveAhead
    - PickupObject(Apple),PutObject(CounterTop)
    - OpenObject(Fridge),CloseObject(Microwave)
    - DONE, FAIL (      )

    Args:
        action_str:        ，     

    Returns:
                ，   action_type, action_name, object_type    
    """
    actions = []
    for action in action_str.split(","):
        action = action.strip()
        if not action:
            continue

        try:
            #    graph.py       
            action_dict = parse_action_string(action)
            actions.append(action_dict)
        except ValueError as e:
            print(f"⚠️       '{action}'   : {e}")
            #          
            continue

    return actions


def extract_golden_actions_from_task(
    task_config: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """
             golden_actions

          ：
    1.        （    ）：
       "golden_actions": {
         "steps": 3,
         "actions": [
           {"step": 1, "action_type": "navigation", "action_name": "MoveAhead"},
           ...
         ]
       }

    2.        （  ，    ）：
       "golden_actions": {
         "steps": 3,
         "actions": ["MoveAhead(0.25)", "RotateLeft", "ToggleObjectOn(Laptop)"]
       }

    3.          （   ）：
       "golden_actions": "MoveAhead,RotateLeft,DONE"

    Args:
        task_config:       

    Returns:
              ，         None
    """
    golden_actions = task_config.get("golden_actions")

    if not golden_actions:
        return None

    #   1&2:     （   steps   actions）
    if isinstance(golden_actions, dict):
        actions = golden_actions.get("actions")
        if isinstance(actions, list) and len(actions) > 0:
            #             （  1）
            if all(isinstance(a, dict) and "action_name" in a for a in actions):
                print(
                    f"✓          golden_actions (         , {len(actions)}    )"
                )
                return actions

            #           （  2，  ）
            elif all(isinstance(a, str) for a in actions):
                print(
                    f"✓          golden_actions (       , {len(actions)}    )"
                )
                #              
                parsed_actions = []
                for action_str in actions:
                    try:
                        # "Done"           ，    
                        if action_str.strip().upper() == "DONE":
                            #    DONE   
                            parsed_actions.append(
                                {
                                    "action_type": "task_completion",
                                    "action_name": "DONE",
                                }
                            )
                            break  #         

                        action_dict = parse_action_string(action_str)
                        parsed_actions.append(action_dict)
                    except ValueError as e:
                        print(f"⚠️       '{action_str}'   : {e}")
                        continue
                return parsed_actions if parsed_actions else None

    #   3:          （   ）
    elif isinstance(golden_actions, str):
        print("✓          golden_actions (         )")
        return parse_action_sequence(golden_actions)

    return None


def copy_to_clipboard(text: str):
    """
               

       macOS, Windows, Linux    

    Args:
        text:       
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        try:
            subprocess.run("pbcopy", text=True, input=text, check=True)
            print("✔ Copied to clipboard (macOS)")
            return
        except Exception as e:
            print(f"⚠️  macOS pbcopy     : {e}")

    elif system == "Windows":
        try:
            subprocess.run("clip", text=True, input=text, check=True)
            print("✔ Copied to clipboard (Windows)")
            return
        except Exception as e:
            print(f"⚠️  Windows clip     : {e}")

    elif system == "Linux":
        #      xclip
        if shutil.which("xclip"):
            try:
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    text=True,
                    input=text,
                    check=True,
                )
                print("✔ Copied to clipboard (Linux + xclip)")
                return
            except Exception as e:
                print(f"⚠️  xclip     : {e}")

        #    xsel
        if shutil.which("xsel"):
            try:
                subprocess.run(
                    ["xsel", "--clipboard", "--input"],
                    text=True,
                    input=text,
                    check=True,
                )
                print("✔ Copied to clipboard (Linux + xsel)")
                return
            except Exception as e:
                print(f"⚠️  xsel     : {e}")

        print("⚠️  Linux clipboard copy needs xclip or xsel")

    else:
        print(f"⚠️  Unsupported platform for clipboard copy: {system}")


def convert_action_sequence_to_string_array(
    action_sequence: List[Dict[str, Any]],
) -> List[str]:
    """
         （    ）          

          task.json   golden_actions.actions   
           "Done"           

    Args:
        action_sequence:     （    ，       action_name, object_type  ）

    Returns:
             ，    ["MoveAhead(0.25)", "RotateLeft(90)", "ToggleObjectOn(Laptop)", "Done"]
    """
    action_strings = []

    for action in action_sequence:
        action_name = action.get("action_name", "")
        object_type = action.get("object_type")
        magnitude = action.get("magnitude")
        degrees = action.get("degrees")
        parameters = action.get("parameters", {})

        if action_name in ["DONE", "FAIL"]:
            #       /      （      ）
            continue

        #        ，    
        if object_type:
            #     ：ActionName(ObjectType)   ActionName(ObjectType, param)
            #     ：FillObjectWithLiquid           
            if action_name == "FillObjectWithLiquid" and "fillLiquid" in parameters:
                fill_liquid = parameters["fillLiquid"]
                action_str = f"{action_name}({object_type}, {fill_liquid})"
            else:
                action_str = f"{action_name}({object_type})"
        elif magnitude is not None:
            #     （  ）：ActionName(magnitude)
            action_str = f"{action_name}({magnitude})"
        elif degrees is not None:
            #     （  ）：ActionName(degrees)
            action_str = f"{action_name}({degrees})"
        else:
            #     （   ）：     action_name
            action_str = action_name

        action_strings.append(action_str)

    #      "Done"           
    action_strings.append("Done")

    return action_strings


def determine_task_level(action_sequence: List[Dict[str, Any]]) -> str:
    """
                  

    - Level1 Navigation:     navigation   
    - Level2 Interaction:     interaction   
    - Level3 Hybrid:      navigation   interaction   

    Args:
        action_sequence:     

    Returns:
               
    """
    has_navigation = False
    has_interaction = False

    for action in action_sequence:
        action_type = action.get("action_type", "")
        if action_type == "navigation":
            has_navigation = True
        elif action_type == "interaction":
            has_interaction = True

    if has_navigation and has_interaction:
        return "Level3 Hybrid"
    elif has_interaction:
        return "Level2 Interaction"
    elif has_navigation:
        return "Level1 Navigation"
    else:
        return "Unknown"


def generate_csv_row(
    task_config: Dict[str, Any], action_sequence: List[Dict[str, Any]], task_id: str
) -> Dict[str, str]:
    """
       CSV    

                  （ 16 ）：
    Task ID, Task Name, Instruction, Golden Action, Step Number, Evaluation,
    Category, Level, Evaluation Type, Plan, Env, Comment, Annotation, Anotator, Check, Checker

    Golden Action          ：["MoveAhead", "RotateLeft", "ToggleObjectOn(Laptop)"]

    Args:
        task_config:     
        action_sequence:     
        task_id:    ID（  ai2thor_000）

    Returns:
        CSV      （     ）
    """
    #           
    action_strings = convert_action_sequence_to_string_array(action_sequence)

    #    JSON          
    import json

    golden_actions_str = json.dumps(action_strings, ensure_ascii=False)

    #       
    level = determine_task_level(action_sequence)

    #     （   Done   ）
    steps = len(action_strings)

    #    CSV  （       ， 16 ）
    row = {
        "Task ID": task_id,
        "Task Name": task_config.get("task_name", ""),
        "Instruction": task_config.get("instruction", "")
        or task_config.get("description", ""),
        "Golden Action": golden_actions_str,
        "Step Number": str(steps),
        "Evaluation": "",
        "Category": "Daily Household (Kitchen/Bedroom)",
        "Level": level,
        "Evaluation Type": "Conditional",
        "Plan": "",
        "Env": "ai2thor",
        "Comment": "",
        "Annotation": "",
        "Anotator": "",
        "Check": "",
        "Checker": "",
    }

    return row


def save_csv_row_to_file(row: Dict[str, str], output_dir: str) -> str:
    """
    [   ]        ，TSV           
    """
    return ""


def simulate_action_sequence(
    action_sequence: List[Dict[str, Any]], task_config: Dict[str, Any], output_dir: str
) -> Dict[str, Any]:
    """
            （        ）

    Args:
        action_sequence:     
        task_config:     
        output_dir:     

    Returns:
              
    """
    print(f"\n{'=' * 60}")
    print("Simulating action sequence")
    print(f"{'=' * 60}")

    #      
    result = {
        "task_name": task_config.get("name", "custom_sequence"),
        "action_sequence": action_sequence,
        "executed_actions": [],
        "success": False,
        "evaluation_score": 0.0,
        "step_count": 0,
        "error_messages": [],
        "final_state": None,
    }

    #       
    simulated_state = {
        "objects": [
            {"objectType": "Fridge", "isOpen": False, "distance": 2.0},
            {"objectType": "Microwave", "isOpen": False, "distance": 3.0},
            {"objectType": "Egg", "isPickedUp": False, "distance": 1.5},
        ],
        "inventoryObjects": [],
        "agent": {"position": {"x": 0, "y": 0, "z": 0}},
        "lastActionSuccess": True,
    }

    #         
    for i, action in enumerate(action_sequence, 1):
        print(f"\nSimulated action {i}/{len(action_sequence)}: {action}")

        #         
        success = True  #       
        reward = 0.1
        error_message = None

        #          （   ）
        if action["action_name"] == "MoveAhead":
            simulated_state["agent"]["position"]["z"] += 0.25
            reward = 0.05
        elif action["action_name"] == "RotateLeft":
            reward = 0.02
        elif action["action_name"] == "PickupObject" and action["parameters"].get(
            "objectId"
        ):
            #       
            obj_type = action["parameters"].get("objectId", "Unknown")
            simulated_state["inventoryObjects"].append({"objectType": obj_type})
            reward = 0.5
        elif action["action_name"] == "OpenObject":
            #       
            for obj in simulated_state["objects"]:
                if obj["objectType"] == "Fridge":
                    obj["isOpen"] = True
                    reward = 0.8
                    break

        #       
        action_result = {
            "step": i,
            "action": action,
            "success": success,
            "error_message": error_message,
            "reward": reward,
            "done": False,
        }
        result["executed_actions"].append(action_result)
        result["step_count"] = i

        if error_message:
            result["error_messages"].append(f"Step {i}: {error_message}")
            print(f"⚠️  Simulation warning: {error_message}")

        print(f"✓ Simulation result: success={success}, reward={reward}")

    #       
    #         （     success_conditions     ）
    success_conditions = task_config.get("success_conditions", [])
    success_logic = task_config.get("success_logic", "AND").upper()

    if success_conditions:
        #    ：     
        condition_results = []
        for condition in success_conditions:
            condition_type = condition.get("type", "object_state")
            if condition_type == "object_state":
                field = condition.get("field") or condition.get("state", "isOpen")
                value = condition.get("value", True)
                object_type = condition.get("object_type")

                #            
                matched = False
                for obj in simulated_state["objects"]:
                    if object_type and obj.get("objectType") != object_type:
                        continue
                    if obj.get(field) == value:
                        matched = True
                        break
                condition_results.append(matched)
            elif condition_type == "object_in_receptacle":
                #     ：      
                condition_results.append(False)
            else:
                condition_results.append(False)

        #     
        if success_logic == "AND":
            result["success"] = all(condition_results)
        else:  # OR
            result["success"] = any(condition_results)
        result["evaluation_score"] = 1.0 if result["success"] else 0.0
    else:
        #    ：     
        success_condition = task_config.get("success_condition", {})
        if success_condition.get("type") == "object_state":
            field = success_condition.get("field", "isOpen")
            value = success_condition.get("value", True)

            #            
            for obj in simulated_state["objects"]:
                if obj.get(field) == value:
                    result["success"] = True
                    result["evaluation_score"] = 1.0
                    break

    result["final_state"] = simulated_state

    #        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        output_dir, f"action_sequence_simulation_{timestamp}.json"
    )

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n💾         : {result_file}")

    return result


def execute_action_sequence(
    env,  # AI2ThorEnvWrapper
    action_sequence: List[Dict[str, Any]],
    task_config: Dict[str, Any],
    output_dir: str,
    init_action_strings: List[str] = None,
    step_by_step: bool = False,
    reset_scene: Optional[str] = None,
) -> Dict[str, Any]:
    """
                  

       graph.py    execute_action()   perform_final_evaluation()    

    Args:
        env:     
        action_sequence:     
        task_config:     
        output_dir:     
        init_action_strings:           （  ），           
        step_by_step:           
        reset_scene:    ，    ``env.reset``        （  ``--task`` / init.json    scene   ）

    Returns:
              
    """
    print(f"\n{'=' * 60}")
    print("Executing action sequence")
    print(f"{'=' * 60}")

    #      
    result = {
        "task_name": task_config.get("name", "custom_sequence"),
        "action_sequence": action_sequence,
        "executed_actions": [],
        "success": False,
        "evaluation_score": 0.0,
        "step_count": 0,
        "error_messages": [],
        "final_state": None,
        "init_action_count": 0,
        "init_steps": [],
    }

    #     （      ）
    observation = env.reset(
        task_config.get("description", "Execute action sequence"),
        scene=reset_scene,
    )

    #        （   ）
    if init_action_strings:
        print(f"\n{'=' * 60}")
        print(f"📁 Executing init actions ({len(init_action_strings)} actions)")
        print(f"{'=' * 60}")

        for i, action_str in enumerate(init_action_strings, 1):
            action_str = action_str.strip()

            if not action_str or action_str.upper() == "DONE":
                break

            print(f"  {i}. Init action: {action_str}")

            init_row: Dict[str, Any] = {
                "index": i,
                "action": action_str,
                "phase": "init",
                "success": False,
                "error_message": None,
                "parse_error": None,
            }

            try:
                #     
                action_dict = parse_action_string(action_str)
            except Exception as e:
                init_row["parse_error"] = str(e)
                result["init_steps"].append(init_row)
                print(f"     ❌ Init action parse failed: {e}")
                continue

            try:
                #     
                observation, error_message = env.step_with_action_dict(action_dict)

                result["init_action_count"] += 1
                las = True
                if observation and observation.metadata:
                    las = bool(observation.metadata.get("lastActionSuccess", True))
                init_row["success"] = error_message is None and las
                init_row["error_message"] = error_message

                if error_message:
                    print(f"     ⚠️  {error_message}")
                else:
                    print("     ✓ Init action succeeded")

            except Exception as e:
                init_row["error_message"] = str(e)
                init_row["success"] = False
                print(f"     ❌ Init action execution failed: {e}")

            result["init_steps"].append(init_row)

        print(f"\n✓ Init actions complete ({result['init_action_count']} executed)\n")

    #       
    for i, action in enumerate(action_sequence, 1):
        if step_by_step:
            print(f"\n🛑 [Step-by-step] Next action: {action}")
            if i > 1:
                print(f"   (previous step {i - 1} already executed)")
            else:
                print("   (starting first step)")
            input(f"⌨️  Press Enter to execute step {i}/{len(action_sequence)}...")

        print(f"\nCurrent action {i}/{len(action_sequence)}: {action}")

        #    graph.py    execute_action()       
        #          DONE/FAIL        
        new_observation, error_message, is_task_completion = execute_action(env, action)

        if is_task_completion:
            #       （DONE/FAIL），         
            action_name = action.get("action_name", "").upper()
            print(f"🏁 Task completion action: {action_name}")

            #       
            action_result = {
                "step": i,
                "action": action,
                "phase": "golden",
                "success": True,
                "error_message": None,
                "reward": 0,
                "done": True,
                "is_task_completion": True,
            }
            result["executed_actions"].append(action_result)
            result["step_count"] = i

            #         
            break

        #    observation
        observation = new_observation
        if (
            observation
            and hasattr(observation, "image_path")
            and observation.image_path
        ):
            print(f"🖼️  Image saved: {observation.image_path}")
        print(f"Step result: reward={observation.reward}, done={observation.done}")
        if error_message:
            print(f"Error message: {error_message}")

        #       
        #   metadata     lastActionSuccess
        last_action_success = (
            observation.metadata.get("lastActionSuccess", True)
            if observation.metadata
            else True
        )
        action_result = {
            "step": i,
            "action": action,
            "phase": "golden",
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

    #    graph.py    perform_final_evaluation()         
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
    #   metadata     lastActionSuccess
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

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n💾       : {result_file}")

    return result


def main():
    """   """
    parser = argparse.ArgumentParser(
        description="Evaluate an AI2-THOR action sequence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Basic navigation (headless, no X11 required):
   python evaluate_action_sequence.py --headless --actions "MoveAhead,RotateLeft,MoveAhead"

2. Object interaction:
   python evaluate_action_sequence.py --headless --actions "PickupObject(Apple),PutObject(CounterTop)"

3. Task preset:
   python evaluate_action_sequence.py --headless --task open_fridge --scene FloorPlan1 --actions "MoveAhead,RotateLeft"

4. Action file:
   python evaluate_action_sequence.py --headless --action-file actions.txt

5. Simulation only (parse and evaluate without launching AI2-THOR):
   python evaluate_action_sequence.py --simulate --actions "MoveAhead,RotateLeft"

Important options:
- --headless: use CloudRendering without X11
- --simulate: parse and score actions without launching AI2-THOR

Action examples:
- Navigation: MoveAhead, RotateLeft, MoveBack, RotateRight, LookUp, LookDown
- Interaction: PickupObject(Apple), OpenObject(Fridge), PutObject(CounterTop)
        """,
    )

    parser.add_argument(
        "--config", type=str, default="experiments/configs/ai2thor/config_close_gpt-5.yaml", help="Config path"
    )

    parser.add_argument(
        "--task", type=str, default=None, help="Task ID under tasks/ or task preset name"
    )

    parser.add_argument(
        "--scene", type=str, default=None, help="Scene name, e.g. FloorPlan1"
    )

    parser.add_argument(
        "--actions", type=str, default=None, help="Comma-separated action sequence"
    )

    parser.add_argument(
        "--override-actions",
        action="store_true",
        help="When --task is used, override task.json golden_actions with --actions (append DONE manually if needed)",
    )

    parser.add_argument(
        "--action-file", type=str, default=None, help="File containing an action sequence"
    )

    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Simulation mode: parse and score actions without launching AI2-THOR",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Headless mode: use CloudRendering without X11",
    )

    parser.add_argument(
        "--step-by-step",
        action="store_true",
        help="Pause before each action and wait for Enter",
    )

    args = parser.parse_args()

    #     （     --task  ，     --actions   --action-file）
    if not args.task and not args.actions and not args.action_file:
        print("❌ Missing input. Provide one of:")
        print(
            "   1. --task <task_id_or_preset> (task.json or task_presets golden_actions)"
        )
        print("   2. --actions <comma_separated_actions>")
        print("   3. --action-file <action_sequence_file>")
        sys.exit(1)

    #     
    config = load_config(args.config)

    #       
    if args.task:
        try:
            task_config = config.apply_task_by_name(args.task)
        except ValueError as e:
            print(f"❌ {e}")
            sys.exit(1)
    else:
        #         
        task_config = {
            "name": "custom_action_sequence",
            "description": "Execute custom action sequence",
            "max_steps": 50,
        }

    #     
    scene = args.scene or task_config.get("scene", "FloorPlan1")

    #       （   ：golden_actions > --actions > --action-file）
    action_sequence = None

    #    1：           golden_actions
    if args.task and not (args.override_actions and args.actions):
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
        print("   Possible fixes:")
        print("   1. Use --task with task.json golden_actions")
        print("   2. Use --actions with a comma-separated sequence")
        print("   3. Use --action-file with a sequence file")
        sys.exit(1)

    #             init.json
    init_action_strings = []
    if args.task:
        task_folder = os.path.join("tasks", args.task)
        if os.path.isdir(task_folder):
            init_file = os.path.join(task_folder, "init.json")
            if os.path.exists(init_file):
                try:
                    with open(init_file, "r", encoding="utf-8") as f:
                        init_data = json.load(f)

                    # init.json       ：
                    # 1.    : ["MoveAhead", "RotateLeft", ...]
                    # 2.   : {"scene": "FloorPlan1", "actions": [...]}
                    if isinstance(init_data, list):
                        init_action_strings = init_data
                        init_scene = None
                    elif isinstance(init_data, dict):
                        init_action_strings = init_data.get("actions", [])
                        init_scene = init_data.get("scene")
                    else:
                        init_action_strings = []
                        init_scene = None

                    #    init     scene，   
                    if init_scene:
                        scene = init_scene
                        print(f"✓   init.json     : {scene}")

                    #    "Done"   （   ）
                    if (
                        init_action_strings
                        and init_action_strings[-1].strip().upper() == "DONE"
                    ):
                        init_action_strings = init_action_strings[:-1]

                    if init_action_strings:
                        print(
                            f"✓   {init_file}    {len(init_action_strings)}       "
                        )

                except Exception as e:
                    print(f"⚠️     init.json   : {e}")

    print(f"        ({len(action_sequence)}    ):")
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
    output_dir = os.path.join(args.output_dir, f"action_eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    #           
    if args.simulate:
        #     （        ）
        print("📋 Simulation mode (no environment launch)")
        result = simulate_action_sequence(action_sequence, task_config, output_dir)
    else:
        #       
        try:
            from envs.ai2thor import AI2ThorEnvWrapper

            #     
            env_config = config.get_all()

            #         ，   CloudRendering   
            if args.headless:
                print("🖥️         (CloudRendering)")
                if "env" not in env_config:
                    env_config["env"] = {}
                env_config["env"]["platform"] = "CloudRendering"

            env = AI2ThorEnvWrapper(
                scene=scene,
                grid_size=config.get("env.grid_size", 0.25),
                render_depth_image=False,
                render_instance_segmentation=False,
                width=config.get("env.width", 800),
                height=config.get("env.height", 600),
                output_dir=output_dir,
                config=env_config,
            )
        except Exception as e:
            print(f"❌ Failed to initialize AI2-THOR environment: {e}")
            if not args.headless:
                print(
                    "💡 Hint: try --headless to use CloudRendering without X11"
                )
            print("💡 Hint: use --simulate to test parsing without launching AI2-THOR")
            sys.exit(1)

        try:
            #          
            result = execute_action_sequence(
                env,
                action_sequence,
                task_config,
                output_dir,
                init_action_strings,
                step_by_step=args.step_by_step,
                reset_scene=scene,
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

    #       TSV   （     ）
    if args.task and action_sequence:
        print(f"\n{'=' * 80}")
        print("TSV row (paste into Excel/Sheet)")
        print(f"{'=' * 80}")

        try:
            #      ID（      task_config）
            task_id = args.task
            if task_id.startswith("ai2thor"):
                #   ai2thor001     ai2thor_001   
                if "_" not in task_id:
                    task_id = "ai2thor_" + task_id[7:].zfill(3)

            #    CSV    
            csv_row = generate_csv_row(task_config, action_sequence, task_id)

            #       （          ）
            field_order = [
                "Task ID",
                "Task Name",
                "Instruction",
                "Golden Action",
                "Step Number",
                "Evaluation",
                "Category",
                "Level",
                "Evaluation Type",
                "Plan",
                "Env",
                "Comment",
                "Annotation",
                "Anotator",
                "Check",
                "Checker",
            ]

            #    TSV  （     ）
            tsv_values = []
            for field in field_order:
                value = csv_row.get(field, "")
                tsv_values.append(str(value) if value else "")

            tsv_line = "\t".join(tsv_values)

            #    TSV  （      ，    ）
            print("\n")
            print(f'"{tsv_line}"')
            print("\n")

            #         （          TSV   ）
            copy_to_clipboard(tsv_line)

            print(f"{'=' * 80}")

        except Exception as e:
            print(f"❌    TSV     : {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
