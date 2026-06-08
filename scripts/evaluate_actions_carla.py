"""
CARLA Action Sequence Evaluator
         （        golden_actions     ）

    ：
1.           golden_actions    ：
   python scripts/carla/evaluate_action_sequence.py --task carla00001

2.         ：
   python scripts/carla/evaluate_action_sequence.py --task carla00001 --actions "WalkForward,TurnRight,Done"

3.          ：
   python scripts/carla/evaluate_action_sequence.py --task carla00001 --action-file actions.txt
"""

import os
import sys
import json
import argparse
import platform
import subprocess
import shutil
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

CARLA_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CARLA_ROOT.parent
for import_root in (str(REPO_ROOT), str(CARLA_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from mllm_base_agent.console import configure_utf8_stdio

configure_utf8_stdio()


def copy_to_clipboard(text: str):
    """           （macOS / Windows / Linux） """
    system = platform.system()

    if system == "Darwin":  # macOS
        try:
            subprocess.run("pbcopy", text=True, input=text, check=True)
            print("✔ Copied to clipboard (macOS)")
            return
        except Exception as e:
            print(f"⚠️  macOS pbcopy failed: {e}")

    elif system == "Windows":
        try:
            subprocess.run("clip", text=True, input=text, check=True)
            print("✔ Copied to clipboard (Windows)")
            return
        except Exception as e:
            print(f"⚠️  Windows clip failed: {e}")

    elif system == "Linux":
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
                print(f"⚠️  xclip failed: {e}")

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
                print(f"⚠️  xsel failed: {e}")

        print("⚠️  Linux clipboard copy needs xclip or xsel")

    else:
        print(f"⚠️  Unsupported platform for clipboard copy: {system}")


def normalize_action_strings(action_strs: List[str]) -> List[str]:
    """          ，        Done """
    normalized = [str(a).strip() for a in (action_strs or []) if str(a).strip()]
    if not normalized:
        return normalized

    if normalized[-1].upper() != "DONE":
        normalized.append("Done")
    return normalized


def determine_task_level(action_strs: List[str]) -> str:
    """              （   ai2thor   Level     ） """
    nav_prefixes = {
        "WALK",
        "MOVE",
        "TURN",
        "ROTATE",
        "LANE",
        "BRAKE",
        "THROTTLE",
        "STEER",
    }
    interaction_prefixes = {
        "OPEN",
        "CLOSE",
        "PICKUP",
        "PUT",
        "TOGGLE",
        "INTERACT",
    }

    has_navigation = False
    has_interaction = False
    for action in action_strs:
        name = action.strip().upper()
        if "(" in name:
            name = name.split("(", 1)[0]

        if any(name.startswith(p) for p in nav_prefixes):
            has_navigation = True
        if any(name.startswith(p) for p in interaction_prefixes):
            has_interaction = True

    if has_navigation and has_interaction:
        return "Level3 Hybrid"
    if has_interaction:
        return "Level2 Interaction"
    if has_navigation:
        return "Level1 Navigation"
    return "Unknown"


def generate_csv_row(
    task_config: Dict[str, Any], action_strs: List[str], task_id: str
) -> Dict[str, str]:
    """    ai2thor evaluate     16   TSV   """
    normalized_actions = normalize_action_strings(action_strs)
    golden_actions_str = json.dumps(normalized_actions, ensure_ascii=False)

    row = {
        "Task ID": task_id,
        "Task Name": task_config.get("task_name", "") or task_config.get("name", ""),
        "Instruction": task_config.get("instruction", "")
        or task_config.get("description", ""),
        "Golden Action": golden_actions_str,
        "Step Number": str(len(normalized_actions)),
        "Evaluation": "",
        "Category": "Autonomous Driving",
        "Level": determine_task_level(normalized_actions),
        "Evaluation Type": "Conditional",
        "Plan": "",
        "Env": "carla",
        "Comment": "",
        "Annotation": "",
        "Anotator": "",
        "Check": "",
        "Checker": "",
    }
    return row


def load_init_from_folder(task_folder_path: str) -> tuple:
    """          init.json    

    Returns:
        (init_data, scene_name)
        - init_data: dict，     actions   initial_location/initial_rotation
        - scene_name: init.json      scene（  ）
    """

    init_file = os.path.join(task_folder_path, "init.json")
    if not os.path.exists(init_file):
        return None, None

    try:
        with open(init_file, "r", encoding="utf-8") as f:
            init_data = json.load(f)

        # init.json       ：
        # 1)    : ["WalkForward", ...]  ->    actions
        # 2)   (  ): {"scene": "Town10HD", "actions": [...]}
        # 3)   (  ): {"scene": "Town10HD", "initial_location": [...], "initial_rotation": [...]}
        if isinstance(init_data, list):
            return {"actions": init_data}, None
        if isinstance(init_data, dict):
            scene_name = init_data.get("scene")
            if "initial_location" in init_data or "actions" in init_data:
                return init_data, scene_name

        print("⚠️ init.json format is invalid; skipping init data")
        return None, None
    except Exception as e:
        print(f"⚠️ Failed to load init.json: {e}")
        return None, None


def _apply_init_coordinates(env, init_data: dict) -> bool:
    try:
        import carla
    except Exception:
        print("⚠️ carla package is unavailable; cannot apply init coordinates")
        return False

    location_list = init_data.get("initial_location")
    rotation_list = init_data.get("initial_rotation")
    if not location_list:
        return False

    location = carla.Location(
        x=location_list[0], y=location_list[1], z=location_list[2]
    )
    if rotation_list and len(rotation_list) == 3:
        rotation = carla.Rotation(
            pitch=rotation_list[0], yaw=rotation_list[1], roll=rotation_list[2]
        )
    else:
        rotation = carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)

    transform = carla.Transform(location, rotation)

    actor = None
    if hasattr(env, "walker") and getattr(env, "walker") is not None:
        actor = env.walker
    elif hasattr(env, "vehicle") and getattr(env, "vehicle") is not None:
        actor = env.vehicle
    elif hasattr(env, "actor") and getattr(env, "actor") is not None:
        actor = env.actor

    if actor is None:
        print("⚠️        transform   actor")
        return False

    actor.set_transform(transform)

    #       transform   
    if hasattr(env, "world") and env.world is not None:
        for _ in range(10):
            env.world.tick()

    #           
    actual_loc = actor.get_transform().location
    print(f"✓     init.json     : {location_list}")
    print(
        f"✓       : [{actual_loc.x:.2f}, {actual_loc.y:.2f}, {actual_loc.z:.2f}]"
    )
    return True


def _execute_init_actions(env, init_actions: list, executor_type) -> int:
    if not init_actions:
        return 0

    from envs.carla.actions import CarlaUnifiedAction

    count = 0
    for action_str in init_actions:
        action_str = str(action_str).strip()
        if not action_str or action_str.upper() == "DONE":
            break

        try:
            action_obj = CarlaUnifiedAction.from_string(action_str, executor_type)
        except Exception as e:
            print(f"⚠️ init action     : {action_str} ({e})")
            continue

        action_dict = {
            "action_name": action_obj.action_type.value,
            "parameters": action_obj.parameters,
        }
        _, error = env.step_with_action_dict(action_dict)
        count += 1
        if error:
            print(f"⚠️ init action     : {error}")

    if count:
        print(f"✓ Init actions complete: {count} actions")
    return count


def create_env(config_loader, task_config: Dict[str, Any], output_dir: str):
    """   CARLA   （   python -m scripts.carla.work.run_task      ）"""
    from envs.carla import VehicleExecutor, WalkerExecutor

    full_config = config_loader.get_all()
    if "task" not in full_config:
        full_config["task"] = {}
    full_config["task"].update(task_config)

    if "env" not in full_config:
        full_config["env"] = {}

    task_scene = (
        task_config.get("map")
        or task_config.get("scene")
        or full_config["env"].get("map", "Town01")
    )
    task_weather = task_config.get("weather")

    #      env.map/env.weather，   executor           /  
    full_config["env"]["map"] = task_scene
    if task_weather:
        full_config["env"]["weather"] = task_weather

    executor_name = task_config.get("executor", "vehicle")
    host = full_config["env"].get("host", "localhost")
    port = full_config["env"].get("port", 2000)
    timeout = full_config["env"].get("timeout", 10.0)

    if executor_name == "walker":
        return WalkerExecutor(
            host=host,
            port=port,
            timeout=timeout,
            config=full_config,
            output_dir=output_dir,
        )
    return VehicleExecutor(
        host=host, port=port, timeout=timeout, config=full_config, output_dir=output_dir
    )


def execute_sequence(
    env, actions: List[Any], task_config: Dict[str, Any]
) -> Dict[str, Any]:
    """      """
    print(f"\n{'=' * 60}")
    print(f"         ({len(actions)}  )")
    print(f"{'=' * 60}")

    results = {
        "success": False,
        "steps_executed": 0,
        "final_distance": float("inf"),
        "collision": False,
        "history": [],
    }

    task_desc = task_config.get("instruction", "Execute action sequence")
    scene = task_config.get("map") or task_config.get("scene") or "Town01"

    #    init.json（   ）：    scene       
    task_id = task_config.get("task_id")
    from envs.carla.actions import ExecutorType

    executor_type = ExecutorType(task_config.get("executor", "vehicle"))
    init_data = None
    if task_id:
        task_folder = os.path.join("tasks", task_id)
        init_data, init_scene = load_init_from_folder(task_folder)
        if init_scene:
            scene = init_scene

    #        
    env.reset(task_desc, scene=scene)

    #      
    if isinstance(init_data, dict):
        if "actions" in init_data and isinstance(init_data.get("actions"), list):
            _execute_init_actions(env, init_data.get("actions"), executor_type)
        elif "initial_location" in init_data:
            _apply_init_coordinates(env, init_data)

    #     
    for i, action in enumerate(actions, 1):
        print(f"\nStep {i}/{len(actions)}: {action.action_type.value}")

        #       
        action_dict = {
            "action_name": action.action_type.value,
            "parameters": action.parameters,
        }

        #   
        obs, error = env.step_with_action_dict(action_dict)

        #     
        step_info = {
            "step": i,
            "action": action.action_type.value,
            "reward": obs.reward,
            "done": obs.done,
            "collision": False,  # TODO:   obs       
            "error": error,
        }
        results["history"].append(step_info)
        results["steps_executed"] = i

        if error:
            print(f"⚠️ Error: {error}")
            lower_error = str(error).lower()
            if any(
                token in lower_error
                for token in [
                    "not initialized",
                    "world missing",
                    "spawn failed",
                    "not available",
                ]
            ):
                results["fatal_error"] = str(error)
                print("❌ Fatal environment error; stopping sequence")
                break

        if obs.done:
            print("🎉 Environment returned done=True")
            results["success"] = True
            break

    #       （      ）
    if hasattr(env, "get_distance_to_target"):
        try:
            final_dist = float(env.get_distance_to_target())
            results["final_distance"] = final_dist
            if final_dist != float("inf"):
                print(f"📏 Final distance to target: {final_dist:.2f}m")
        except Exception as e:
            print(f"⚠️  Failed to compute final distance: {e}")

    #            done，     _check_success       
    if not results["success"] and hasattr(env, "_check_success"):
        try:
            results["success"] = bool(env._check_success())
        except Exception:
            pass

    return results


def main():
    #                import     
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from config import load_config
    from envs.carla.actions import ExecutorType, parse_action_sequence

    parser = argparse.ArgumentParser(description="Evaluate a CARLA action sequence")
    parser.add_argument("--task", required=True, help="Task ID, e.g. carla00001")
    parser.add_argument(
        "--config",
        default="experiments/configs/carla/config_close_gpt-5.yaml",
        help="CARLA config path (default: experiments/configs/carla/config_close_gpt-5.yaml)",
    )
    parser.add_argument("--actions", help="Comma-separated action sequence")
    parser.add_argument("--action-file", help="File containing an action sequence")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    args = parser.parse_args()

    # 1.       
    config_loader = load_config(args.config)
    try:
        task_config = config_loader.apply_task_by_name(args.task)
        if not task_config:
            print(f"❌ Task not found: {args.task}")
            return
    except Exception as e:
        print(f"❌ Failed to load task: {e}")
        return

    print(f"Task name: {task_config.get('task_name')}")
    print(f"Scene: {task_config.get('scene')}")
    print(f"Executor: {task_config.get('executor')}")

    # 2.       
    executor_type = ExecutorType(task_config.get("executor", "vehicle"))
    action_objects = []
    action_strs = []

    if args.actions:
        #      
        action_strs = [s.strip() for s in args.actions.split(",") if s.strip()]
        action_objects = parse_action_sequence(action_strs, executor_type)
    elif args.action_file:
        #     
        with open(args.action_file, "r") as f:
            content = f.read().strip()
        action_strs = [s.strip() for s in content.split(",") if s.strip()]
        action_objects = parse_action_sequence(action_strs, executor_type)
    else:
        #   task.json   golden_actions   
        golden = task_config.get("golden_actions", {})
        if isinstance(golden, dict):
            action_strs = golden.get("actions", [])
        else:
            action_strs = []

        if not action_strs:
            print("❌ No actions found. Provide --actions/--action-file or task.json golden_actions.")
            return
        print(f"✓ Loaded golden_actions from task.json ({len(action_strs)} actions)")
        action_objects = parse_action_sequence(action_strs, executor_type)

    if not action_objects:
        print("❌ No valid actions found")
        return

    # 3.        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"eval_{args.task}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    env = None
    try:
        env = create_env(config_loader, task_config, run_dir)
        results = execute_sequence(env, action_objects, task_config)

        # 4.     
        print(f"\n{'=' * 60}")
        print("Final Summary")
        print(f"{'=' * 60}")
        print(f"Success: {'✅ Yes' if results['success'] else '❌ No'}")
        print(f"Steps executed: {results['steps_executed']}")

        #     
        result_file = os.path.join(run_dir, "result.json")
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Result saved: {result_file}")

        # 5.       TSV   （  ai2thor evaluate     ）
        if args.task and action_strs:
            print(f"\n{'=' * 80}")
            print("TSV row (paste into Excel/Sheet)")
            print(f"{'=' * 80}")

            csv_row = generate_csv_row(task_config, action_strs, args.task)
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

            tsv_values = []
            for field in field_order:
                value = csv_row.get(field, "")
                tsv_values.append(str(value) if value else "")

            tsv_line = "\t".join(tsv_values)

            print("\n")
            print(f'"{tsv_line}"')
            print("\n")

            #         （   TSV）
            copy_to_clipboard(tsv_line)
            print(f"{'=' * 80}")

    except Exception as e:
        print(f"\n❌     : {e}")
        import traceback

        traceback.print_exc()
    finally:
        if env:
            env.close()


if __name__ == "__main__":
    main()
