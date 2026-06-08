"""
Main program - Unified entry point supporting multiple environments
Automatically identifies environment type (AI2-THOR or CARLA) from configuration file
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from actions.max_steps import resolve_max_steps_from_task

from config import load_config, print_config
from core.llm.provider import get_vlm
from core.agent.graph import create_agent_graph


EXTERNAL_FAILURE_TYPES = {"api_error", "env_error", "external_error"}


def _default_tasks_root() -> str:
    """       ；       SPATIAL_TASKS_ROOT   （     taskabs/） """
    return os.environ.get("SPATIAL_TASKS_ROOT", "data/virtualhome/tasks")


def _derive_task_result(state: Dict[str, Any]) -> Optional[str]:
    """Derive success/failure/null task_result for logs."""
    success = state.get("success")
    failure_type = state.get("failure_type")
    if success is True:
        return "success"
    if failure_type in EXTERNAL_FAILURE_TYPES:
        return None
    if success is False:
        return "failure"
    return None


def load_init_actions_from_folder(task_folder_path: str, env_type: str = "ai2thor") -> tuple:
    """          init.json   （            ）

    Returns:
        (init_data, scene_name)   ：
        - init_data:   ，        
        - scene_name: init.json       ，       None
    """
    import json

    init_file = os.path.join(task_folder_path, "init.json")

    if not os.path.exists(init_file):
        return None, None

    env_type = str(env_type or "ai2thor").lower()

    if env_type == "virtualhome":
        try:
            from scripts.evaluate_actions_virtualhome import (
                load_discrete_init_data,
            )

            loaded = load_discrete_init_data(init_file, yaw_step_degrees=30.0)
            if loaded.get("error"):
                print(f"⚠️     init.json   : {loaded['error']}")
                return None, None

            init_data = {
                "actions": loaded.get("init_action_strings", []),
                "character_position": loaded.get("init_char_position"),
                "character_rotation": loaded.get("init_char_rotation"),
                "character_yaw_degrees": loaded.get("init_char_yaw_degrees"),
                "camera_pitch": loaded.get("init_camera_pitch"),
                "recorded_user_actions": loaded.get("recorded_user_actions"),
            }
            has_pose = bool(
                init_data.get("character_position")
                and (
                    init_data.get("character_rotation") is not None
                    or init_data.get("character_yaw_degrees") is not None
                )
            )
            has_actions = bool(init_data.get("actions"))

            if not has_pose and not has_actions:
                return None, loaded.get("init_scene")

            return init_data, loaded.get("init_scene")
        except Exception as e:
            print(f"⚠️     init.json   : {e}")
            return None, None

    try:
        with open(init_file, "r", encoding="utf-8") as f:
            init_data = json.load(f)

        #       ：
        # 1.    : ["MoveAhead", "RotateLeft", ...] （     ）
        # 2.   (  ): {"scene": "FloorPlan1", "actions": [...]} （   ）
        # 3.   (  ): {"scene": "Town10HD", "initial_location": [...], "initial_rotation": [...]} （   ）

        if isinstance(init_data, list):
            #    ：     
            return {"actions": init_data}, None
        if isinstance(init_data, dict):
            scene_name = init_data.get("scene")

            if "initial_location" in init_data:
                return init_data, scene_name
            if "actions" in init_data:
                return init_data, scene_name

            print("⚠️  init.json     ，     ")
            return None, None

        print("⚠️  init.json     ，     ")
        return None, None

    except Exception as e:
        print(f"⚠️     init.json   : {e}")
        return None, None


def apply_init_coordinates(env, init_data: dict) -> int:
    """       （         ）"""
    try:
        import carla
    except ImportError:
        print("⚠️  carla    ，         ")
        return 0

    location_list = init_data.get("initial_location")
    rotation_list = init_data.get("initial_rotation")

    if not location_list or not rotation_list:
        print("⚠️  init.json           ")
        return 0

    print(f"\n{'=' * 60}")
    print("📍        ")
    print(f"{'=' * 60}")
    print(
        f"    : ({location_list[0]:.2f}, {location_list[1]:.2f}, {location_list[2]:.2f})"
    )
    print(
        f"    : (pitch={rotation_list[0]:.1f}°, yaw={rotation_list[1]:.1f}°, roll={rotation_list[2]:.1f}°)"
    )

    try:
        location = carla.Location(
            x=location_list[0], y=location_list[1], z=location_list[2]
        )
        rotation = carla.Rotation(
            pitch=rotation_list[0], yaw=rotation_list[1], roll=rotation_list[2]
        )
        transform = carla.Transform(location, rotation)

        if hasattr(env, "walker") and getattr(env, "walker"):
            env.walker.set_transform(transform)
            print("  ✓        ")
        elif hasattr(env, "vehicle") and getattr(env, "vehicle"):
            env.vehicle.set_transform(transform)
            env.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            print("  ✓        ")
        else:
            print("  ⚠️         actor")
            return 0

        if hasattr(env, "world"):
            for _ in range(5):
                env.world.tick()

        print(f"{'=' * 60}\n")
        return 1

    except Exception as e:
        print(f"  ❌       : {e}")
        import traceback

        traceback.print_exc()
        return 0


def execute_init_actions(env, init_action_strings: list, config: dict) -> tuple:
    """         

    Args:
        env:     
        init_action_strings:           
        config:     

    Returns:
        (init_action_count, last_observation)   ：
        - init_action_count:          
        - last_observation:        
    """
    from core.agent.graph import parse_action_string

    if not init_action_strings:
        return 0, None

    env_type = str(config.get("env", {}).get("type", "ai2thor")).lower()
    if env_type == "virtualhome":
        from envs.virtualhome import parse_vh_action_string

        parse_init_action = parse_vh_action_string
    else:
        parse_init_action = parse_action_string

    print(f"\n{'=' * 60}")
    print(f"📁           ({len(init_action_strings)}  )")
    print(f"{'=' * 60}")

    init_count = 0
    last_observation = None

    for i, action_str in enumerate(init_action_strings, 1):
        action_str = action_str.strip()

        if not action_str or action_str.upper() == "DONE":
            break

        print(f"  {i}.        : {action_str}")

        try:
            #     
            action_dict = parse_init_action(action_str)

            #     
            observation, error_message = env.step_with_action_dict(action_dict)
            last_observation = observation

            init_count += 1

            if error_message:
                print(f"     ⚠️  {error_message}")
            else:
                print("     ✓   ")

        except Exception as e:
            print(f"     ❌   /    : {e}")
            continue

    print(f"\n✓       (  {init_count}  )\n")

    return init_count, last_observation




def load_task_from_folder(task_folder_path: str) -> Dict[str, Any]:
    """ tasks        

    Args:
        task_folder_path:           ID

    Returns:
              

    Raises:
        FileNotFoundError:       task.json   
        ValueError: task.json    
    """
    import json
    from pathlib import Path

    #      ID（ ai2thor00000），      
    if not os.path.exists(task_folder_path):
        #     tasks（  SPATIAL_TASKS_ROOT）     
        tasks_base = Path(_default_tasks_root())
        task_id = os.path.basename(task_folder_path)
        task_folder_path = tasks_base / task_id

    task_folder = Path(task_folder_path)

    if not task_folder.exists():
        raise FileNotFoundError(f"        : {task_folder}")

    task_json_path = task_folder / "task.json"

    if not task_json_path.exists():
        raise FileNotFoundError(f"task.json     : {task_json_path}")

    try:
        with open(task_json_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)

        success_condition = task_data.get("success_condition")
        success_conditions = task_data.get("success_conditions")
        if success_condition is None:
            if isinstance(success_conditions, list) and success_conditions:
                if isinstance(success_conditions[0], dict):
                    success_condition = success_conditions[0]
            elif isinstance(success_conditions, dict):
                success_condition = success_conditions
        if success_conditions is None and isinstance(success_condition, dict):
            success_conditions = [success_condition]

        #    config     
        task_config = {
            "name": task_data.get("task_id", task_folder.name),
            "scene": task_data.get("scene", "FloorPlan1"),
            "instruction": task_data.get("instruction")
            or task_data.get("task_name", ""),
            "description": task_data.get(
                "target_description", task_data.get("instruction", "")
            ),
            "target_object_types": task_data.get("target_object_types", []),
            "success_condition": success_condition,
            "success_conditions": success_conditions,
            "success_logic": task_data.get("success_logic", "AND"),
            "max_steps": resolve_max_steps_from_task(task_data, 50),
            "task_folder_path": str(task_folder.absolute()),
            "task_id": task_data.get("task_id", task_folder.name),
            "_from_tasks_folder": True,  #        tasks   
        }

        return task_config

    except json.JSONDecodeError as e:
        raise ValueError(f"task.json    : {e}")
    except Exception as e:
        raise ValueError(f"        : {e}")


def create_env(
    env_type: str,
    config: Dict[str, Any],
    output_dir: str,
    scene: Optional[str] = None,
    executor_type: Optional[str] = None,
):
    """Dynamically create environment instance

    Args:
        env_type: Environment type ('ai2thor', 'carla', or 'virtualhome')
        config: Configuration dictionary
        output_dir: Output directory
        scene: Task-specific scene name (overrides config default)

    Returns:
        Environment instance

    Raises:
        ValueError: Unsupported environment type
    """
    if env_type == "ai2thor":
        from envs.ai2thor import AI2ThorEnvWrapper

        env_config = config.get("env", {})
        # Use task-specific scene if provided, otherwise fallback to config default
        actual_scene = scene or env_config.get("scene", "FloorPlan1")
        return AI2ThorEnvWrapper(
            scene=actual_scene,
            grid_size=env_config.get("grid_size", 0.25),
            render_depth_image=env_config.get("render_depth", False),
            render_instance_segmentation=env_config.get(
                "render_instance_segmentation", False
            ),
            width=env_config.get("width", 800),
            height=env_config.get("height", 600),
            output_dir=output_dir,
            config=config,
        )

    elif env_type == "carla":
        from envs.carla import VehicleExecutor, WalkerExecutor

        env_config = config.get("env", {})
        executor_type = executor_type or "vehicle"

        if executor_type == "walker":
            return WalkerExecutor(
                host=env_config.get("host", "localhost"),
                port=env_config.get("port", 2000),
                timeout=env_config.get("timeout", 10.0),
                config=config,
                output_dir=output_dir,
            )
        return VehicleExecutor(
            host=env_config.get("host", "localhost"),
            port=env_config.get("port", 2000),
            timeout=env_config.get("timeout", 10.0),
            config=config,
            output_dir=output_dir,
        )

    elif env_type == "virtualhome":
        from envs.virtualhome import VirtualHomeEnvWrapper

        env_config = config.get("env", {})
        actual_scene = scene if scene is not None else env_config.get("scene", 0)
        try:
            actual_scene = int(actual_scene)
        except Exception:
            # Fallback to env.scene, then 0, to tolerate accidental string scenes.
            try:
                actual_scene = int(env_config.get("scene", 0))
            except Exception:
                actual_scene = 0

        return VirtualHomeEnvWrapper(
            scene=actual_scene,
            executable_path=env_config.get("executable_path"),
            port=env_config.get("port", "8080"),
            url=env_config.get("url", "127.0.0.1"),
            x_display=env_config.get("x_display"),
            width=env_config.get("width", 300),
            height=env_config.get("height", 300),
            output_dir=output_dir,
            config=config,
            num_agents=env_config.get("num_agents", 1),
            char_resource=env_config.get("char_resource", "Chars/Female1"),
        )

    else:
        raise ValueError(f"Unsupported environment type: {env_type}")


def main():
    """Main function"""
    # Load environment variables (API keys and other sensitive information)
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Multi-environment embodied agent framework - Supports AI2-THOR, CARLA, and VirtualHome",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Use default config file (automatically identify environment type)
  python -m scripts.virtualhome.work.run_task
  
  # Specify config file (read environment type from config file)
  python -m scripts.virtualhome.work.run_task --config experiments/configs/ai2thor/config_close_gpt-5.yaml
  
  # Specify tasks
  python -m scripts.virtualhome.work.run_task --tasks open_fridge open_microwave
  
  # Override scene and steps
  python -m scripts.virtualhome.work.run_task --scene FloorPlan5 --max-steps 50
  
  # Manually specify environment type (override config file)
  python -m scripts.virtualhome.work.run_task --env carla --config experiments/configs/carla/config_close_gpt-5.yaml
  python -m scripts.virtualhome.work.run_task --env virtualhome --config experiments/configs/virtualhome/config_close_gpt-5.yaml
  
  # View configuration information
  python -m scripts.virtualhome.work.run_task --print-config
""",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/virtualhome/config_close_gpt-5.yaml",
        help="Configuration file path (default: experiments/configs/virtualhome/config_close_gpt-5.yaml, read environment type from config file)",
    )

    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["ai2thor", "carla", "virtualhome"],
        help="Environment type (optional, overrides env.type in config file)",
    )

    parser.add_argument(
        "--scene", type=str, default=None, help="Override scene name in config file"
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override maximum steps in config file",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specify task list to execute (task preset names or task IDs from tasks/ directory), if not specified execute all tasks",
    )

    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print configuration information and exit",
    )

    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override output directory"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"\n{'=' * 60}")
    print("🔧 Loading Configuration")
    print(f"{'=' * 60}")
    print(f"Config file: {args.config}")

    config = load_config(args.config)

    # Read environment type from config file, default to ai2thor if not found
    env_type = args.env  # Command line argument takes priority
    if env_type is None:
        env_config = config.get_section("env")
        env_type = env_config.get("type", "ai2thor")

    print(f"Environment type: {env_type}")

    # Save default scene specified by command line
    default_scene_override = args.scene
    if default_scene_override:
        print(f"✓ Default scene set to: {default_scene_override}")

    if args.max_steps:
        config.update("max_steps", args.max_steps)
        print(f"✓ Maximum steps overridden: {args.max_steps}")

    if args.output_dir:
        config.update("experiment.output_dir", args.output_dir)
        print(f"✓ Output directory overridden: {args.output_dir}")

    # If only printing config, print and exit
    if args.print_config:
        print_config(config)
        return

    # Get task list to execute
    task_names = []
    tasks_from_folders = []  #    tasks         

    if args.tasks:
        #      tasks      ID
        error_messages = []  #       
        for task_input in args.tasks:
            #              ID（ ai2thor00000）
            if (
                task_input.startswith("ai2thor")
                or task_input.startswith("carla")
                or task_input.startswith("virtualhome")
                or os.path.exists(task_input)
            ):
                try:
                    task_config = load_task_from_folder(task_input)
                    task_name = task_config["name"]
                    task_names.append(task_name)
                    tasks_from_folders.append((task_name, task_config))
                    print(f"✓ Loaded task from folder: {task_input} -> {task_name}")
                except Exception as e:
                    error_msg = f"Failed to load task from {task_input}: {e}"
                    print(f"⚠️  {error_msg}")
                    print(f"   Treating '{task_input}' as preset task name")
                    error_messages.append(error_msg)
                    task_names.append(task_input)
            else:
                #         
                task_names.append(task_input)
        
        #        ，       （      ）
        if error_messages and len(error_messages) > 1:
            print(f"\n⚠️     {len(error_messages)}        :")
            for msg in error_messages:
                print(f"   - {msg}")

        if task_names:
            print(f"✓ Will execute specified tasks: {', '.join(task_names)}")
    else:
        task_names = config.get_all_task_names()
        print(f"✓ Will execute all tasks in order: {', '.join(task_names)}")

    # Create output directory
    output_dir = config.get("experiment.output_dir", "outputs")

    # If in multi-process mode, use specified directory directly
    if args.output_dir and os.path.basename(args.output_dir).startswith("worker_"):
        run_output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(output_dir, f"run_{timestamp}")

    os.makedirs(run_output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("🚀 Multi-Environment Embodied Agent Framework")
    print(f"{'=' * 60}")
    print(f"Environment: {env_type}")
    print(f"VLM Model: {config.get('model.vlm.model_name')}")
    print(f"Task count: {len(task_names)}")
    print(f"Output directory: {run_output_dir}")
    print(f"{'=' * 60}\n")

    # Initialize LLM
    vlm_config = config.get_section("model").get("vlm", {})
    default_temperature = 1.0 if env_type == "virtualhome" else 0.2
    default_max_tokens = 4096 if env_type == "virtualhome" else 2000
    vlm_model_kwargs: Dict[str, Any] = {}
    _seed_raw = vlm_config.get("seed")
    if _seed_raw is None:
        _seed_raw = os.environ.get("VLM_SEED")
    if _seed_raw is not None:
        try:
            vlm_model_kwargs["seed"] = int(_seed_raw)
        except (TypeError, ValueError):
            pass

    vlm = get_vlm(
        provider=vlm_config.get("provider", "openai"),
        model_name=vlm_config.get("model_name", "gpt-4o"),
        temperature=vlm_config.get("temperature", default_temperature),
        top_p=vlm_config.get("top_p", 0.9),
        max_tokens=vlm_config.get("max_tokens", default_max_tokens),
        base_url=vlm_config.get("base_url"),
        api_key=vlm_config.get("api_key"),
        model_kwargs=vlm_model_kwargs if vlm_model_kwargs else None,
    )

    # Create Agent graph
    agent_graph = create_agent_graph()

    # Statistics results
    all_results = []

    # Loop through each task
    for task_idx, task_name in enumerate(task_names, 1):
        print(f"\n{'=' * 60}")
        print(f"📋 Task {task_idx}/{len(task_names)}: {task_name}")
        print(f"{'=' * 60}")

        #       tasks       
        task_folder_config = None
        for folder_task_name, folder_config in tasks_from_folders:
            if folder_task_name == task_name:
                task_folder_config = folder_config
                break

        # Apply current task configuration
        if task_folder_config:
            #    tasks       
            print(
                f"📁 Using task from folder: {task_folder_config.get('task_folder_path', 'N/A')}"
            )
            #         ，       task_presets
            #    apply_task_by_name      task.json   
            task_config = config.apply_task_by_name(task_name)
        else:
            #         
            task_config = config.apply_task_by_name(task_name)

        # Use instruction if available, fallback to description for backward compatibility
        task_instruction = task_config.get("instruction") or task_config.get(
            "description", "No description"
        )
        print(f"Instruction: {task_instruction}")
        print(f"Max steps: {task_config.get('max_steps', 30)}\n")

        # Create independent output directory for each task
        task_output_dir = os.path.join(run_output_dir, task_name)
        os.makedirs(task_output_dir, exist_ok=True)

        # Get task scene configuration (supports scene or map field)
        env_config = config.get_section("env")

        # Set default scene based on environment type
        if env_type == "carla":
            # CARLA uses map field, default Town01
            default_map = env_config.get("map", "Town01")
            task_scene = (
                task_config.get("map")
                or task_config.get("scene")
                or default_scene_override
                or default_map
            )
        elif env_type == "virtualhome":
            # VirtualHome uses integer scene index, default from env config (typically 0)
            task_scene = (
                task_config.get("scene")
                if task_config.get("scene") is not None
                else (
                    default_scene_override
                    if default_scene_override is not None
                    else env_config.get("scene", 0)
                )
            )
        else:
            # AI2-THOR uses scene field, default FloorPlan1
            task_scene = (
                task_config.get("scene") or default_scene_override or "FloorPlan1"
            )

        print(f"Scene/Map: {task_scene}")

        # Executor/input modality for CARLA
        executor_type = task_config.get(
            "executor", "vehicle" if env_type == "carla" else None
        )
        input_modality = task_config.get("input_modality")

        # Goal image (image_goal)
        goal_image_path = None
        if env_type == "carla" and input_modality == "image_goal":
            image_url = task_config.get("image_url")
            if image_url:
                from pathlib import Path

                if os.path.isabs(image_url):
                    goal_image_path = image_url
                else:
                    base_dir = task_config.get("task_folder_path") or os.path.join(
                        _default_tasks_root(), task_name
                    )
                    goal_image_path = str(Path(base_dir) / image_url)

                if os.path.exists(goal_image_path):
                    print(f"🎯 Goal Image: {goal_image_path}")
                else:
                    print(f"⚠️  Goal image not found: {goal_image_path}")
                    goal_image_path = None

        if executor_type:
            print(f"Executor: {executor_type}")

        # Dynamically create environment with task-specific scene
        try:
            env = create_env(
                env_type,
                config.get_all(),
                task_output_dir,
                scene=task_scene,
                executor_type=executor_type,
            )
        except NotImplementedError as e:
            print(f"❌ {e}")
            continue
        except Exception as e:
            print(f"❌ Failed to create environment: {e}")
            import traceback

            traceback.print_exc()
            continue

        try:
            # Use instruction if available, fallback to description for backward compatibility
            task_description = task_config.get("instruction") or task_config.get(
                "description", "Complete task"
            )

            # Load init actions from task folder (if exists)
            #       tasks   ，       ；    tasks/{task_name}
            if task_config.get("_from_tasks_folder") and task_config.get(
                "task_folder_path"
            ):
                task_folder_for_init = task_config.get("task_folder_path")
                print(f"📁 Loading init from task folder: {task_folder_for_init}")
            else:
                task_folder_for_init = os.path.join(_default_tasks_root(), task_name)

            init_data, init_scene = load_init_actions_from_folder(
                task_folder_for_init, env_type=env_type
            )

            # If init specifies a scene, use it instead
            if init_scene is not None:
                task_scene = init_scene

            reset_kwargs = {"scene": task_scene}
            if env_type == "virtualhome" and init_data:
                reset_kwargs.update(
                    {
                        "char_position": init_data.get("character_position"),
                        "char_rotation": init_data.get("character_rotation"),
                        "char_yaw_degrees": init_data.get("character_yaw_degrees"),
                        "camera_pitch": init_data.get("camera_pitch"),
                    }
                )

            observation = env.reset(task_description, **reset_kwargs)

            # Execute init actions and record the count
            init_action_count = 0
            if init_data:
                if "actions" in init_data:
                    actions = init_data.get("actions", [])
                    #      Done
                    if actions and str(actions[-1]).strip().upper() == "DONE":
                        actions = actions[:-1]

                    init_action_count, last_observation = execute_init_actions(
                        env, actions, config.get_all()
                    )
                    if last_observation:
                        observation = last_observation
                elif "initial_location" in init_data:
                    init_action_count = apply_init_coordinates(env, init_data)
                    if hasattr(env, "_get_current_observation"):
                        observation = env._get_current_observation()

            # Initialize state
            initial_state = {
                "observation": observation,
                "task_prompt": task_description,
                "env": env,
                "vlm": vlm,
                "step_count": 0,
                "max_steps": task_config.get("max_steps", 30),
                "structured_trajectory": [],
                "conversation_history": [],
                "short_term_history": [],
                "long_term_summary": "",
                "should_continue": True,
                "success": False,
                "fail_reason": None,
                "failure_type": None,
                "next_action": None,
                "task_done_by_model": False,
                "task_fail_by_model": False,
                "config": config.get_all(),
                "run_output_dir": task_output_dir,
                "init_action_count": init_action_count,  #         
                "executor_type": executor_type,
                "input_modality": input_modality,
                "goal_image_path": goal_image_path,
            }

            # Run Agent graph
            print("🎬 Starting task execution...\n")
            
            # Use stream() to capture intermediate states, so we can save files even when GraphRecursionError occurs
            from mllm_base_agent.agent.graph import GraphRecursionError
            from mllm_base_agent.agent.graph import final_node
            
            final_state = None
            last_state = initial_state
            
            recursion_limit = 1000
            try:
                # Stream through the graph execution to capture states
                for chunk in agent_graph.stream(
                    initial_state, config={"recursion_limit": recursion_limit}
                ):
                    # Update last_state with the latest state from the stream
                    for node_name, state_update in chunk.items():
                        if isinstance(state_update, dict):
                            # Merge the state update into last_state
                            last_state = {**last_state, **state_update}
                
                # If we get here, execution completed normally
                final_state = last_state
                
            except GraphRecursionError as e:
                # Recursion limit reached - save files with last state
                print(f"\n\n⚠️  Recursion limit reached ({recursion_limit} steps): {e}")
                final_state = last_state
                # Mark as failure with specific reason
                final_state["success"] = False
                final_state["failure_type"] = "model_error"
                final_state["fail_reason"] = (
                    f"Reached recursion limit ({recursion_limit} steps)"
                )
                
                # Save files using the last valid state
                try:
                    save_conversation_log(final_state, task_output_dir)
                    final_node(final_state)  # Save episode_*.json
                except Exception as save_error:
                    print(f"\n⚠️  Failed to save files after recursion limit: {save_error}")
                    import traceback
                    traceback.print_exc()
            else:
                # Normal execution completed - save conversation log
                # Note: final_node is already called by the graph execution
                if final_state:
                    try:
                        save_conversation_log(final_state, task_output_dir)
                    except Exception as save_error:
                        print(f"\n⚠️  Failed to save conversation log: {save_error}")

            # Record task result
            if final_state:
                all_results.append(
                    {
                        "task_name": task_name,
                        "success": final_state.get("success", False),
                        "step_count": final_state.get("step_count", 0),
                        "fail_reason": final_state.get("fail_reason"),
                    }
                )

                # Print single task result
                print_final_results(final_state)
            else:
                # Fallback if final_state is None
                all_results.append(
                    {
                        "task_name": task_name,
                        "success": False,
                        "step_count": 0,
                        "fail_reason": "Unknown error: no final state",
                    }
                )

        except KeyboardInterrupt:
            print(
                f"\n\n⚠️  User interrupted execution, completed {task_idx - 1}/{len(task_names)} tasks"
            )
            env.close()
            break

        except Exception as e:
            print(f"\n\n❌ Error occurred while executing task '{task_name}': {e}")
            import traceback

            traceback.print_exc()

            all_results.append(
                {
                    "task_name": task_name,
                    "success": False,
                    "step_count": 0,
                    "fail_reason": f"Exception: {str(e)}",
                }
            )

        finally:
            env.close()

    # Print summary results for all tasks
    print_summary_results(all_results, run_output_dir)


def save_conversation_log(state: dict, output_dir: str):
    """Save complete conversation history as multi-turn dialogue JSON format"""
    import json

    log_file = os.path.join(output_dir, "log.json")

    # Build multi-turn conversation JSON structure
    conversation_json = {
        "metadata": {
            "task_description": state.get("task_prompt", "Unknown task"),
            "task_result": _derive_task_result(state),
            "failure_type": state.get("failure_type"),
            "fail_reason": state.get("fail_reason"),
            "total_steps": state.get("step_count", 0),
            "max_steps": state.get("max_steps", 0),
            "token_usage": state.get(
                "token_usage",
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "api_calls": 0,
                },
            ),
        },
        "messages": [],
        "images": [],
    }

    # Get conversation history from short_term_history (which contains image paths)
    short_term_history = state.get("short_term_history", [])

    # Also get structured_trajectory for complete history
    structured_trajectory = state.get("structured_trajectory", [])

    # Use structured_trajectory as the primary source (it has complete history)
    history_source = (
        structured_trajectory if structured_trajectory else short_term_history
    )

    if history_source:
        for i, entry in enumerate(history_source):
            step_id = entry.get("step", i + 1)
            image_path = entry.get("image_path", "")
            raw_response = entry.get("raw_response", "")
            error_message = entry.get("error_message")
            action_string = entry.get("action_string", "")
            reward = entry.get("reward", 0)

            # Build user message content (same as API call: only step info + image)
            user_content = f"Step {step_id}"
            if image_path:
                user_content += "\n<image>"
                conversation_json["images"].append(image_path)

            # User message: content matches API call, extra fields for logging only
            conversation_json["messages"].append(
                {
                    "role": "user",
                    "content": user_content,
                    # --- Fields below are for logging/analysis only, not sent to API ---
                    "step": step_id,
                    "image_path": image_path,
                }
            )

            # Assistant message: content matches API response, extra fields for logging only
            conversation_json["messages"].append(
                {
                    "role": "assistant",
                    "content": raw_response,
                    # --- Fields below are for logging/analysis only, not sent to API ---
                    "step": step_id,
                    "action_executed": action_string,
                    "reward": reward,
                    "error_message": error_message,
                }
            )

    # Save as JSON
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(conversation_json, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Conversation log saved: {log_file}")


def print_final_results(state: dict):
    """Print final results"""
    print(f"\n{'=' * 60}")
    print("📊 Final Results")
    print(f"{'=' * 60}")
    print(f"Success Status: {'✅ Success' if state['success'] else '❌ Failure'}")
    print(f"Total Steps: {state['step_count']}/{state['max_steps']}")

    if state["fail_reason"]:
        print(f"Failure Reason: {state['fail_reason']}")

    print("\n📁 Detailed JSON logs saved in current task directory")
    print(f"{'=' * 60}\n")


def print_summary_results(results: list, output_dir: str):
    """Print summary results for all tasks"""
    print(f"\n{'=' * 80}")
    print("🎉 All Tasks Completed")
    print(f"{'=' * 80}\n")

    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)

    print(f"Total Tasks: {total_count}")
    print(f"Success: {success_count} | Failure: {total_count - success_count}")

    if total_count > 0:
        print(f"Success Rate: {success_count / total_count * 100:.1f}%\n")
    else:
        print("Success Rate: N/A (no tasks executed)\n")

    if total_count > 0:
        print("Detailed Results:")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            status = "✅ Success" if result["success"] else "❌ Failure"
            print(
                f"{i}. {result['task_name']:20s} | {status:8s} | Steps: {result['step_count']:3d}",
                end="",
            )
            if result["fail_reason"]:
                print(f" | Reason: {result['fail_reason']}")
            else:
                print()
        print("-" * 80)

    print(f"\n📁 All results saved to: {output_dir}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
