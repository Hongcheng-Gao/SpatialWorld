"""
AI2-THOR task runner.

Runs one or more AI2-THOR tasks from a config. Benchmark scripts call this file as their worker.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from actions.max_steps import resolve_max_steps_from_task

from config import load_config, print_config
from core.llm.provider import get_vlm
from core.agent.graph import create_agent_graph


def load_init_actions_from_folder(task_folder_path: str) -> tuple:
    """Load init.json from task folder (supports action sequence or coordinate init).

    Returns:
        (init_data, scene_name): init_data may contain actions or initial_location/initial_rotation;
        scene_name from init.json or None.
    """
    import json

    init_file = os.path.join(task_folder_path, "init.json")

    if not os.path.exists(init_file):
        return None, None

    try:
        with open(init_file, "r", encoding="utf-8") as f:
            init_data = json.load(f)

        # Supported: list of actions; or dict with scene+actions; or dict with initial_location/initial_rotation
        if isinstance(init_data, list):
            return {"actions": init_data}, None
        if isinstance(init_data, dict):
            scene_name = init_data.get("scene")

            if "initial_location" in init_data:
                return init_data, scene_name
            if "actions" in init_data:
                return init_data, scene_name

            print("⚠️  init.json format error, skipping init")
            return None, None

        print("⚠️  init.json format error, skipping init")
        return None, None

    except Exception as e:
        print(f"⚠️  Failed to read init.json: {e}")
        return None, None


def apply_init_coordinates(env, init_data: dict) -> int:
    """Apply initial coordinates (location and rotation)."""
    try:
        import carla
    except ImportError:
        print("⚠️  carla not installed, cannot apply coordinate init")
        return 0

    location_list = init_data.get("initial_location")
    rotation_list = init_data.get("initial_rotation")

    if not location_list or not rotation_list:
        print("⚠️  init.json missing location or rotation")
        return 0

    print(f"\n{'=' * 60}")
    print("📍 Applying initial coordinates")
    print(f"{'=' * 60}")
    print(
        f"  Location: ({location_list[0]:.2f}, {location_list[1]:.2f}, {location_list[2]:.2f})"
    )
    print(
        f"  Rotation: (pitch={rotation_list[0]:.1f}°, yaw={rotation_list[1]:.1f}°, roll={rotation_list[2]:.1f}°)"
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
            print("  ✓ Walker position set")
        elif hasattr(env, "vehicle") and getattr(env, "vehicle"):
            env.vehicle.set_transform(transform)
            env.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            print("  ✓ Vehicle position set")
        else:
            print("  ⚠️  No actor to set")
            return 0

        if hasattr(env, "world"):
            for _ in range(5):
                env.world.tick()

        print(f"{'=' * 60}\n")
        return 1

    except Exception as e:
        print(f"  ❌ Failed to set position: {e}")
        import traceback

        traceback.print_exc()
        return 0


def execute_init_actions(env, init_action_strings: list, config: dict) -> tuple:
    """Execute init action sequence.

    Returns:
        (init_action_count, last_observation).
    """
    from core.agent.graph import parse_action_string

    if not init_action_strings:
        return 0, None

    print(f"\n{'=' * 60}")
    print(f"📁 Executing init action sequence ({len(init_action_strings)} actions)")
    print(f"{'=' * 60}")

    init_count = 0
    last_observation = None

    for i, action_str in enumerate(init_action_strings, 1):
        action_str = action_str.strip()

        if not action_str or action_str.upper() == "DONE":
            break

        print(f"  {i}. Executing init action: {action_str}")

        try:
            action_dict = parse_action_string(action_str)
            observation, error_message = env.step_with_action_dict(action_dict)
            last_observation = observation

            init_count += 1

            if error_message:
                print(f"     ⚠️  {error_message}")
            else:
                print("     ✓ Success")

        except Exception as e:
            print(f"     ❌ Parse/execute failed: {e}")
            continue

    print(f"\n✓ Init complete ({init_count} steps)\n")

    return init_count, last_observation




def load_task_from_folder(task_folder_path: str) -> Dict[str, Any]:
    """Load task config from tasks/ folder.

    Args:
        task_folder_path: Path or task ID (e.g. ai2thor00000).

    Returns:
        Task config dict.

    Raises:
        FileNotFoundError: Folder or task.json not found.
        ValueError: task.json format error.
    """
    import json
    from pathlib import Path

    input_path = Path(task_folder_path).expanduser()
    candidates = []
    if input_path.is_absolute():
        candidates.append(input_path)
    else:
        candidates.extend([Path.cwd() / input_path, Path(__file__).resolve().parents[3] / input_path])

    task_names = [input_path.name]
    if "_" in input_path.name:
        task_names.append(input_path.name.replace("_", ""))
    elif input_path.name.lower().startswith("ai2thor"):
        task_names.append(input_path.name.replace("ai2thor", "ai2thor_", 1))

    repo_root = Path(__file__).resolve().parents[3]
    for root in (repo_root / "tasks", repo_root / "data" / "ai2thor" / "tasks"):
        for name in task_names:
            candidates.append(root / name)

    task_folder = next(
        (candidate for candidate in candidates if candidate.is_dir() and (candidate / "task.json").exists()),
        candidates[0],
    )

    if not task_folder.exists():
        raise FileNotFoundError(f"Task folder not found: {task_folder}")

    task_json_path = task_folder / "task.json"

    if not task_json_path.exists():
        raise FileNotFoundError(f"task.json not found: {task_json_path}")

    try:
        with open(task_json_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)

        task_config = {
            "name": task_data.get("task_id", task_folder.name),
            "scene": task_data.get("scene", "FloorPlan1"),
            "instruction": task_data.get("instruction")
            or task_data.get("task_name", ""),
            "description": task_data.get(
                "target_description", task_data.get("instruction", "")
            ),
            "target_object_types": task_data.get("target_object_types", []),
            "success_condition": task_data.get("success_conditions", []),
            "max_steps": resolve_max_steps_from_task(task_data, 50),
            "task_folder_path": str(task_folder.absolute()),
            "task_id": task_data.get("task_id", task_folder.name),
            "_from_tasks_folder": True,
        }

        return task_config

    except json.JSONDecodeError as e:
        raise ValueError(f"task.json format error: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load task config: {e}")


def create_env(
    env_type: str,
    config: Dict[str, Any],
    output_dir: str,
    scene: Optional[str] = None,
    executor_type: Optional[str] = None,
):
    """Dynamically create environment instance

    Args:
        env_type: Environment type ('ai2thor' or 'carla')
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

    else:
        raise ValueError(f"Unsupported environment type: {env_type}")


def main():
    """Main function"""
    # Load environment variables (API keys and other sensitive information)
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="AI2-THOR task runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Use default config file (automatically identify environment type)
  python -m scripts.ai2thor.work.run_task
  
  # Specify config file (read environment type from config file)
  python -m scripts.ai2thor.work.run_task --config experiments/configs/ai2thor/config_close_gpt-5.yaml
  
  # Specify tasks
  python -m scripts.ai2thor.work.run_task --tasks open_fridge open_microwave
  
  # Override scene and steps
  python -m scripts.ai2thor.work.run_task --scene FloorPlan5 --max-steps 50
  
  # View configuration information
  python -m scripts.ai2thor.work.run_task --print-config
""",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/ai2thor/config_close_gpt-5.yaml",
        help="Configuration file path (default: experiments/configs/ai2thor/config_close_gpt-5.yaml, read environment type from config file)",
    )

    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=["ai2thor", "carla"],
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

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run AI2-THOR with CloudRendering (no display server)",
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

    if args.headless:
        config.update("env.platform", "CloudRendering")
        print("✓ Headless mode enabled: env.platform=CloudRendering")

    # If only printing config, print and exit
    if args.print_config:
        print_config(config)
        return

    task_names = []
    tasks_from_folders = []

    if args.tasks:
        error_messages = []
        for task_input in args.tasks:
            if (
                task_input.startswith("ai2thor")
                or task_input.startswith("carla")
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
                task_names.append(task_input)
        
        if error_messages and len(error_messages) > 1:
            print(f"\n⚠️  {len(error_messages)} task load error(s):")
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
    if vlm_config.get("provider", "openai").lower() == "openai" and not (
        vlm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
    ):
        print(
            "❌ Missing API key for the OpenAI-compatible VLM provider. "
            "Set OPENAI_API_KEY or model.vlm.api_key in the config."
        )
        sys.exit(1)
    vlm = get_vlm(
        provider=vlm_config.get("provider", "openai"),
        model_name=vlm_config.get("model_name", "gpt-4o"),
        temperature=vlm_config.get("temperature", 0.2),
        max_tokens=vlm_config.get("max_tokens", 2000),
        top_p=vlm_config.get("top_p"),
        base_url=vlm_config.get("base_url"),
        api_key=vlm_config.get("api_key"),
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

        task_folder_config = None
        for folder_task_name, folder_config in tasks_from_folders:
            if folder_task_name == task_name:
                task_folder_config = folder_config
                break

        if task_folder_config:
            print(
                f"📁 Using task from folder: {task_folder_config.get('task_folder_path', 'N/A')}"
            )
            task_config = task_folder_config
            config.config["task"] = task_config
        else:
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
                        "tasks", task_name
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
            # Reset environment
            # Use instruction if available, fallback to description for backward compatibility
            task_description = task_config.get("instruction") or task_config.get(
                "description", "Complete task"
            )

            # Both AI2-THOR and CARLA need scene/map in reset for task-specific scenes
            observation = env.reset(task_description, scene=task_scene)

            # Load init actions from task folder (if exists)
            if task_config.get("_from_tasks_folder") and task_config.get(
                "task_folder_path"
            ):
                task_folder_for_init = task_config.get("task_folder_path")
                print(f"📁 Loading init from task folder: {task_folder_for_init}")
            else:
                task_folder_for_init = os.path.join("tasks", task_name)

            init_data, init_scene = load_init_actions_from_folder(task_folder_for_init)

            # If init specifies a scene, use it instead
            if init_scene:
                task_scene = init_scene
                observation = env.reset(task_description, scene=task_scene)

            # Execute init actions and record the count
            init_action_count = 0
            if init_data:
                if "actions" in init_data:
                    actions = init_data.get("actions", [])
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
                "max_steps_override": args.max_steps,
                "structured_trajectory": [],
                "conversation_history": [],
                "short_term_history": [],
                "long_term_summary": "",
                "should_continue": True,
                "success": False,
                "fail_reason": None,
                "failure_type": None,  # 'api_error', 'parse_error', 'env_error', or None
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "api_calls": 0,
                },
                "next_action": None,
                "think_failed": False,  # Flag to track if think_node failed
                "task_done_by_model": False,
                "task_fail_by_model": False,
                "config": config.get_all(),
                "run_output_dir": task_output_dir,
                "init_action_count": init_action_count,
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
            
            try:
                # Stream through the graph execution to capture states
                for chunk in agent_graph.stream(
                    initial_state, config={"recursion_limit": 1000}
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
                print(f"\n\n⚠️  Recursion limit reached (100 steps): {e}")
                final_state = last_state
                # Mark as failure with specific reason
                final_state["success"] = False
                final_state["fail_reason"] = "Reached recursion limit (100 steps)"
                
                # Save files using the last valid state
                try:
                    save_conversation_log(final_state, task_output_dir)
                    final_node(final_state)  # Save episode_*.json
                except Exception as save_error:
                    print(f"\n⚠️  Failed to save files after recursion limit: {save_error}")
                    import traceback
                    traceback.print_exc()

            except Exception as graph_error:
                # Any other exception from graph execution (API errors, env crashes, etc.)
                # Save files with last known state so run_csv_benchmark can categorize failure
                print(f"\n\n⚠️  Graph execution error: {graph_error}")
                import traceback
                traceback.print_exc()
                final_state = last_state
                final_state["success"] = False
                if not final_state.get("fail_reason"):
                    final_state["fail_reason"] = f"Graph execution error: {str(graph_error)}"
                if not final_state.get("failure_type"):
                    # Infer failure type from error message
                    err_str = str(graph_error).lower()
                    if any(kw in err_str for kw in ["api", "connection", "timeout", "http", "429", "500", "rate"]):
                        final_state["failure_type"] = "api_error"
                    else:
                        final_state["failure_type"] = "env_error"
                
                try:
                    save_conversation_log(final_state, task_output_dir)
                    final_node(final_state)
                except Exception as save_error:
                    print(f"\n⚠️  Failed to save files after graph error: {save_error}")
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
            "task_result": "success" if state.get("success") else "failure",
            "fail_reason": state.get("fail_reason"),
            "failure_type": state.get("failure_type"),  # 'api_error', 'parse_error', 'env_error', or None
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
            llm_token_usage = entry.get("llm_token_usage")

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
                    "llm_token_usage": llm_token_usage,
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
