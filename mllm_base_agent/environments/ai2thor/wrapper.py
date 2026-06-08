"""
AI2-THOR Environment Wrapper
Provides unified environment interaction interface, supports generalized configuration for various 3D tasks

Update: Supports loading environment, task, action, and reward parameters from YAML configuration files
"""

import os
import json
import math
from typing import Optional, Callable, List, Dict, Any, Tuple
from datetime import datetime
from PIL import Image
import numpy as np
from ai2thor.platform import CloudRendering

try:
    import ai2thor.controller
except ImportError:
    print("Warning: ai2thor not installed, please run 'pip install ai2thor'")

from core.llm.schemas import EnvAction, EnvObservation
from envs.base import BaseEnv


# ============================================================================
#      ：    ->           
#          ，      PickupObject(Apple)  ，
#         ["Apple", "AppleSliced"]         
# ============================================================================
SEMANTIC_OBJECT_MAPPING = {
    #      
    "Apple": ["Apple", "AppleSliced"],
    "Bread": ["Bread", "BreadSliced", "BreadToasted"],
    "Tomato": ["Tomato", "TomatoSliced"],
    "Lettuce": ["Lettuce", "LettuceSliced"],
    "Potato": ["Potato", "PotatoSliced", "PotatoCooked"],
    "Egg": ["Egg", "EggSliced", "EggBroken", "EggCooked"],
    #      
    "Bottle": ["Bottle", "BottleBroken"],
    "Cup": ["Cup", "CupBroken"],
    "Mug": ["Mug", "MugBroken"],
    "Plate": ["Plate", "PlateBroken"],
    "Vase": ["Vase", "VaseBroken"],
    "WineBottle": ["WineBottle", "WineBottleBroken"],
    "Window": ["Window", "WindowBroken"],
    "Statue": ["Statue", "StatueBroken"],
    #      
    "PaperTowelRoll": ["PaperTowelRoll", "PaperTowel"],
}


class AI2ThorEnvWrapper(BaseEnv):
    """AI2-THOR Environment Wrapper

    Wraps AI2-THOR's underlying API, provides unified environment interaction interface.
    Supports configuring task parameters via YAML configuration or Python code, enabling generalization for various 3D tasks.

    Configuration source priority:
    1. YAML configuration file (passed via config parameter)
    2. configure_task() method parameters (runtime configuration)
    3. Default values (backward compatibility)
    """

    def __init__(
        self,
        scene: str = "FloorPlan1",
        grid_size: float = 0.25,
        render_depth_image: bool = False,
        render_instance_segmentation: bool = False,
        width: int = 800,
        height: int = 600,
        output_dir: str = "outputs",
        config: Optional[Dict[str, Any]] = None,  # New: Configuration loaded from YAML
    ):
        """Initialize environment

        Args:
            scene: AI2-THOR scene name (e.g., FloorPlan1, FloorPlan28)
            grid_size: Grid size, affects movement precision
            render_depth_image: Whether to render depth image
            render_instance_segmentation: Whether to render instance segmentation
            width: Image width
            height: Image height
            output_dir: Output directory
            config: Complete configuration dictionary loaded from YAML (optional)
        """
        # Call parent class initialization
        super().__init__(config)

        # Save configuration reference (override parent's simple assignment)
        self.config = config or {}

        # Get environment settings from config or parameters
        env_config = self.config.get("env", {})
        self.scene = scene  # Directly use passed scene parameter (determined by task configuration)
        self.grid_size = env_config.get("grid_size", grid_size)
        self.width = env_config.get("width", width)
        self.height = env_config.get("height", height)
        self.field_of_view = env_config.get("field_of_view", 60)  # Camera FOV in degrees, default 60 (non-wide angle)
        render_depth_image = env_config.get("render_depth", render_depth_image)
        render_instance_segmentation = env_config.get(
            "render_instance_segmentation", render_instance_segmentation
        )
        self.text_state_mode = env_config.get("text_state_mode", "first_person")
        if self.text_state_mode not in {"first_person", "omniscient"}:
            print(
                f"⚠️  Unknown text_state_mode: {self.text_state_mode}, falling back to first_person"
            )
            self.text_state_mode = "first_person"
        platform_setting = env_config.get("platform")
        if (
            isinstance(platform_setting, str)
            and platform_setting.lower() == "cloudrendering"
        ):
            self.controller_platform = CloudRendering
        else:
            self.controller_platform = platform_setting

        # Visibility distance configuration (affects obj["visible"] in metadata)
        # Default 1.0m (was 1.5m which is AI2-THOR's default visibilityDistance)
        self.visibility_distance = env_config.get("visibility_distance", 1.0)

        # Multi-agent (AI2-THOR agentCount): each embodied agent has its own body/camera.
        self.agent_count = int(env_config.get("agent_count", 1))
        if self.agent_count < 1:
            self.agent_count = 1

        self.output_dir = output_dir
        # step_counter and action_sequence already initialized in parent class
        self.task_description = ""

        # Task configuration (set from YAML or configure_task)
        task_config = self.config.get("task", {})
        self.target_object_types: List[str] = task_config.get("target_object_types", [])
        self.success_condition: Optional[Dict[str, Any]] = task_config.get(
            "success_condition", None
        )
        self.success_predicate: Optional[Callable[[dict], bool]] = (
            None  # Will be built when needed
        )
        self.target_description: str = task_config.get("target_description", "")

        # Action configuration (loaded from YAML)
        # Step granularity: Large=1m, Medium=0.5m, Small=0.25m
        actions_config = self.config.get("actions", {})
        self.move_small_magnitude = actions_config.get("move_small_magnitude", 0.25)
        self.move_medium_magnitude = actions_config.get("move_medium_magnitude", 0.5)
        self.move_large_magnitude = actions_config.get("move_large_magnitude", 1)
        self.move_ahead_magnitude = actions_config.get("move_ahead_magnitude", self.move_medium_magnitude)
        self.move_back_magnitude = actions_config.get("move_back_magnitude", self.move_medium_magnitude)
        self.move_left_magnitude = actions_config.get("move_left_magnitude", self.move_medium_magnitude)
        self.move_right_magnitude = actions_config.get("move_right_magnitude", self.move_medium_magnitude)
        self.rotate_degrees = actions_config.get("rotate_degrees", 90)

        # Reward configuration (loaded from YAML)
        reward_config = self.config.get("reward", {})
        self.success_reward = reward_config.get("success_reward", 10.0)
        self.step_success_bonus = reward_config.get("step_success_bonus", 0.1)
        self.step_failure_penalty = reward_config.get("step_failure_penalty", -0.05)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize AI2-THOR controller
        print(f"Initializing AI2-THOR scene: {self.scene}")
        controller_kwargs = {
            "scene": self.scene,
            "gridSize": self.grid_size,
            "renderDepthImage": render_depth_image,
            "renderInstanceSegmentation": render_instance_segmentation,
            "width": self.width,
            "height": self.height,
            "fieldOfView": self.field_of_view,  # Camera field of view in degrees
            "visibilityDistance": self.visibility_distance,  #   /      
            # Headless mode optional, may be needed on servers
            # "headless": True,
        }
        if self.controller_platform:
            controller_kwargs["platform"] = self.controller_platform
            platform_name = (
                "CloudRendering"
                if self.controller_platform is CloudRendering
                else str(self.controller_platform)
            )
            print(f"  • Using custom platform parameter: {platform_name}")

        if self.agent_count > 1:
            controller_kwargs["agentCount"] = self.agent_count
            print(f"  • Multi-agent embodied mode: agentCount={self.agent_count}")

        # Debug output: Display FOV value
        print(f"  • Camera Field of View (FOV): {self.field_of_view}°")

        self.controller = ai2thor.controller.Controller(**controller_kwargs)

        # Initialize scene with visibility distance
        self.controller.step(
            action="Initialize",
            gridSize=self.grid_size,
            visibilityDistance=self.visibility_distance,
        )

        print(
            f"✓ AI2-THOR environment initialization complete (visibilityDistance={self.visibility_distance}m)"
        )

        # If task information exists in config, automatically build success_predicate
        if self.success_condition:
            self.success_predicate = self._build_success_predicate_from_config()

    def _build_success_predicate_from_config(self) -> Callable[[dict], bool]:
        """Build success condition predicate from YAML configured success_condition

        Returns:
            Success condition predicate function
        """
        if not self.success_condition:
            return lambda obj: False

        condition_type = self.success_condition.get("type", "object_state")

        if condition_type == "object_state":
            field = self.success_condition.get("field", "isOpen")
            target_value = self.success_condition.get("value", True)

            # Return lambda function: check if object field equals target value
            return lambda obj: obj.get(field, False) == target_value

        elif condition_type == "object_in_receptacle":
            # New: Check if object is in specified receptacle
            # Note: This predicate checks a single object, actual determination is implemented in _check_done
            receptacle_type = self.success_condition.get("receptacle_type", "Plate")
            expected_value = self.success_condition.get(
                "value", True
            )  # Default True for backward compatibility

            # Check if object's parentReceptacles list contains target receptacle type
            def check_in_receptacle(obj):
                parent_receptacles = obj.get("parentReceptacles", [])
                if not parent_receptacles:
                    in_receptacle = False
                else:
                    # Check if there is a matching type in parent receptacle list
                    # parentReceptacles is an objectId list, format like "Plate|+01.23|-02.34"
                    in_receptacle = False
                    for parent_id in parent_receptacles:
                        # Extract type name (part before | in objectId)
                        parent_type = (
                            parent_id.split("|")[0] if "|" in parent_id else parent_id
                        )
                        if parent_type == receptacle_type:
                            in_receptacle = True
                            break

                # Return based on expected value: True means should be in, False means should not be in
                return in_receptacle == expected_value

            return check_in_receptacle

        elif condition_type == "object_in_hand":
            # Check if object is in hand
            # Actual determination is mainly done in _check_done through inventoryObjects
            # Here provides an auxiliary predicate based on isPickedUp
            return lambda obj: obj.get("isPickedUp", False)

        else:
            print(f"⚠️  Unsupported success_condition type: {condition_type}")
            return lambda obj: False

    def configure_task(
        self,
        target_object_types: List[str],
        success_predicate: Callable[[dict], bool],
        target_description: str,
    ):
        """Configure task parameters (generalized support for multiple tasks)

        Args:
            target_object_types: Target object type list (e.g., ["Fridge"] or ["Microwave", "Cabinet"])
            success_predicate: Success condition predicate, receives object metadata dict, returns bool
                              Example: lambda obj: obj.get("isOpen", False)  # Open task
                                  lambda obj: not obj.get("isOpen", False)  # Close task
                                  lambda obj: obj.get("isToggled", False)  # Toggle task
            target_description: Text description of task target (for display in text_state)
                              Example: "Find and open any Fridge"
                                  "Close all open Microwaves"

        Example usage:
            env.configure_task(
                target_object_types=["Fridge"],
                success_predicate=lambda obj: obj.get("isOpen", False),
                target_description="Find and open any Fridge"
            )
        """
        self.target_object_types = target_object_types
        self.success_predicate = success_predicate
        self.target_description = target_description
        print(f"✓ Task configuration complete: {target_description}")

    def _unwrap_agent_event(
        self, event: Any, thor_agent_id: Optional[int] = None
    ) -> Any:
        """Align multi-agent with single-agent: always use ``Event`` and ``event.frame``.

        ``controller.reset`` / ``step`` return ``Event`` when ``agent_count==1``.
        With multiple agents, ai2thor returns ``MultiAgentEvent`` (no ``.frame``);
        unwrap to the same per-agent ``Event`` instance the library uses internally.
        """
        if event is None:
            return None
        if getattr(event, "frame", None) is not None:
            return event
        evs = getattr(event, "events", None)
        if isinstance(evs, list) and evs:
            if thor_agent_id is not None:
                i = int(thor_agent_id)
                if 0 <= i < len(evs):
                    return evs[i]
            active = getattr(event, "_active_event", None)
            if active is not None:
                return active
            return evs[0]
        return event

    def reset(
        self, task_description: str, scene: Optional[str] = None
    ) -> EnvObservation:
        """Reset environment to initial state

        Args:
            task_description: Task description
            scene: Optional scene name, if provided will switch to that scene

        Returns:
            Initial observation
        """
        self.step_counter = 0

        # If new scene provided, switch scene
        if scene and scene != self.scene:
            print(f"🔄 Switching scene: {self.scene} -> {scene}")
            self.scene = scene

        # If task parameters not yet configured, use default "open fridge" task (backward compatibility)
        if not self.target_object_types:
            print(
                "⚠️  Warning: Task parameters not configured, using default task (open Fridge)"
            )
            self.configure_task(
                target_object_types=["Fridge"],
                success_predicate=lambda obj: obj.get("isOpen", False),
                target_description="Find and open any Fridge",
            )

        # Reset scene
        event = self.controller.reset(scene=self.scene)

        # Task description (interactable objects are now fixed in system prompt)
        self.task_description = task_description

        # Same as single-agent: save ``Event.frame`` (unwrap MultiAgentEvent if needed)
        ev = self._unwrap_agent_event(event)
        image_path = self._save_frame(
            ev.frame,
            prefix="reset",
            thor_agent_id=0 if self.agent_count > 1 else None,
        )
        if self.agent_count > 1:
            ev1 = self.controller.step(action="Pass", agentId=1)
            ev1u = self._unwrap_agent_event(ev1, thor_agent_id=1)
            self._save_frame(ev1u.frame, prefix="reset", thor_agent_id=1)

        # Generate text state
        text_state = self._generate_text_state(event.metadata)

        print(f"\n{'=' * 60}")
        print(f"Environment reset | Task: {task_description[:50]}...")
        print(f"Scene: {self.scene}")
        print(f"{'=' * 60}")

        return EnvObservation(
            image_path=image_path,
            text_state=text_state,
            reward=0.0,
            done=False,
            metadata=event.metadata,
        )

    def get_observation_for_agent(self, thor_agent_id: int) -> EnvObservation:
        """Render the current world state from a specific embodied agent's camera.

        Uses a no-op Pass so AI2-THOR returns that agent's frame and metadata.
        Does not increment ``step_counter`` (not counted as a task step).

        Args:
            thor_agent_id: AI2-THOR agent index (0 = first body, 1 = second, ...).

        Returns:
            Observation from that agent's viewpoint.
        """
        step_kwargs: Dict[str, Any] = {"action": "Pass"}
        if self.agent_count > 1:
            step_kwargs["agentId"] = int(thor_agent_id)

        event = self.controller.step(**step_kwargs)
        # Match single-agent step image naming (not ``observe_a*``); per-body path via thor_agent_id.
        ev = self._unwrap_agent_event(event, thor_agent_id=int(thor_agent_id))
        image_path = self._save_frame(
            ev.frame,
            prefix=f"step_{self.step_counter}",
            thor_agent_id=int(thor_agent_id) if self.agent_count > 1 else None,
        )
        text_state = self._generate_text_state(event.metadata)
        done = self._check_done(event.metadata)
        return EnvObservation(
            image_path=image_path,
            text_state=text_state,
            reward=0.0,
            done=done,
            metadata=event.metadata,
        )

    @staticmethod
    def _agent_position_yaw_from_metadata(
        metadata: dict, agent_index: int
    ) -> Optional[Tuple[float, float, float, float]]:
        """Read (x, y, z, yaw_deg) for an embodied agent from THOR metadata."""
        agents = metadata.get("agents")
        if isinstance(agents, list) and len(agents) > agent_index:
            a = agents[agent_index]
            p = a.get("position") or {}
            r = a.get("rotation") or {}
            return (
                float(p.get("x", 0.0)),
                float(p.get("y", 0.0)),
                float(p.get("z", 0.0)),
                float(r.get("y", 0.0)),
            )
        if agent_index == 0 and metadata.get("agent"):
            a = metadata["agent"]
            p = a.get("position") or {}
            r = a.get("rotation") or {}
            return (
                float(p.get("x", 0.0)),
                float(p.get("y", 0.0)),
                float(p.get("z", 0.0)),
                float(r.get("y", 0.0)),
            )
        return None

    def _probe_agent_can_move_ahead_and_back(self, thor_agent_id: int) -> bool:
        """True if agent can MoveAhead then MoveBack (navigable, not stuck)."""
        m = self.move_small_magnitude
        ev = self.controller.step(
            action="MoveAhead",
            moveMagnitude=m,
            agentId=int(thor_agent_id),
        )
        if not ev.metadata.get("lastActionSuccess"):
            return False
        ev2 = self.controller.step(
            action="MoveBack",
            moveMagnitude=m,
            agentId=int(thor_agent_id),
        )
        return bool(ev2.metadata.get("lastActionSuccess"))

    def relocate_second_agent_near_agent1(self) -> bool:
        """Place agentId=1 adjacent to agentId=0 (front/back/left/right in agent1's frame).

        Tries directions in order: forward, back, left, right at a fixed horizontal offset.
        Uses TeleportFull for placement, then a short MoveAhead/MoveBack probe to reject
        stuck / blocked spawns. Controlled by ``config['dual_agent']``:

        - ``relocate_agent2_near_agent1`` (default True): set False to skip.
        - ``second_agent_spawn_offset_m`` (default 0.75): horizontal offset in meters.

        Returns:
            True if agent 2 was placed successfully, False if all candidates failed
            (agent 2 is left at the engine default spawn).
        """
        if self.agent_count < 2:
            return True

        da = self.config.get("dual_agent") or {}
        if da.get("relocate_agent2_near_agent1") is False:
            print("  ℹ️  Skipping agent2 relocation (relocate_agent2_near_agent1=false)")
            return True

        offset_m = float(da.get("second_agent_spawn_offset_m", 0.75))
        if offset_m <= 0:
            offset_m = 0.75

        ev0 = self.controller.step(action="Pass", agentId=0)
        pose0 = self._agent_position_yaw_from_metadata(ev0.metadata, 0)
        if pose0 is None:
            print("⚠️  relocate_second_agent_near_agent1: could not read agent 1 pose")
            return False

        x0, y0, z0, yaw_deg = pose0
        yaw_rad = math.radians(yaw_deg)
        # Horizontal forward / right in XZ from body yaw (THOR convention)
        fx = math.sin(yaw_rad)
        fz = math.cos(yaw_rad)
        rx = math.cos(yaw_rad)
        rz = -math.sin(yaw_rad)

        direction_offsets: List[Tuple[str, float, float]] = [
            ("forward", fx * offset_m, fz * offset_m),
            ("back", -fx * offset_m, -fz * offset_m),
            ("left", -rx * offset_m, -rz * offset_m),
            ("right", rx * offset_m, rz * offset_m),
        ]

        print(
            f"\n{'=' * 60}\n"
            f"Dual-agent spawn: placing agent 2 near agent 1 "
            f"(offset={offset_m:.2f}m, order=forward/back/left/right)\n"
            f"{'=' * 60}"
        )

        rotation = {"x": 0, "y": yaw_deg, "z": 0}

        for label, dx, dz in direction_offsets:
            tx = x0 + dx
            tz = z0 + dz
            ty = y0

            ev = self.controller.step(
                action="TeleportFull",
                x=tx,
                y=ty,
                z=tz,
                rotation=rotation,
                horizon=0,
                standing=True,
                forceAction=True,
                agentId=1,
            )
            if not ev.metadata.get("lastActionSuccess"):
                err = ev.metadata.get("errorMessage", "")
                print(f"  • {label}: TeleportFull failed — {err or 'unknown'}")
                continue

            if self._probe_agent_can_move_ahead_and_back(1):
                print(
                    f"  ✓ Agent 2 placed to the {label} of agent 1 "
                    f"(probe MoveAhead/MoveBack OK)"
                )
                return True

            print(f"  • {label}: teleported but navigation probe failed, trying next")

        print(
            "⚠️  Could not place agent 2 in any of front/back/left/right; "
            "leaving default spawn for agentId=1"
        )
        return False

    def step(self, action: EnvAction) -> EnvObservation:
        """Execute one step action

        Args:
            action: Environment action

        Returns:
            New observation
        """
        self.step_counter += 1

        # Convert and execute action
        thor_actions = self._convert_action(action)

        # Record action
        print(f"\n--- Step {self.step_counter} ---")
        if action.comment:
            print(f"💭 Thinking: {action.comment}")
        print(f"🎬 Action: {self._format_action(action)}")

        # Record action sequence (record before execution to include parameters)
        for thor_action in thor_actions:
            action_name = thor_action.get("action", "Unknown")
            # Extract key parameters
            params_list = []
            if "moveMagnitude" in thor_action:
                params_list.append(f"{thor_action['moveMagnitude']}")
            if "degrees" in thor_action:
                params_list.append(f"{thor_action['degrees']}")
            if "objectId" in thor_action:
                # Only show objectType, not full objectId
                obj_id = thor_action["objectId"]
                obj_type = obj_id.split("|")[0] if "|" in obj_id else "Object"
                params_list.append(f"{obj_type}")

            params_str = ", ".join(params_list) if params_list else ""
            if params_str:
                self.action_sequence.append(f"{action_name}({params_str})")
            else:
                self.action_sequence.append(f"{action_name}()")

        # Execute all converted actions
        event = None
        for thor_action in thor_actions:
            event = self.controller.step(**thor_action)
            if not event.metadata["lastActionSuccess"]:
                print(
                    f"⚠️  Action failed: {event.metadata.get('errorMessage', 'Unknown error')}"
                )
                break

        # Save current frame (same pattern as single-agent: ``Event.frame``)
        ev = self._unwrap_agent_event(event)
        image_path = self._save_frame(
            ev.frame,
            prefix=f"step_{self.step_counter}",
            thor_agent_id=0 if self.agent_count > 1 else None,
        )

        # Generate text state
        text_state = self._generate_text_state(event.metadata)

        # Calculate reward (simple version)
        reward = self._compute_reward(action, event.metadata)

        # Check if done
        done = self._check_done(event.metadata)

        return EnvObservation(
            image_path=image_path,
            text_state=text_state,
            reward=reward,
            done=done,
            metadata=event.metadata,
        )

    def step_with_action_dict(
        self,
        action_dict: dict,
        thor_agent_id: Optional[int] = None,
        vision_metadata: Optional[dict] = None,
    ) -> tuple[EnvObservation, Optional[str]]:
        """Execute action using new action dictionary format

        Args:
            action_dict: Action dictionary containing action_type, action_name, object_type, and optional parameters
            thor_agent_id: When ``agent_count > 1``, which embodied agent executes the action (AI2-THOR ``agentId``).
            vision_metadata: When resolving interactions (e.g. PickupObject), use this metadata (the acting
                agent's last observation) instead of ``controller.last_event``, which may belong to another body
                in multi-agent mode.

        Returns:
            (observation, error_message) tuple, error_message is None if successful
        """
        self.step_counter += 1

        action_type = action_dict.get("action_type")
        action_name = action_dict.get("action_name")
        object_type = action_dict.get("object_type")
        fill_liquid = action_dict.get("fillLiquid")  #         （   ）

        print(f"\n--- Step {self.step_counter} ---")
        print(f"🎬 Action: {action_name}" + (f"({object_type})" if object_type else ""))

        error_message = None
        thor_action = None

        # Convert to AI2-THOR action
        if action_type == "navigation":
            mag = self._resolve_move_magnitude(
                magnitude=action_dict.get("magnitude"),
                granularity=action_dict.get("granularity"),
            )
            deg = action_dict.get("degrees")
            thor_action = self._convert_navigation_action(action_name, magnitude=mag, degrees=deg)
        elif action_type == "interaction":
            thor_action, error_message = self._convert_interaction_action(
                action_name, object_type, scene_metadata=vision_metadata
            )
            #    FillObjectWithLiquid  fillLiquid  ，   thor_action 
            if thor_action and action_name == "FillObjectWithLiquid" and fill_liquid:
                thor_action["fillLiquid"] = fill_liquid
                print(f"  💧 Liquid type: {fill_liquid}")
        elif action_type == "task_completion":
            # Task completion action (DONE/STOP) - just pass without error
            # The actual termination logic should be handled by the caller
            thor_action = {"action": "Pass"}
            print(f"🏁 Task completion action: {action_name}")
        else:
            error_message = f"Unknown action type: {action_type}"
            thor_action = {"action": "Pass"}

        if thor_action is None:
            thor_action = {"action": "Pass"}

        if self.agent_count > 1:
            aid = 0 if thor_agent_id is None else int(thor_agent_id)
            thor_action["agentId"] = aid
            print(f"  🤖 Embodied agent (agentId): {aid}")

        # Record action sequence
        # For PickupObject actions, use semantic parent type instead of actual type
        # This ensures action sequence shows PickupObject(Apple) even if we picked AppleSliced
        action_name_str = thor_action.get("action", "Unknown")
        params_list = []
        if "moveMagnitude" in thor_action:
            params_list.append(f"{thor_action['moveMagnitude']}")
        if "degrees" in thor_action:
            params_list.append(f"{thor_action['degrees']}")
        if "objectId" in thor_action:
            obj_id = thor_action["objectId"]
            # Extract actual object type from objectId
            actual_obj_type = obj_id.split("|")[0] if "|" in obj_id else "Object"

            # For interaction actions, use semantic parent type
            if action_name_str in [
                "PickupObject",
                "PutObject",
                "ThrowObject",
                "SliceObject",
                "BreakObject",
                "CookObject",
                "OpenObject",
                "CloseObject",
                "DirtyObject",
                "ToggleObjectOn",
                "ToggleObjectOff",
                "FillObjectWithLiquid",
                "EmptyLiquidFromObject",
            ]:
                # Try to find semantic parent (e.g., AppleSliced -> Apple)
                semantic_parent = None
                for parent, variants in SEMANTIC_OBJECT_MAPPING.items():
                    if actual_obj_type in variants:
                        semantic_parent = parent
                        break
                # Use semantic parent if found, otherwise use actual type
                obj_type_to_record = (
                    semantic_parent if semantic_parent else actual_obj_type
                )
            else:
                obj_type_to_record = actual_obj_type

            params_list.append(f"{obj_type_to_record}")

            #     ：FillObjectWithLiquid           
            if (
                action_name_str == "FillObjectWithLiquid"
                and "fillLiquid" in thor_action
            ):
                params_list.append(thor_action["fillLiquid"])

        params_str = ", ".join(params_list) if params_list else ""
        if params_str:
            self.action_sequence.append(f"{action_name_str}({params_str})")
        else:
            self.action_sequence.append(f"{action_name_str}()")

        # Execute action
        event = self.controller.step(**thor_action)

        # Check if action succeeded
        if not event.metadata["lastActionSuccess"]:
            raw_error = event.metadata.get("errorMessage", "Unknown error")
            error_message = self._translate_error_message(
                raw_error, action_name, object_type
            )
            print(f"⚠️  Action failed: {error_message}")

        frame_agent_id = (
            thor_action.get("agentId") if self.agent_count > 1 else None
        )
        ev = self._unwrap_agent_event(event, thor_agent_id=frame_agent_id)
        aid_save = (
            (0 if thor_agent_id is None else int(thor_agent_id))
            if self.agent_count > 1
            else None
        )
        image_path = self._save_frame(
            ev.frame, prefix=f"step_{self.step_counter}", thor_agent_id=aid_save
        )

        # Generate text state
        text_state = self._generate_text_state(event.metadata)

        # Calculate reward
        reward = self._compute_reward_from_metadata(
            event.metadata, action_name, error_message
        )

        # Check if done
        done = self._check_done(event.metadata)

        observation = EnvObservation(
            image_path=image_path,
            text_state=text_state,
            reward=reward,
            done=done,
            metadata=event.metadata,
        )

        return observation, error_message

    def _resolve_move_magnitude(
        self, magnitude: Optional[float] = None, granularity: Optional[str] = None
    ) -> float:
        """Resolve move magnitude from explicit value or granularity (Small/Medium/Large).
        Small=0.25m, Medium=0.5m, Large=1m. Default when omitted: Small (0.25m).
        """
        if magnitude is not None and isinstance(magnitude, (int, float)):
            return float(magnitude)
        if granularity:
            g = str(granularity).strip().lower()
            if g == "small":
                return self.move_small_magnitude
            if g == "medium":
                return self.move_medium_magnitude
            if g == "large":
                return self.move_large_magnitude
        return self.move_small_magnitude

    def _convert_navigation_action(
        self, action_name: str, magnitude: Optional[float] = None, degrees: Optional[float] = None
    ) -> dict:
        """Convert navigation action

        Supported navigation actions:
        - Translation: MoveAhead, MoveBack, MoveLeft, MoveRight (use magnitude or default Small)
        - Rotation: RotateLeft, RotateRight (use degrees or default from config)
        - View: LookUp, LookDown (use degrees or default 30)
        - Posture: Crouch, Stand
        """
        move_mag = magnitude if magnitude is not None else self.move_small_magnitude
        rot_deg = float(degrees) if degrees is not None else self.rotate_degrees
        look_deg = float(degrees) if degrees is not None else 30
        action_map = {
            "MoveAhead": {
                "action": "MoveAhead",
                "moveMagnitude": move_mag,
            },
            "MoveBack": {
                "action": "MoveBack",
                "moveMagnitude": move_mag,
            },
            "MoveLeft": {
                "action": "MoveLeft",
                "moveMagnitude": move_mag,
            },
            "MoveRight": {
                "action": "MoveRight",
                "moveMagnitude": move_mag,
            },
            "RotateLeft": {"action": "RotateLeft", "degrees": rot_deg},
            "RotateRight": {"action": "RotateRight", "degrees": rot_deg},
            "LookUp": {"action": "LookUp", "degrees": look_deg},
            "LookDown": {"action": "LookDown", "degrees": look_deg},
            "Crouch": {"action": "Crouch"},
            "Stand": {"action": "Stand"},
        }
        return action_map.get(action_name, {"action": "Pass"})

    def _convert_interaction_action(
        self,
        action_name: str,
        object_type: Optional[str],
        scene_metadata: Optional[dict] = None,
    ) -> tuple[Optional[dict], Optional[str]]:
        """Convert interaction action

        For PickupObject: Supports semantic searching based on SEMANTIC_OBJECT_MAPPING.
        When model outputs PickupObject(Apple), system will search for ["Apple", "AppleSliced"]
        and pick up the nearest visible object.

        Args:
            scene_metadata: Prefer this dict (acting agent's view) over ``last_event`` in multi-agent runs.

        Returns:
            (thor_action, error_message) tuple
        """
        meta = scene_metadata
        if meta is None:
            if self.controller.last_event is None:
                return None, "No environment observation available to resolve object"
            meta = self.controller.last_event.metadata

        # DropHandObject doesn't need object
        if action_name == "DropHandObject":
            # Check if hand has object
            inventory = meta.get("inventoryObjects", [])
            if not inventory:
                return None, "Hand is empty, cannot drop"
            return {"action": "DropHandObject", "forceAction": False}, None

        # Other interaction actions need to find target object
        if not object_type:
            return None, f"{action_name} requires object type specification"

        # ========================================================================
        # STEP 2:      -    PickupObject，      
        # ========================================================================
        if action_name == "PickupObject":
            #                 
            candidate_types = SEMANTIC_OBJECT_MAPPING.get(object_type, [object_type])

            #         
            all_objects = meta["objects"]

            #                
            best_obj = None
            min_dist = float("inf")

            for obj in all_objects:
                if obj["objectType"] in candidate_types and obj["visible"]:
                    if obj["distance"] < min_dist:
                        min_dist = obj["distance"]
                        best_obj = obj

            if best_obj:
                #         （    Apple      AppleSliced）
                object_id = best_obj["objectId"]
                actual_type = best_obj["objectType"]

                #               ，        
                if actual_type != object_type:
                    print(
                        f"📦 Semantic search: PickupObject({object_type}) -> actually picking {actual_type}"
                    )

                thor_action = {
                    "action": "PickupObject",
                    "objectId": object_id,
                    "forceAction": False,
                }
                return thor_action, None
            else:
                #          
                invisible_objects = [
                    obj for obj in all_objects if obj["objectType"] in candidate_types
                ]

                if invisible_objects:
                    nearest_obj = min(
                        invisible_objects, key=lambda obj: obj["distance"]
                    )
                    distance = nearest_obj.get("distance", 0)
                    return (
                        None,
                        f"{object_type} is not in view, need to approach or adjust view (distance: {distance:.1f}m)",
                    )
                else:
                    return (
                        None,
                        f"{object_type} (or its transformed variants) does not exist in scene",
                    )

        # ========================================================================
        #       ：        
        # ========================================================================
        # Find all objects of this type (including invisible ones)
        # Support both exact match and prefix match (e.g., "Blinds" matches "Blinds|1|2")
        all_objects = [
            obj
            for obj in meta["objects"]
            if obj["objectType"] == object_type
            or obj["objectType"].startswith(object_type + "|")
        ]

        if not all_objects:
            return None, f"{object_type} does not exist in scene"

        # Find visible target objects
        visible_objects = [obj for obj in all_objects if obj["visible"]]

        if not visible_objects:
            # Find nearest invisible object, provide distance information
            nearest_obj = min(all_objects, key=lambda obj: obj["distance"])
            distance = nearest_obj.get("distance", 0)
            return (
                None,
                f"{object_type} is not in view, need to approach or adjust view (distance: {distance:.1f}m)",
            )

        # Select nearest visible object
        target_obj = min(visible_objects, key=lambda obj: obj["distance"])

        # Note: AI2-THOR internally validates reachability. Let it determine success/failure
        # rather than pre-filtering based on distance estimates

        object_id = target_obj["objectId"]

        # Map action name to AI2-THOR action
        action_map = {
            # Pickup and place
            "PickupObject": "PickupObject",
            "PutObject": "PutObject",
            "ThrowObject": "ThrowObject",
            # State changes
            "OpenObject": "OpenObject",
            "CloseObject": "CloseObject",
            "ToggleObjectOn": "ToggleObjectOn",
            "ToggleObjectOff": "ToggleObjectOff",
            "SliceObject": "SliceObject",
            "BreakObject": "BreakObject",
            "CookObject": "CookObject",
            "DirtyObject": "DirtyObject",
            "CleanObject": "CleanObject",
            "FillObjectWithLiquid": "FillObjectWithLiquid",
            "EmptyLiquidFromObject": "EmptyLiquidFromObject",
            "UseUpObject": "UseUpObject",
            # Push and pull
            "PushObject": "PushObject",
            "PullObject": "PullObject",
            "DirectionalPush": "DirectionalPush",
        }

        thor_action_name = action_map.get(action_name)
        if not thor_action_name:
            return None, f"Unknown interaction action: {action_name}"

        # Special handling: some actions need extra parameters
        thor_action = {
            "action": thor_action_name,
            "objectId": object_id,
            "forceAction": False,
        }

        # FillObjectWithLiquid needs to specify liquid type (default is water)
        if action_name == "FillObjectWithLiquid":
            thor_action["fillLiquid"] = "water"  # Optional: coffee, wine, water

        # ThrowObject needs throw force (default 150 Newtons)
        if action_name == "ThrowObject":
            thor_action["moveMagnitude"] = 150.0

        # PushObject/PullObject needs push force
        if action_name in ["PushObject", "PullObject"]:
            thor_action["moveMagnitude"] = 100.0

        return thor_action, None

    def _translate_error_message(
        self, raw_error: str, action_name: str, object_type: Optional[str]
    ) -> str:
        """Translate AI2-THOR error messages to user-friendly English

        Args:
            raw_error: AI2-THOR raw error message
            action_name: Action name
            object_type: Object type (optional)

        Returns:
            English error message
        """
        error_lower = raw_error.lower()
        obj = object_type or "object"

        # Hand state errors
        if "hand has something" in error_lower or "already holding" in error_lower:
            return "Hand already has an object, cannot pick up new object (please use DropHandObject first to drop the object in hand)"
        elif "hand is empty" in error_lower:
            return "Hand is empty, nothing to drop or place"

        # Distance and visibility errors
        elif "not in range" in error_lower or "too far" in error_lower:
            return f"Too far away, cannot interact with {obj} (suggest approaching before trying again)"
        elif "not visible" in error_lower:
            return f"{obj} is not within interaction range (suggest approaching before trying again)"
        elif "out of reach" in error_lower:
            return f"Cannot reach {obj} (too far away or blocked)"

        # Interaction capability errors
        elif "not interactable" in error_lower:
            return (
                f"{obj} cannot be interacted with (may be a static background object)"
            )
        elif "not pickupable" in error_lower or "can't pickup" in error_lower:
            return f"{obj} cannot be picked up (may be too large or fixed in scene)"
        elif "not receptacle" in error_lower:
            return f"{obj} is not a container, cannot place items"

        # State errors
        elif "not openable" in error_lower or "can't open" in error_lower:
            return f"{obj} cannot be opened (may not have door or lid)"
        elif "already open" in error_lower:
            return f"{obj} is already open"
        elif "already closed" in error_lower:
            return f"{obj} is already closed"
        elif "not toggleable" in error_lower:
            return f"{obj} cannot be toggled (not an appliance or light)"
        elif "already on" in error_lower or "already powered on" in error_lower:
            return f"{obj} is already on"
        elif "already off" in error_lower or "already powered off" in error_lower:
            return f"{obj} is already off"

        # Object state limitations
        elif "not sliceable" in error_lower:
            return f"{obj} cannot be sliced (not food)"
        elif "already sliced" in error_lower:
            return f"{obj} has already been sliced"
        elif "not breakable" in error_lower:
            return f"{obj} cannot be broken"
        elif "not cookable" in error_lower:
            return f"{obj} cannot be cooked"
        elif "not cleanable" in error_lower:
            return f"{obj} cannot be cleaned"
        elif "not fillable" in error_lower:
            return f"{obj} cannot be filled with liquid (not a container)"

        # Object does not exist
        elif "no object" in error_lower or "object not found" in error_lower:
            return f"Cannot find {obj} type object in scene"
        elif "no such object" in error_lower:
            return f"Specified {obj} does not exist"

        # Navigation and movement errors
        elif "path blocked" in error_lower or "collision" in error_lower:
            return "Path is blocked, cannot move (obstacle ahead)"
        elif "can't move" in error_lower or "cannot move" in error_lower:
            return "Cannot move to target position (space limited)"

        # Other common errors
        elif "invalid action" in error_lower:
            return f"Invalid action: {action_name}"
        elif "failed" in error_lower:
            return f"Failed to execute {action_name}: {raw_error}"

        # Unknown error, return original message
        else:
            return f"[Original error] {raw_error}"

    def _compute_reward_from_metadata(
        self, metadata: dict, action_name: str, error_message: Optional[str]
    ) -> float:
        """Calculate reward based on metadata and action result"""
        # Base failure penalty
        if error_message:
            return -0.05

        # Give small reward on success
        if metadata["lastActionSuccess"]:
            return 0.1

        return -0.05

    def get_action_sequence(self) -> str:
        """Get string representation of action sequence"""
        return "->".join(self.action_sequence)

    def _convert_action(self, action: EnvAction) -> list[dict]:
        """Convert EnvAction to AI2-THOR action sequence

        **Dual-mode support (optimized)**:
        1. Priority mode: If action.thor_actions is non-empty, use low-level actions (automatically handle objectId)
        2. Fallback mode: If thor_actions is empty, use traditional move/turn/interact abstract action conversion

        **Key improvements**:
        - Automatically handle all objectId parameters, VLM doesn't need to specify specific objects
        - Automatically select appropriate interaction objects based on visible objects and task goals
        - Execute only one action at a time (no longer support action sequences)

        Args:
            action: Environment action

        Returns:
            AI2-THOR action list (usually only 1 element)
        """
        # Priority mode: If VLM directly provides low-level actions, process and use them
        if action.thor_actions and len(action.thor_actions) > 0:
            # Only take first action (enforce single action)
            ta = action.thor_actions[0]
            print(f"  ✓ Using low-level action: {ta.name}")

            # Automatically handle interaction actions that need objectId
            object_actions = [
                "OpenObject",
                "CloseObject",
                "ToggleObjectOn",
                "ToggleObjectOff",
                "PickupObject",
                "PutObject",
                "SliceObject",
                "BreakObject",
                "CookObject",
                "DirtyObject",
                "CleanObject",
                "FillObjectWithLiquid",
                "EmptyLiquidFromObject",
                "UseUpObject",
            ]

            if ta.name in object_actions:
                # Ignore objectId provided by VLM, automatically select appropriate object
                target_obj = self._find_interaction_target(ta.name)
                if target_obj:
                    thor_action = {
                        "action": ta.name,
                        "objectId": target_obj["objectId"],
                    }

                    # Special handling: FillObjectWithLiquid needs additional fillLiquid parameter
                    if ta.name == "FillObjectWithLiquid":
                        # Get fillLiquid from args, default to water
                        fill_liquid = ta.args.get("fillLiquid", "water")
                        thor_action["fillLiquid"] = fill_liquid
                        print(
                            f"    → Auto-selected interaction object: {target_obj['objectType']} (fill: {fill_liquid}, distance: {target_obj['distance']:.2f}m)"
                        )
                    else:
                        print(
                            f"    → Auto-selected interaction object: {target_obj['objectType']} (distance: {target_obj['distance']:.2f}m)"
                        )

                    return [thor_action]
                else:
                    print(
                        f"    ⚠️  No suitable interaction object found, action cancelled"
                    )
                    return [{"action": "Pass"}]
            # Special handling: DropHandObject and ThrowObject don't need objectId
            elif ta.name in ["DropHandObject", "ThrowObject"]:
                thor_action = {"action": ta.name}
                # ThrowObject needs moveMagnitude parameter
                if ta.name == "ThrowObject":
                    throw_force = ta.args.get("moveMagnitude", 150.0)
                    thor_action["moveMagnitude"] = throw_force
                return [thor_action]
            else:
                # Non-interaction actions (move/rotate), resolve magnitude for move and use
                thor_action = {"action": ta.name, **ta.args}
                if ta.name in ("MoveAhead", "MoveBack", "MoveLeft", "MoveRight"):
                    thor_action["moveMagnitude"] = self._resolve_move_magnitude(
                        magnitude=ta.args.get("moveMagnitude"),
                        granularity=ta.args.get("granularity"),
                    )
                return [thor_action]

        # Fallback mode: Use traditional abstract action conversion logic
        print(f"  ℹ️  Using traditional action conversion logic (move/turn/interact)")
        return self._convert_from_move_turn_interact(action)

    def _find_interaction_target(self, action_name: str) -> Optional[dict]:
        """Automatically find suitable interaction target object

        Priority:
        1. Visible target type objects (target_object_types)
        2. Visible general interactable objects (openable/toggleable/pickupable)

        Selection strategy:
        - Nearest object by distance
        - State matches action requirements (e.g., OpenObject prefers unopened objects)

        Args:
            action_name: Action name (e.g., OpenObject, PickupObject)

        Returns:
            Selected object metadata, returns None if not found
        """
        event = self.controller.last_event

        # Strategy 1: Prioritize finding target type objects
        target_candidates = []
        if self.target_object_types:
            target_candidates = [
                obj
                for obj in event.metadata["objects"]
                if obj["visible"] and obj["objectType"] in self.target_object_types
            ]

        # Strategy 2: If no target objects, find general interactable objects
        if not target_candidates:
            # Filter candidate objects based on different action types
            if action_name in ["OpenObject", "CloseObject"]:
                target_candidates = [
                    obj
                    for obj in event.metadata["objects"]
                    if obj["visible"] and obj.get("openable", False)
                ]
            elif action_name in ["ToggleObjectOn", "ToggleObjectOff"]:
                target_candidates = [
                    obj
                    for obj in event.metadata["objects"]
                    if obj["visible"] and obj.get("toggleable", False)
                ]
            elif action_name == "PickupObject":
                target_candidates = [
                    obj
                    for obj in event.metadata["objects"]
                    if obj["visible"] and obj.get("pickupable", False)
                ]
            elif action_name == "PutObject":
                # Need to find receptacle
                target_candidates = [
                    obj
                    for obj in event.metadata["objects"]
                    if obj["visible"] and obj.get("receptacle", False)
                ]
            elif action_name == "SliceObject":
                target_candidates = [
                    obj
                    for obj in event.metadata["objects"]
                    if obj["visible"] and obj.get("sliceable", False)
                ]
            elif action_name == "BreakObject":
                target_candidates = [
                    obj
                    for obj in event.metadata["objects"]
                    if obj["visible"] and obj.get("breakable", False)
                ]
            elif action_name == "CookObject":
                target_candidates = [
                    obj
                    for obj in event.metadata["objects"]
                    if obj["visible"] and obj.get("cookable", False)
                ]
            elif action_name in ["DirtyObject", "CleanObject"]:
                target_candidates = [
                    obj
                    for obj in event.metadata["objects"]
                    if obj["visible"] and obj.get("dirtyable", False)
                ]
            elif action_name in ["FillObjectWithLiquid", "EmptyLiquidFromObject"]:
                target_candidates = [
                    obj
                    for obj in event.metadata["objects"]
                    if obj["visible"] and obj.get("canFillWithLiquid", False)
                ]
            elif action_name == "UseUpObject":
                target_candidates = [
                    obj
                    for obj in event.metadata["objects"]
                    if obj["visible"] and obj.get("canBeUsedUp", False)
                ]

        if not target_candidates:
            return None

        # Filter objects with appropriate state based on action type
        if action_name == "OpenObject":
            # Prefer unopened objects
            closed_objects = [
                obj for obj in target_candidates if not obj.get("isOpen", False)
            ]
            target_candidates = closed_objects if closed_objects else target_candidates
        elif action_name == "CloseObject":
            # Prefer opened objects
            open_objects = [
                obj for obj in target_candidates if obj.get("isOpen", False)
            ]
            target_candidates = open_objects if open_objects else target_candidates
        elif action_name == "ToggleObjectOn":
            # Prefer untoggled objects
            toggled_off = [
                obj for obj in target_candidates if not obj.get("isToggled", False)
            ]
            target_candidates = toggled_off if toggled_off else target_candidates
        elif action_name == "ToggleObjectOff":
            # Prefer toggled objects
            toggled_on = [
                obj for obj in target_candidates if obj.get("isToggled", False)
            ]
            target_candidates = toggled_on if toggled_on else target_candidates
        elif action_name == "SliceObject":
            # Prefer unsliced objects
            unsliced = [
                obj for obj in target_candidates if not obj.get("isSliced", False)
            ]
            target_candidates = unsliced if unsliced else target_candidates
        elif action_name == "BreakObject":
            # Prefer unbroken objects
            unbroken = [
                obj for obj in target_candidates if not obj.get("isBroken", False)
            ]
            target_candidates = unbroken if unbroken else target_candidates
        elif action_name == "CookObject":
            # Prefer uncooked objects
            uncooked = [
                obj for obj in target_candidates if not obj.get("isCooked", False)
            ]
            target_candidates = uncooked if uncooked else target_candidates
        elif action_name == "CleanObject":
            # Prefer dirty objects
            dirty = [obj for obj in target_candidates if obj.get("isDirty", False)]
            target_candidates = dirty if dirty else target_candidates
        elif action_name == "DirtyObject":
            # Prefer clean objects
            clean = [obj for obj in target_candidates if not obj.get("isDirty", False)]
            target_candidates = clean if clean else target_candidates
        elif action_name == "FillObjectWithLiquid":
            # Prefer empty containers
            empty = [
                obj
                for obj in target_candidates
                if not obj.get("isFilledWithLiquid", False)
            ]
            target_candidates = empty if empty else target_candidates
        elif action_name == "EmptyLiquidFromObject":
            # Prefer filled containers
            filled = [
                obj for obj in target_candidates if obj.get("isFilledWithLiquid", False)
            ]
            target_candidates = filled if filled else target_candidates
        elif action_name == "UseUpObject":
            # Prefer unused objects
            not_used_up = [
                obj for obj in target_candidates if not obj.get("isUsedUp", False)
            ]
            target_candidates = not_used_up if not_used_up else target_candidates

        # Select nearest object
        return min(target_candidates, key=lambda obj: obj["distance"])

    def _convert_from_move_turn_interact(self, action: EnvAction) -> list[dict]:
        """Traditional abstract action conversion logic (backward compatible)

        Uses "prioritize target object" strategy:
        1. If target object types are configured, prioritize finding target objects in visible objects
        2. Select appropriate interaction action based on task semantics (open/close/toggle)
        3. If no target objects visible, fall back to general interaction logic

        **Uses action parameters from YAML config**:
        - rotate_degrees: Rotation angle
        - move_*_magnitude: Movement distance

        Args:
            action: Environment action

        Returns:
            AI2-THOR action list
        """
        thor_actions = []

        # Handle rotation (use configured rotate_degrees)
        if action.turn is not None:
            degrees = self.rotate_degrees
            if action.turn > 0:
                # Right turn
                thor_actions.append({"action": "RotateRight", "degrees": degrees})
            else:
                # Left turn
                thor_actions.append({"action": "RotateLeft", "degrees": degrees})

        # Handle movement (use Small as default for abstract move/turn/interact)
        if action.move:
            move_map = {
                "forward": ("MoveAhead", self.move_small_magnitude),
                "back": ("MoveBack", self.move_small_magnitude),
                "left": ("MoveLeft", self.move_small_magnitude),
                "right": ("MoveRight", self.move_small_magnitude),
            }

            if action.move in move_map:
                action_name, magnitude = move_map[action.move]
                thor_actions.append({"action": action_name, "moveMagnitude": magnitude})

        # Handle interaction (generalized version, based on task goals)
        if action.interact:
            event = self.controller.last_event
            target_obj = None

            # Strategy 1: If target object types configured, prioritize finding target objects
            if self.target_object_types:
                target_candidates = [
                    obj
                    for obj in event.metadata["objects"]
                    if obj["visible"] and obj["objectType"] in self.target_object_types
                ]

                if target_candidates:
                    # Select nearest target object
                    target_obj = min(target_candidates, key=lambda obj: obj["distance"])
                    print(
                        f"  🎯 Target object found: {target_obj['objectType']} (distance: {target_obj['distance']:.2f}m)"
                    )

            # Strategy 2: If no target object found, fall back to general interactable objects
            if target_obj is None:
                # Find all interactable visible objects (openable or toggleable)
                interactable_objects = [
                    obj
                    for obj in event.metadata["objects"]
                    if obj["visible"]
                    and (obj.get("openable", False) or obj.get("toggleable", False))
                ]

                if interactable_objects:
                    target_obj = min(
                        interactable_objects, key=lambda obj: obj["distance"]
                    )
                    print(
                        f"  ℹ️  Fallback to general interaction object: {target_obj['objectType']} (distance: {target_obj['distance']:.2f}m)"
                    )

            # Determine action based on found object and task semantics
            if target_obj:
                interaction_action = self._determine_interaction_action(target_obj)
                if interaction_action:
                    thor_actions.append(interaction_action)
                else:
                    print(
                        f"  ⚠️  Cannot determine interaction action for {target_obj['objectType']}"
                    )
                    thor_actions.append({"action": "Pass"})
            else:
                # No interactable objects at all
                print("  ⚠️  No interactable objects in current view")
                thor_actions.append({"action": "Pass"})

        return thor_actions if thor_actions else [{"action": "Pass"}]

    def _determine_interaction_action(self, obj: dict) -> Optional[dict]:
        """Determine interaction action based on object properties and task semantics

        Args:
            obj: Object metadata

        Returns:
            AI2-THOR action dictionary, returns None if cannot determine
        """
        object_id = obj["objectId"]

        # If object is openable/closable (openable)
        if obj.get("openable", False):
            # Infer task intent: if success_predicate expects isOpen=True, open; otherwise close
            current_open = obj.get("isOpen", False)

            # Simple heuristic: if already open and predicate expects open, don't operate; otherwise try to open
            # Stricter approach would be to directly call success_predicate, but here use conservative strategy
            if self.success_predicate:
                # Test if "open" state satisfies
                test_open_state = {**obj, "isOpen": True}
                should_open = self.success_predicate(test_open_state)

                if should_open and not current_open:
                    return {
                        "action": "OpenObject",
                        "objectId": object_id,
                        "forceAction": False,
                    }
                elif not should_open and current_open:
                    return {
                        "action": "CloseObject",
                        "objectId": object_id,
                        "forceAction": False,
                    }
                elif should_open and current_open:
                    # Already satisfies condition, but still interacting suggests may want to confirm state
                    return {
                        "action": "OpenObject",
                        "objectId": object_id,
                        "forceAction": False,
                    }
                else:
                    return {
                        "action": "CloseObject",
                        "objectId": object_id,
                        "forceAction": False,
                    }
            else:
                # When no predicate, default to open
                return {
                    "action": "OpenObject",
                    "objectId": object_id,
                    "forceAction": False,
                }

        # If object is toggleable
        if obj.get("toggleable", False):
            current_toggled = obj.get("isToggled", False)

            if self.success_predicate:
                test_toggled_state = {**obj, "isToggled": True}
                should_toggle_on = self.success_predicate(test_toggled_state)

                if should_toggle_on and not current_toggled:
                    return {
                        "action": "ToggleObjectOn",
                        "objectId": object_id,
                        "forceAction": False,
                    }
                elif not should_toggle_on and current_toggled:
                    return {
                        "action": "ToggleObjectOff",
                        "objectId": object_id,
                        "forceAction": False,
                    }
            else:
                # When no predicate, default to on
                return {
                    "action": "ToggleObjectOn",
                    "objectId": object_id,
                    "forceAction": False,
                }

        # If object is pickupable
        if obj.get("pickupable", False):
            return {
                "action": "PickupObject",
                "objectId": object_id,
                "forceAction": False,
            }

        # If object is sliceable
        if obj.get("sliceable", False) and not obj.get("isSliced", False):
            return {
                "action": "SliceObject",
                "objectId": object_id,
                "forceAction": False,
            }

        # If object is cookable
        if obj.get("cookable", False) and not obj.get("isCooked", False):
            return {"action": "CookObject", "objectId": object_id, "forceAction": False}

        # If object is breakable
        if obj.get("breakable", False) and not obj.get("isBroken", False):
            return {
                "action": "BreakObject",
                "objectId": object_id,
                "forceAction": False,
            }

        # If object is dirtyable
        if obj.get("dirtyable", False):
            if obj.get("isDirty", False):
                return {
                    "action": "CleanObject",
                    "objectId": object_id,
                    "forceAction": False,
                }
            else:
                return {
                    "action": "DirtyObject",
                    "objectId": object_id,
                    "forceAction": False,
                }

        # If object can be filled with liquid (canFillWithLiquid)
        if obj.get("canFillWithLiquid", False):
            if obj.get("isFilledWithLiquid", False):
                return {
                    "action": "EmptyLiquidFromObject",
                    "objectId": object_id,
                    "forceAction": False,
                }
            else:
                return {
                    "action": "FillObjectWithLiquid",
                    "objectId": object_id,
                    "fillLiquid": "water",
                    "forceAction": False,
                }

        # If object can be used up (canBeUsedUp)
        if obj.get("canBeUsedUp", False) and not obj.get("isUsedUp", False):
            return {
                "action": "UseUpObject",
                "objectId": object_id,
                "forceAction": False,
            }

        # Cannot determine action
        return None

    @staticmethod
    def _thor_agent_image_subdir(thor_agent_id: int) -> str:
        """Per-body folder under output_dir: agent1, agent2, ... (matches dual_agent exports)."""
        return f"agent{int(thor_agent_id) + 1}"

    def _save_frame(
        self,
        frame: np.ndarray,
        prefix: str = "frame",
        thor_agent_id: Optional[int] = None,
    ) -> str:
        """Save image frame (same naming as single-agent: ``{prefix}_{timestamp}.png``).

        When ``agent_count > 1``, saves under ``agent{k}/`` for the embodied agent so the
        layout matches single-agent flat files, but namespaced per body—no duplicate root copies.

        Args:
            frame: RGB image array
            prefix: Filename prefix (e.g. ``reset``, ``step_3``)
            thor_agent_id: If multi-agent, which body's folder to write (0 -> agent1, ...).

        Returns:
            Saved image path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        if self.agent_count > 1 and thor_agent_id is not None:
            sub = self._thor_agent_image_subdir(thor_agent_id)
            out_dir = os.path.join(self.output_dir, sub)
            os.makedirs(out_dir, exist_ok=True)
            filepath = os.path.join(out_dir, filename)
        else:
            filepath = os.path.join(self.output_dir, filename)

        image = Image.fromarray(frame)
        image.save(filepath)

        return filepath

    def _generate_text_state(self, metadata: dict) -> str:
        """Generate text state description (supports first_person / omniscient modes)"""
        if self.text_state_mode == "omniscient":
            return self._generate_omniscient_text_state(metadata)
        return self._generate_first_person_text_state(metadata)

    def _generate_first_person_text_state(self, metadata: dict) -> str:
        """Strict first-person text: only provides action feedback and task hints"""
        lines = [
            " First-Person Mode Please mainly rely on images to judge the environment, no additional global state provided.",
            f"Task description: {self.task_description}",
            f"Current step: {self.step_counter}",
            f"Last action success: {'Yes' if metadata['lastActionSuccess'] else 'No'}",
        ]
        if not metadata["lastActionSuccess"]:
            error_msg = metadata.get("errorMessage", "Unknown")
            lines.append(f"Error message: {error_msg}")
            if (
                "collision" in error_msg.lower()
                or "block" in error_msg.lower()
                or "obstacle" in error_msg.lower()
            ):
                lines.append(
                    "Hint: There may be an obstacle ahead, try interacting or changing direction to bypass."
                )
            elif "out of reach" in error_msg.lower() or "too far" in error_msg.lower():
                lines.append(
                    "Hint: Target may be too far, please approach first before interacting."
                )
            elif "not visible" in error_msg.lower():
                lines.append(
                    "Hint: Target not found in current view, please rotate or move to search again."
                )
        return "\n".join(lines)

    def _generate_omniscient_text_state(self, metadata: dict) -> str:
        """Original omniscient view text state"""
        agent = metadata["agent"]
        position = agent["position"]
        rotation = agent["rotation"]

        visible_objects = [
            obj["objectType"] for obj in metadata["objects"] if obj["visible"]
        ]
        visible_objects = sorted(set(visible_objects))

        if len(visible_objects) > 10:
            visible_objects_str = (
                ", ".join(visible_objects[:10])
                + f", ... (total {len(visible_objects)} objects)"
            )
        else:
            visible_objects_str = (
                ", ".join(visible_objects) if visible_objects else "None"
            )

        text_state = f"""Scene: {self.scene}
Agent position: ({position["x"]:.2f}, {position["y"]:.2f}, {position["z"]:.2f})
Agent heading: {rotation["y"]:.1f}°
Visible objects: {visible_objects_str}"""

        if self.target_object_types:
            task_summary = self._generate_task_status_summary(metadata)
            text_state += f"\n\n Task Status \n{task_summary}"

        text_state += f"\n\nLast action success: {'Yes' if metadata['lastActionSuccess'] else 'No'}"

        if not metadata["lastActionSuccess"]:
            error_msg = metadata.get("errorMessage", "Unknown")
            text_state += f"\nError message: {error_msg}"

            if (
                "collision" in error_msg.lower()
                or "block" in error_msg.lower()
                or "obstacle" in error_msg.lower()
            ):
                text_state += "\n⚠️  Movement blocked! Obstacle ahead. Suggestions:"
                text_state += "\n   1. If the object ahead is the target, try interacting with it (interact=true)"
                text_state += "\n   2. Otherwise rotate view to change direction, or back up/side step to bypass"
                text_state += "\n   3. Don't continuously move in the same direction"
            elif "out of reach" in error_msg.lower() or "too far" in error_msg.lower():
                text_state += "\n⚠️  Object too far! Please approach target object first before interacting"
            elif "not visible" in error_msg.lower():
                text_state += (
                    "\n⚠️  Target not visible! Please rotate view or move to find target"
                )

        return text_state

    def _generate_task_status_summary(self, metadata: dict) -> str:
        """Generate status summary of task-related objects

        Provides key information about task goals to help Agent understand current task progress

        Args:
            metadata: AI2-THOR metadata

        Returns:
            Task status summary text
        """
        summary_lines = []

        # Add task goal description
        summary_lines.append(f"Task goal: {self.target_description}")

        # Count target objects
        all_target_objects = [
            obj
            for obj in metadata["objects"]
            if obj["objectType"] in self.target_object_types
        ]

        visible_target_objects = [obj for obj in all_target_objects if obj["visible"]]

        # Count objects that satisfy success conditions
        success_objects = []
        if self.success_predicate:
            success_objects = [
                obj for obj in all_target_objects if self.success_predicate(obj)
            ]

        # Generate statistics
        summary_lines.append(
            f"Target object types: {', '.join(self.target_object_types)}"
        )
        summary_lines.append(
            f"Total target objects in scene: {len(all_target_objects)}"
        )
        summary_lines.append(
            f"Currently visible target objects: {len(visible_target_objects)}"
        )
        summary_lines.append(
            f"Objects satisfying success conditions: {len(success_objects)}"
        )

        # If there are visible target objects, list detailed information
        if visible_target_objects:
            summary_lines.append("\nVisible target object details:")
            for obj in visible_target_objects[:3]:  # Show at most 3
                obj_status = []
                if obj.get("openable", False):
                    obj_status.append(
                        f"Openable, current: {'Open' if obj.get('isOpen', False) else 'Closed'}"
                    )
                if obj.get("toggleable", False):
                    obj_status.append(
                        f"Toggleable, current: {'On' if obj.get('isToggled', False) else 'Off'}"
                    )
                if obj.get("pickupable", False):
                    obj_status.append("Pickupable")

                status_str = ", ".join(obj_status) if obj_status else "No special state"
                summary_lines.append(
                    f"  - {obj['objectType']} (distance: {obj['distance']:.2f}m, {status_str})"
                )
        else:
            summary_lines.append(
                "\n⚠️  No target objects in current view, suggest rotating view or moving to explore"
            )

        # If objects already satisfy success conditions, give hint
        if success_objects:
            summary_lines.append(
                f"\n✓ {len(success_objects)} objects already satisfy success conditions"
            )

        return "\n".join(summary_lines)

    def _compute_reward(self, action: EnvAction, metadata: dict) -> float:
        """Calculate reward (generalized version, supports different tasks)

        **Uses reward parameters from YAML config**:
        - step_success_bonus: Small reward when action succeeds
        - step_failure_penalty: Penalty when action fails
        - success_reward: Large reward when task completes

        Args:
            action: Executed action
            metadata: Environment metadata

        Returns:
            Reward value
        """
        reward = 0.0

        # Base reward: whether action succeeded (from YAML config)
        if metadata["lastActionSuccess"]:
            reward += self.step_success_bonus
        else:
            reward += self.step_failure_penalty  # Note: penalty is already negative

        # Task completion large reward (from YAML config)
        if self._check_done(metadata):
            reward += self.success_reward
            print(f"  🎉 Task completion reward: +{self.success_reward}")

        return reward

    def _check_done(self, metadata: dict) -> bool:
        """Check if task is complete (generalized version, based on task config)

        Determines task completion through target_object_types and success_predicate:
        - Iterate through all objects of target object types
        - Apply success_predicate to each object
        - Task completes if any object satisfies condition

        Args:
            metadata: Environment metadata

        Returns:
            Whether complete
        """
        # If task parameters not configured, return False
        if not self.target_object_types or not self.success_predicate:
            return False

        # Special handling: object_in_receptacle type
        if (
            self.success_condition
            and self.success_condition.get("type") == "object_in_receptacle"
        ):
            object_type = self.success_condition.get("object_type", "Apple")
            receptacle_type = self.success_condition.get("receptacle_type", "Plate")

            # Find objects of specified type
            # Support both exact match and prefix match (e.g., "Blinds" matches "Blinds|1|2")
            for obj in metadata["objects"]:
                obj_type = obj["objectType"]
                if obj_type == object_type or obj_type.startswith(object_type + "|"):
                    # Use predicate to check if in receptacle
                    if self.success_predicate(obj):
                        print(
                            f"  ✓ Task completion condition met: {object_type} placed in {receptacle_type}"
                        )
                        return True

            return False

        # Special handling: object_in_hand type
        if (
            self.success_condition
            and self.success_condition.get("type") == "object_in_hand"
        ):
            target_type = self.success_condition.get("object_type")
            inventory = metadata.get("inventoryObjects", [])
            # Support both exact match and prefix match (e.g., "Blinds" matches "Blinds|1|2")
            for item in inventory:
                item_type = item["objectType"]
                if item_type == target_type or item_type.startswith(target_type + "|"):
                    print(f"  ✓ Task completion condition met: {target_type} in hand")
                    return True
            return False

        # Standard handling: iterate through all objects in scene
        for obj in metadata["objects"]:
            # Check if is target object type
            if obj["objectType"] in self.target_object_types:
                # Apply success condition predicate
                if self.success_predicate(obj):
                    print(
                        f"  ✓ Task completion condition met: {obj['objectType']} meets success criteria"
                    )
                    return True

        return False

    def _format_action(self, action: EnvAction) -> str:
        """Format action for display

        Args:
            action: Environment action

        Returns:
            Formatted action string
        """
        parts = []
        if action.move:
            parts.append(f"move={action.move}")
        if action.turn is not None:
            direction = "right" if action.turn > 0 else "left"
            parts.append(f"{direction} {abs(action.turn)}°")
        if action.interact:
            parts.append("interact=True")

        return ", ".join(parts) if parts else "No action"

    def get_action_sequence(self) -> str:
        """Get string representation of action sequence

        Returns:
            Action sequence string, e.g.: "MoveAhead(0.25)->MoveAhead(0.25)->RotateRight(90)->OpenObject(Fridge)"
        """
        if not self.action_sequence:
            return "(No action records)"
        return "->".join(self.action_sequence)

    def close(self):
        """Close environment"""
        try:
            if hasattr(self, "controller"):
                self.controller.stop()
                print("✓ AI2-THOR environment closed")
        except Exception as e:
            print(f"⚠️  Error closing AI2-THOR environment: {e}")
