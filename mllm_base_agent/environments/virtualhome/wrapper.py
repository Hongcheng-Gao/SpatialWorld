"""
VirtualHome Environment Wrapper
Mirrors the AI2ThorEnvWrapper pattern to integrate VirtualHome into this project.

VirtualHome uses a graph-based state model:
  - Environment graph: nodes (objects w/ states/properties) + edges (relations)
  - Script-based execution: '<char0> [Action] <object_name> (object_id)'
  - Step execution:  comm.render_script([line], skip_animation=depends_on_action)
  - State query:     comm.environment_graph()
  - Image capture:   comm.camera_image(camera_ids)

Reference: http.
"""

import os
import io
import re
import json
import sys
import importlib.util
import collections
import collections.abc
from typing import Optional, Callable, List, Dict, Any, Tuple
from datetime import datetime

# Python 3.12+ removed collections.Iterable/Mapping;
# virtualhome internally still references them.
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping

try:
    import numpy as np
    from PIL import Image
except ImportError as e:
    raise ImportError(f"Required package missing: {e}. Run: pip install numpy Pillow")


VH_VISIBILITY_SOURCES: Tuple[str, str] = ("seg_inst", "api")
VH_STABLE_VISIBILITY_RETRIES = 4
VH_STABLE_VISIBILITY_DELAY_SEC = 0.35
VH_INTERACT_TIME_SCALE = 5.0


_VH_VISIBILITY_SOURCE_ALIASES = {
    "seg": "seg_inst",
    "seginst": "seg_inst",
    "seg_inst": "seg_inst",
    "segmentation": "seg_inst",
    "api": "api",
    "visible_objects": "api",
    "get_visible_objects": "api",
    "unity": "api",
    "fov": "fov",
    "geometry": "fov",
    "geometry_fallback": "fov",
}


def _parse_visibility_sources(raw: Any) -> List[str]:
    """Normalize configured visibility sources to an ordered unique list."""
    default = list(VH_VISIBILITY_SOURCES)
    if raw is None:
        return default
    if isinstance(raw, str):
        items = [part.strip() for part in raw.replace("+", ",").split(",")]
    elif isinstance(raw, (list, tuple, set)):
        items = [str(part).strip() for part in raw]
    else:
        return default

    sources: List[str] = []
    for item in items:
        key = item.lower().replace("-", "_").replace(" ", "_")
        normalized = _VH_VISIBILITY_SOURCE_ALIASES.get(key)
        if normalized and normalized not in sources:
            sources.append(normalized)
    return sources or default


def _ensure_installed_virtualhome_path() -> bool:
    """  pip     virtualhome        sys.path（    __init__.py    ）"""
    try:
        spec = importlib.util.find_spec("virtualhome")
    except Exception:
        return False
    if spec is None or not spec.submodule_search_locations:
        return False

    package_root = list(spec.submodule_search_locations)[0]
    simulation_dir = os.path.join(package_root, "simulation")
    if not os.path.isdir(simulation_dir):
        return False

    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    return True


UnityCommunication = None
UNITY_COMM_IMPORT_ERROR: Optional[Exception] = None
try:
    from simulation.unity_simulator.comm_unity import UnityCommunication
except ImportError as err:
    UNITY_COMM_IMPORT_ERROR = err

    if _ensure_installed_virtualhome_path():
        try:
            from simulation.unity_simulator.comm_unity import UnityCommunication

            UNITY_COMM_IMPORT_ERROR = None
        except ImportError as err_retry:
            UNITY_COMM_IMPORT_ERROR = err_retry

if UnityCommunication is None:
    try:
        from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication

        UNITY_COMM_IMPORT_ERROR = None
    except ImportError as err2:
        UNITY_COMM_IMPORT_ERROR = err2  # Will be checked at instantiation time

from core.llm.schemas import EnvObservation
from envs.base import BaseEnv
from mllm_base_agent.environments.virtualhome.backend_utils import resolve_backend_host, resolve_backend_port
from envs.virtualhome.utils import (
    ACTION_NAME_MAP,
    VH_NAVIGATION_WITH_OBJECT,
    VH_NAVIGATION_NO_OBJECT,
    VH_INTERACTION_ONE_OBJECT,
    VH_INTERACTION_TWO_OBJECTS,
    find_character_ids,
    find_object_id_in_graph,
    build_object_metadata,
    generate_text_state,
    format_vh_action_dict,
    coerce_init_rotation_quat,
    parse_vh_action_string,
    canonicalize_object_type_name,
)

# First-person camera presets (mirrors interact_virtualhome.py strategy)
_VH_FP_PITCH_CAMERAS = [
    (-60, "wasd_fp_u60"),
    (-30, "wasd_fp_u30"),
    (0, "wasd_fp"),
    (30, "wasd_fp_d30"),
    (60, "wasd_fp_d60"),
]
_VH_FP_DEFAULT_IDX = 2  # horizontal camera
_VH_FP_CAMERA_OFFSET = [0, 1.8, 0.15]


def _parse_cam_names(raw: Any) -> list:
    """Parse character_cameras() output that may be list or JSON string."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    return []


class VirtualHomeEnvWrapper(BaseEnv):
    """VirtualHome Environment Wrapper

    Wraps the VirtualHome Python API (UnityCommunication) to provide a unified
    environment interaction interface compatible with this project's agent framework.

    The wrapper converts action dictionaries (action_type / action_name / object_type)
    into VirtualHome script strings, executes them via render_script(), and returns
    EnvObservation objects with image paths, text states, and AI2THOR-compatible
    metadata dicts (so the existing evaluators in evaluators/base.py work unchanged).

    Configuration source priority (same as AI2ThorEnvWrapper):
      1. YAML config dict passed to __init__
      2. configure_task() runtime parameters
      3. Default values
    """

    def __init__(
        self,
        scene: int = 0,
        executable_path: Optional[str] = None,
        port: str = "8080",
        url: str = "127.0.0.1",
        x_display: Optional[str] = None,
        width: int = 300,
        height: int = 300,
        output_dir: str = "outputs",
        config: Optional[Dict[str, Any]] = None,
        num_agents: int = 1,
        char_resource: str = "Chars/Female1",
    ):
        """Initialize VirtualHome environment wrapper.

        Args:
            scene: VirtualHome scene index (0–49 correspond to different apartments).
            executable_path: Path to the VirtualHome Unity executable. Pass None when
                             connecting to an already-running instance.
            port: Port that the Unity process is listening on.
            url: Host URL of the Unity process.
            x_display: X11 display string (Linux, e.g. ':0'). None for headless.
            width: Image capture width in pixels.
            height: Image capture height in pixels.
            output_dir: Directory where frame images and results are saved.
            config: Complete configuration dictionary loaded from a YAML file.
            num_agents: Number of character agents to add to the scene.
            char_resource: Character asset to load (e.g. 'Chars/Female1').
        """
        if UnityCommunication is None:
            detail = (
                f" Original import error: {UNITY_COMM_IMPORT_ERROR!r}"
                if UNITY_COMM_IMPORT_ERROR is not None
                else ""
            )
            raise ImportError(
                "virtualhome package not installed. "
                "Install it with: pip install virtualhome  "
                "(or clone http.)" + detail
            )

        super().__init__(config)

        self.config = config or {}
        env_cfg = self.config.get("env", {})

        # Scene / connection settings
        self.scene = int(env_cfg.get("scene", scene))
        self.port = str(resolve_backend_port(self.config, port))
        self.url = resolve_backend_host(self.config, url)
        self.width = int(env_cfg.get("width", width))
        self.height = int(env_cfg.get("height", height))
        self.text_state_mode = env_cfg.get("text_state_mode", "first_person")
        if self.text_state_mode not in {"first_person", "omniscient"}:
            self.text_state_mode = "first_person"

        self.output_dir = output_dir
        self.num_agents = num_agents
        self.char_resource = char_resource
        self.task_description = ""
        self.require_visible_for_interaction = bool(
            env_cfg.get("require_visible_for_interaction", True)
        )
        self.require_close_for_interaction = True
        self.bound_instance_update_policy = str(
            env_cfg.get("bound_instance_update_policy", "latest")
        ).lower()
        if self.bound_instance_update_policy not in {"latest", "first"}:
            self.bound_instance_update_policy = "latest"

        # Task configuration
        task_cfg = self.config.get("task", {})
        self.target_object_types: List[str] = task_cfg.get("target_object_types", [])
        (
            self.success_condition,
            self.success_conditions,
            self.success_logic,
        ) = self._normalize_success_condition_config(
            task_cfg.get("success_condition"),
            task_cfg.get("success_conditions"),
            task_cfg.get("success_logic"),
        )
        self.success_predicate: Optional[Callable[[dict], bool]] = None
        self.success_predicates: List[Callable[[dict], bool]] = []
        self.success_evaluator = None
        self.target_description: str = task_cfg.get("target_description", "")

        # Reward configuration
        reward_cfg = self.config.get("reward", {})
        self.success_reward = float(reward_cfg.get("success_reward", 10.0))
        self.step_success_bonus = float(reward_cfg.get("step_success_bonus", 0.1))
        self.step_failure_penalty = float(reward_cfg.get("step_failure_penalty", -0.05))

        # Navigation: match AI2-THOR-style small/medium/large meters via repeated WalkForward
        actions_config = self.config.get("actions", {})
        self.move_small_magnitude = float(
            actions_config.get("move_small_magnitude", 0.25)
        )
        self.move_medium_magnitude = float(
            actions_config.get("move_medium_magnitude", 0.5)
        )
        self.move_large_magnitude = float(
            actions_config.get("move_large_magnitude", 1.0)
        )
        self.walk_forward_base_unit_meters = float(
            actions_config.get("walk_forward_base_unit_meters", 0.25)
        )
        # Each native TurnLeft/TurnRight in VH is one 30° step
        self.vh_turn_step_degrees = float(
            actions_config.get("vh_turn_step_degrees", 30.0)
        )
        self.turn_default_degrees = float(
            actions_config.get("turn_default_degrees", 90.0)
        )
        self.walk_yaw_restore_min_delta_degrees = float(
            actions_config.get("walk_yaw_restore_min_delta_degrees", 30.0)
        )
        self.grab_yaw_restore_min_delta_degrees = float(
            actions_config.get("grab_yaw_restore_min_delta_degrees", 20.0)
        )
        self.grab_exact_restore_tolerance_degrees = float(
            actions_config.get("grab_exact_restore_tolerance_degrees", 2.0)
        )
        self.grab_yaw_residual_warn_degrees = float(
            actions_config.get("grab_yaw_residual_warn_degrees", 8.0)
        )
        self.grab_yaw_target_tolerance_degrees = float(
            actions_config.get("grab_yaw_target_tolerance_degrees", 6.0)
        )
        self.grab_yaw_closed_loop_max_turns = int(
            actions_config.get("grab_yaw_closed_loop_max_turns", 14)
        )
        self.grab_reset_camera_pose = bool(
            actions_config.get("grab_reset_camera_pose", True)
        )
        self.strict_init_rotation = bool(
            actions_config.get("strict_init_rotation", False)
        )
        # Minimum displacement (m) after WalkForward to consider the char actually moved.
        # If actual movement < threshold the step is treated as stuck and returns an error.
        self.stuck_walk_position_threshold = float(
            actions_config.get("stuck_walk_position_threshold", 0.05)
        )
        # If character Y exceeds this after reset(), spawn is invalid (off-NavMesh / floating).
        self.spawn_max_valid_y = float(
            actions_config.get("spawn_max_valid_y", 2.0)
        )
        self.visible_seg_pixel_threshold = int(
            actions_config.get("visible_seg_pixel_threshold", 25)
        )
        self.visibility_sources = _parse_visibility_sources(
            actions_config.get("visibility_sources", ["seg_inst", "api"])
        )
        self.interaction_distance_meters = float(
            actions_config.get("interaction_distance_meters", 0.0)
        )
        self.collision_rollback_enabled = bool(
            actions_config.get("collision_rollback_enabled", True)
        )
        self.collision_bbox_shrink_meters = float(
            actions_config.get("collision_bbox_shrink_meters", 0.02)
        )
        self.atomic_walkforward_enabled = bool(
            actions_config.get("atomic_walkforward_enabled", True)
        )
        self.atomic_walkforward_tolerance_meters = float(
            actions_config.get("atomic_walkforward_tolerance_meters", 0.05)
        )

        # Internal state
        self._current_graph: Optional[Dict[str, Any]] = None
        self._char_ids: List[int] = []
        self._char_camera_ids: List[int] = []  # First-person camera IDs
        self._instance_color_to_id: Optional[Dict[Tuple[int, int, int], int]] = None
        self._last_visibility_source: str = "none"
        self._last_visibility_camera_id: Optional[int] = None
        self._seg_visibility_cache: Dict[
            Tuple[int, int, int, int, int], Tuple[set, Dict[int, int]]
        ] = {}
        self._bound_instances: Dict[str, int] = {}
        self._pitch_idx: int = _VH_FP_DEFAULT_IDX
        self._char_yaw_rad: Optional[float] = None
        self._skip_first_walk_yaw_restore = False
        self._last_char_position: Optional[List[float]] = None  # For stuck-walk detection
        self._scene_interactable_object_types: List[str] = []
        self._scene_all_object_types: List[str] = []
        os.makedirs(output_dir, exist_ok=True)

        # Connect to VirtualHome Unity simulator
        print(f"Connecting to VirtualHome (url={url}, port={port})...")
        try:
            self.comm = UnityCommunication(
                url=self.url,
                port=self.port,
                file_name=executable_path,
                x_display=x_display,
                #        exe      no_graphics；          False
                no_graphics=(executable_path is not None and x_display is None),
            )
            print("✓ VirtualHome connection established")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to VirtualHome: {exc}\n"
                f"Make sure the Unity executable is running (or pass executable_path)."
            ) from exc

        # Build success predicates/evaluator from config.
        if self.success_condition or self.success_conditions:
            self._refresh_success_runtime()

    @classmethod
    def from_existing_comm(
        cls,
        comm,
        *,
        scene: int = 0,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "outputs",
        num_agents: int = 1,
        char_resource: str = "Chars/Female1",
    ) -> "VirtualHomeEnvWrapper":
        """Create a VirtualHomeEnvWrapper that reuses an already-connected comm.

        Use this factory method when the UnityCommunication object already exists
        (e.g. in InteractiveController) so the same connection is shared for both
        interactive and evaluation execution paths.

        Args:
            comm: An already-connected UnityCommunication instance.
            scene: VirtualHome scene index (used for reset()).
            config: Configuration dict (same format as __init__ ``config`` argument).
            output_dir: Directory for frame images and results.
            num_agents: Number of character agents.
            char_resource: Character asset name.

        Returns:
            A fully-initialised VirtualHomeEnvWrapper sharing ``comm``.
        """
        # Bypass __init__ entirely to avoid a new UnityCommunication connection.
        obj = cls.__new__(cls)

        # Minimal BaseEnv init
        try:
            from envs.base import BaseEnv
            BaseEnv.__init__(obj, config or {})
        except Exception:
            pass

        obj.config = config or {}
        env_cfg = obj.config.get("env", {})
        actions_config = obj.config.get("actions", {})

        # Scene / connection settings
        obj.scene = int(env_cfg.get("scene", scene))
        obj.port = str(resolve_backend_port(config))
        obj.url = resolve_backend_host(config)
        obj.width = int(env_cfg.get("width", 300))
        obj.height = int(env_cfg.get("height", 300))
        obj.text_state_mode = env_cfg.get("text_state_mode", "first_person")
        if obj.text_state_mode not in {"first_person", "omniscient"}:
            obj.text_state_mode = "first_person"

        obj.output_dir = output_dir
        obj.num_agents = num_agents
        obj.char_resource = char_resource
        obj.task_description = ""
        obj.require_visible_for_interaction = bool(
            env_cfg.get("require_visible_for_interaction", True)
        )
        obj.require_close_for_interaction = True
        obj.bound_instance_update_policy = str(
            env_cfg.get("bound_instance_update_policy", "latest")
        ).lower()
        if obj.bound_instance_update_policy not in {"latest", "first"}:
            obj.bound_instance_update_policy = "latest"

        obj.target_object_types: List[str] = []
        obj.success_condition = None
        obj.success_conditions = []
        obj.success_logic = "AND"
        obj.success_predicate = None
        obj.success_predicates = []
        obj.success_evaluator = None
        obj.target_description = ""

        obj.success_reward = 10.0
        obj.step_success_bonus = 0.1
        obj.step_failure_penalty = -0.05

        obj.move_small_magnitude = float(actions_config.get("move_small_magnitude", 0.25))
        obj.move_medium_magnitude = float(actions_config.get("move_medium_magnitude", 0.5))
        obj.move_large_magnitude = float(actions_config.get("move_large_magnitude", 1.0))
        obj.walk_forward_base_unit_meters = float(
            actions_config.get("walk_forward_base_unit_meters", 0.25)
        )
        obj.vh_turn_step_degrees = float(actions_config.get("vh_turn_step_degrees", 30.0))
        obj.turn_default_degrees = float(actions_config.get("turn_default_degrees", 90.0))
        obj.walk_yaw_restore_min_delta_degrees = float(
            actions_config.get("walk_yaw_restore_min_delta_degrees", 30.0)
        )
        obj.grab_yaw_restore_min_delta_degrees = float(
            actions_config.get("grab_yaw_restore_min_delta_degrees", 20.0)
        )
        obj.grab_exact_restore_tolerance_degrees = float(
            actions_config.get("grab_exact_restore_tolerance_degrees", 2.0)
        )
        obj.grab_yaw_residual_warn_degrees = float(
            actions_config.get("grab_yaw_residual_warn_degrees", 8.0)
        )
        obj.grab_yaw_target_tolerance_degrees = float(
            actions_config.get("grab_yaw_target_tolerance_degrees", 6.0)
        )
        obj.grab_yaw_closed_loop_max_turns = int(
            actions_config.get("grab_yaw_closed_loop_max_turns", 14)
        )
        obj.grab_reset_camera_pose = bool(actions_config.get("grab_reset_camera_pose", True))
        obj.strict_init_rotation = bool(actions_config.get("strict_init_rotation", False))
        obj.stuck_walk_position_threshold = float(actions_config.get("stuck_walk_position_threshold", 0.05))
        obj.spawn_max_valid_y = float(actions_config.get("spawn_max_valid_y", 2.0))
        obj.visible_seg_pixel_threshold = int(actions_config.get("visible_seg_pixel_threshold", 25))
        obj.visibility_sources = _parse_visibility_sources(
            actions_config.get("visibility_sources", ["seg_inst", "api"])
        )
        obj.interaction_distance_meters = float(actions_config.get("interaction_distance_meters", 0.0))
        obj.collision_rollback_enabled = bool(actions_config.get("collision_rollback_enabled", True))
        obj.collision_bbox_shrink_meters = float(actions_config.get("collision_bbox_shrink_meters", 0.02))
        obj.atomic_walkforward_enabled = bool(actions_config.get("atomic_walkforward_enabled", True))
        obj.atomic_walkforward_tolerance_meters = float(
            actions_config.get("atomic_walkforward_tolerance_meters", 0.05)
        )

        # Internal state — will be synced from InteractiveController before each step
        obj._current_graph: Optional[Dict[str, Any]] = None
        obj._char_ids: List[int] = []
        obj._char_camera_ids: List[int] = []
        obj._instance_color_to_id: Optional[Dict[Tuple[int, int, int], int]] = None
        obj._last_visibility_source: str = "none"
        obj._last_visibility_camera_id: Optional[int] = None
        obj._seg_visibility_cache: Dict[
            Tuple[int, int, int, int, int], Tuple[set, Dict[int, int]]
        ] = {}
        obj._bound_instances: Dict[str, int] = {}
        obj._pitch_idx: int = _VH_FP_DEFAULT_IDX
        obj._char_yaw_rad: Optional[float] = None
        obj._skip_first_walk_yaw_restore = False
        obj._last_char_position: Optional[List[float]] = None  # For stuck-walk detection
        obj._scene_interactable_object_types: List[str] = []
        obj._scene_all_object_types: List[str] = []
        obj.step_counter = 0
        obj.action_sequence: List[str] = []

        os.makedirs(output_dir, exist_ok=True)

        # Reuse existing comm — no new connection
        obj.comm = comm
        return obj

    # =========================================================================
    # Task configuration (mirrors AI2ThorEnvWrapper.configure_task)
    # =========================================================================

    def configure_task(
        self,
        target_object_types: List[str],
        success_predicate: Callable[[dict], bool],
        target_description: str,
        success_condition: Optional[Dict[str, Any]] = None,
        success_conditions: Optional[List[Dict[str, Any]]] = None,
        success_logic: str = "AND",
    ):
        """Configure task parameters at runtime.

        Args:
            target_object_types: Object class names to track (e.g. ['microwave']).
            success_predicate: Function (object_dict) -> bool that returns True when
                               the success condition is met for an object.
            target_description: Human-readable task description.
            success_condition: Optional structured success condition config.
            success_conditions: Optional list of structured success conditions.
            success_logic: Logic operator for multi-condition tasks ("AND"/"OR").
        """
        self.target_object_types = [
            canonicalize_object_type_name(t) for t in (target_object_types or [])
        ]
        self.target_description = target_description
        (
            normalized_condition,
            normalized_conditions,
            normalized_logic,
        ) = self._normalize_success_condition_config(
            success_condition,
            success_conditions,
            success_logic,
        )
        if normalized_condition or normalized_conditions:
            self.success_condition = normalized_condition
            self.success_conditions = normalized_conditions
            self.success_logic = normalized_logic
            self._refresh_success_runtime()
        else:
            self.success_condition = None
            self.success_conditions = []
            self.success_logic = "AND"
            self.success_predicate = success_predicate
            self.success_predicates = [success_predicate] if callable(success_predicate) else []
            self.success_evaluator = None
        print(f"✓ Task configured: {target_description}")

    def _normalize_success_condition_config(
        self,
        success_condition: Optional[Dict[str, Any]],
        success_conditions: Optional[Any],
        success_logic: Optional[str],
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], str]:
        """Normalize success condition inputs to a canonical internal representation."""
        normalized_conditions: List[Dict[str, Any]] = []

        if isinstance(success_conditions, list):
            normalized_conditions = [
                cond for cond in success_conditions if isinstance(cond, dict) and cond
            ]
        elif isinstance(success_conditions, dict) and success_conditions:
            normalized_conditions = [success_conditions]

        normalized_single = (
            success_condition
            if isinstance(success_condition, dict) and success_condition
            else None
        )
        if normalized_single is None and normalized_conditions:
            normalized_single = normalized_conditions[0]

        if not normalized_conditions and normalized_single is not None:
            normalized_conditions = [normalized_single]

        logic = str(success_logic or "AND").upper()
        if logic not in {"AND", "OR"}:
            print(f"⚠️  Unsupported success_logic '{success_logic}', fallback to AND")
            logic = "AND"

        return normalized_single, normalized_conditions, logic

    def _refresh_success_runtime(self):
        """Refresh predicates/evaluator after success-condition configuration changes."""
        self.success_predicates = []
        if self.success_conditions:
            for cond in self.success_conditions:
                self.success_predicates.append(
                    self._build_success_predicate_from_condition(cond)
                )
        elif self.success_condition:
            self.success_predicates = [
                self._build_success_predicate_from_condition(self.success_condition)
            ]

        self.success_predicate = (
            self.success_predicates[0] if self.success_predicates else None
        )

        self.success_evaluator = None
        try:
            from evaluation.procthor.base import create_evaluator_from_config

            evaluator_cfg = {
                "target_object_types": self.target_object_types,
                "success_condition": self.success_condition,
                "success_conditions": self.success_conditions,
                "success_logic": self.success_logic,
            }
            self.success_evaluator = create_evaluator_from_config(evaluator_cfg)
        except Exception as exc:
            print(f"⚠️  Failed to initialize success evaluator, fallback to predicate mode: {exc}")
            self.success_evaluator = None

    def _build_success_predicate_from_config(self) -> Callable[[dict], bool]:
        """Build a success condition predicate from the YAML success_condition block."""
        return self._build_success_predicate_from_condition(self.success_condition)

    def _build_success_predicate_from_condition(
        self, condition: Optional[Dict[str, Any]]
    ) -> Callable[[dict], bool]:
        """Build a predicate for a single condition dict."""
        if not condition:
            return lambda obj: False

        ctype = condition.get("type", "object_state")

        if ctype == "object_state":
            field = condition.get("state") or condition.get("field", "isOpen")
            target_value = condition.get("value", True)
            return lambda obj, f=field, v=target_value: obj.get(f, False) == v

        elif ctype == "object_in_receptacle":
            receptacle_type = canonicalize_object_type_name(
                condition.get("receptacle_type", "")
            )
            expected = condition.get("value", True)

            def check_in_receptacle(obj, rt=receptacle_type, exp=expected):
                parents = [
                    canonicalize_object_type_name(p)
                    for p in obj.get("parentReceptacles", [])
                ]
                in_rec = any(p == rt or p.startswith(rt) for p in parents)
                return in_rec == exp

            return check_in_receptacle

        elif ctype == "object_in_hand":
            expected = condition.get("value", True)
            return lambda obj, exp=expected: obj.get("isPickedUp", False) == exp

        else:
            print(f"⚠️  Unsupported success_condition type: {ctype}")
            return lambda obj: False

    # =========================================================================
    # Core interface (BaseEnv abstract methods)
    # =========================================================================

    def _get_environment_graph_with_retry(
        self,
        retries: int = 8,
        delay_sec: float = 1.0,
        stage: str = "",
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Fetch environment_graph() with retries for Unity transient instability."""
        import time as _time

        last_exc: Optional[Exception] = None
        for i in range(retries):
            try:
                ok, graph = self.comm.environment_graph()
            except Exception as exc:
                ok, graph = False, None
                last_exc = exc

            if ok and graph:
                return True, graph

            if i < retries - 1:
                tag = f"[{stage}] " if stage else ""
                print(
                    f"  ⚠️  {tag}environment_graph() not ready "
                    f"(attempt {i + 1}/{retries}), retrying..."
                )
                _time.sleep(delay_sec)

        if last_exc is not None:
            tag = f"[{stage}] " if stage else ""
            print(f"  ⚠️  {tag}environment_graph() exception: {last_exc}")
        return False, None

    def reset(
        self,
        task_description: str = "",
        *,
        scene: Optional[int] = None,
        char_position=None,
        char_rotation=None,
        char_yaw_degrees=None,
        camera_pitch=None,
    ) -> EnvObservation:
        """Reset the VirtualHome environment.

        This binds to the currently running Unity scene and restores the
        benchmark init pose/camera state. Unity scene switching is handled by
        launching an isolated backend per task.

        Args:
            task_description: Natural-language task description.
            scene: Optional scene label override (int or str). Used for wrapper
                bookkeeping/logging only; does not switch Unity scene.
            char_position: Optional character spawn position [x, y, z].
            char_rotation: Optional target rotation quaternion [x, y, z, w].
            char_yaw_degrees: Optional target yaw in degrees (preferred init field).
            camera_pitch: Optional camera pitch angle in degrees (positive=down).

        Returns:
            Initial EnvObservation.
        """
        self.step_counter = 0
        self.action_sequence = []
        self._bound_instances = {}
        self.task_description = task_description
        self._char_yaw_rad = None
        self._pitch_idx = _VH_FP_DEFAULT_IDX
        self._last_char_position = None
        resolved_char_rotation = coerce_init_rotation_quat(
            char_rotation=char_rotation,
            char_yaw_degrees=char_yaw_degrees,
            step_degrees=30.0,
        )
        self._skip_first_walk_yaw_restore = bool(resolved_char_rotation)

        # Scene override
        if scene is not None:
            try:
                self.scene = int(scene)
            except (ValueError, TypeError):
                print(
                    f"⚠️  Cannot convert scene '{scene}' to int; keeping current {self.scene}"
                )

        # Ensure task is configured
        if not self.target_object_types:
            print(
                "⚠️  Task not configured; using generic success_predicate (always False)"
            )

        import time as _time

        print(f"Binding current VirtualHome Unity scene (configured scene={self.scene}).")

        _, raw = self.comm.character_cameras()
        existing = _parse_cam_names(raw)
        for pitch_angle, cam_name in _VH_FP_PITCH_CAMERAS:
            if cam_name not in existing:
                self.comm.add_character_camera(
                    position=_VH_FP_CAMERA_OFFSET,
                    rotation=[pitch_angle, 0, 0],
                    field_view=60,
                    name=cam_name,
                )
            else:
                try:
                    self.comm.update_character_camera(
                        position=_VH_FP_CAMERA_OFFSET,
                        rotation=[pitch_angle, 0, 0],
                        field_view=60,
                        name=cam_name,
                    )
                except Exception:
                    pass

        # Add/reuse character after binding the current backend scene.
        self._char_ids = []
        existing_char_count = 0
        ok_pre_graph, pre_graph = self._get_environment_graph_with_retry(
            retries=3, delay_sec=0.5, stage="pre-check"
        )
        if ok_pre_graph and pre_graph:
            existing_char_count = len(find_character_ids(pre_graph))
        resource = self.char_resource
        if existing_char_count == 0:
            if char_position:
                self.comm.add_character(resource, position=char_position)
                _time.sleep(1.0)
                # add_character   NavMesh         ；  move_character   
                try:
                    self.comm.move_character(0, char_position)
                except Exception:
                    pass
            else:
                self.comm.add_character(resource)
        elif char_position:
            try:
                self.comm.move_character(0, char_position)
            except Exception:
                pass

        # Wait Unity to settle character/camera attachment.
        _time.sleep(1.0)

        #   ：             live yaw，           
        live_char_yaw = self._get_live_char_yaw()
        if live_char_yaw is not None:
            self._char_yaw_rad = live_char_yaw

        # Restore character rotation if specified (VirtualHome only supports 30° discrete turn steps).
        if resolved_char_rotation:
            self._restore_rotation(
                resolved_char_rotation,
                exact=True,
                require_exact=self.strict_init_rotation,
                position_hint=char_position,
            )

        # Fetch initial graph
        ok, graph = self._get_environment_graph_with_retry(
            retries=10, delay_sec=1.0, stage="post-bind"
        )
        if not ok:
            if ok_pre_graph and pre_graph:
                print("  ⚠️ post-bind graph unavailable; reusing pre-check graph")
                graph = pre_graph
            else:
                raise RuntimeError("Failed to get environment graph after scene bind")
        self._current_graph = graph
        self._refresh_scene_object_type_cache(graph)
        if self._scene_interactable_object_types:
            print(
                "  🧩 Scene interactable object tokens: "
                f"{len(self._scene_interactable_object_types)}"
            )

        # Resolve character node IDs
        self._char_ids = find_character_ids(graph)
        if self._char_yaw_rad is None:
            self._char_yaw_rad = self._get_char_yaw()
        if self._char_yaw_rad is None:
            raise RuntimeError("Failed to resolve character yaw after scene bind")

        # ── Bug B fix: validate spawn Y to detect off-NavMesh / floating character ──
        self._validate_and_fix_spawn(graph)

        # Determine camera ID used for observation/visibility checks.
        # Prefer dedicated first-person character cameras; fallback to heuristic only if setup fails.
        fp_camera_ids = self._ensure_first_person_cameras()
        if fp_camera_ids:
            self._char_camera_ids = fp_camera_ids
            if self._pitch_idx >= len(self._char_camera_ids):
                self._pitch_idx = max(0, len(self._char_camera_ids) - 1)
            selected_camera = self._char_camera_ids[self._pitch_idx]
            print(
                f"  📷 Using first-person camera: {selected_camera} "
                f"(fp_set={self._char_camera_ids})"
            )
        else:
            ok2, cam_count = self.comm.camera_count()
            if ok2 and cam_count > 0:
                selected_camera = self._select_primary_camera_id(graph, cam_count)
                self._char_camera_ids = [selected_camera]
                print(
                    f"  📷 Fallback camera for eval: {selected_camera} (total={cam_count})"
                )
            else:
                self._char_camera_ids = [0]

        # Match interact mode: pitch is controlled by the selected first-person camera,
        # not by sending LookUp/LookDown scripts to Unity.
        self._restore_pitch(camera_pitch if camera_pitch is not None else 0)

        # Capture initial frame
        image_path = self._save_frame(prefix="reset")

        # Build text state
        text_state = self._generate_text_state(graph, last_action_success=True)

        print(
            f"✓ VirtualHome reset complete | Scene: {self.scene} | Task: {task_description[:60]}"
        )

        return EnvObservation(
            image_path=image_path,
            text_state=text_state,
            reward=0.0,
            done=False,
            metadata=self._build_metadata(graph, last_action_success=True),
        )


    # =========================================================================
    # Bug-fix helpers
    # =========================================================================

    def _get_char_position_from_graph(self) -> "Optional[List[float]]":
        """Return current character [x, y, z] from the cached graph snapshot."""
        if not self._current_graph or not self._char_ids:
            return None
        char_id = self._char_ids[0]
        for node in self._current_graph.get("nodes", []):
            if node.get("id") != char_id:
                continue
            pos = self._node_position(node)
            if pos is not None:
                return [float(pos[0]), float(pos[1]), float(pos[2])]
        return None

    @staticmethod
    def _position_distance(a: "List[float]", b: "List[float]") -> float:
        """Euclidean distance between two [x, y, z] positions (horizontal only)."""
        import math
        dx = a[0] - b[0]
        dz = a[2] - b[2]
        return math.hypot(dx, dz)

    def _interaction_restore_candidates(
        self,
        target_position: List[float],
        *,
        avoid_object_id: Optional[int] = None,
    ) -> "List[List[float]]":
        """Positions to try when the exact pre-interaction anchor is blocked.

        Opening doors can make the old foot position invalid in Unity.  Prefer
        small steps away from the interacted object, then a compact local search.
        """
        import math

        candidates: List[List[float]] = [list(target_position)]
        directions: List[Tuple[float, float]] = []

        avoid_node = self._get_node_by_id(avoid_object_id)
        avoid_center = None
        if avoid_node is not None:
            avoid_center, _size = self._node_bbox_center_size(avoid_node)
            if avoid_center is None:
                obj_pos = self._node_position(avoid_node)
                if obj_pos is not None:
                    avoid_center = obj_pos

        if avoid_center is not None:
            dx = float(target_position[0]) - float(avoid_center[0])
            dz = float(target_position[2]) - float(avoid_center[2])
            norm = math.hypot(dx, dz)
            if norm > 1e-6:
                away = (dx / norm, dz / norm)
                directions.extend(
                    [
                        away,
                        (-away[1], away[0]),
                        (away[1], -away[0]),
                        (-away[0], -away[1]),
                    ]
                )

        if not directions:
            directions.extend(
                [
                    (1.0, 0.0),
                    (-1.0, 0.0),
                    (0.0, 1.0),
                    (0.0, -1.0),
                    (0.7071, 0.7071),
                    (-0.7071, 0.7071),
                    (0.7071, -0.7071),
                    (-0.7071, -0.7071),
                ]
            )

        seen = {
            (
                round(float(target_position[0]), 3),
                round(float(target_position[2]), 3),
            )
        }
        for radius in (0.10, 0.20, 0.35, 0.50, 0.75, 1.00):
            for dir_x, dir_z in directions:
                candidate = [
                    float(target_position[0]) + dir_x * radius,
                    float(target_position[1]),
                    float(target_position[2]) + dir_z * radius,
                ]
                key = (round(candidate[0], 3), round(candidate[2], 3))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)

        return candidates

    def _restore_character_position_after_interaction(
        self,
        target_position: Optional[List[float]],
        *,
        avoid_object_id: Optional[int] = None,
        tolerance_meters: float = 0.05,
        retries: int = 3,
    ) -> bool:
        """Restore the character anchor after VH interaction root motion.

        Some VirtualHome interaction animations move the character to an object
        anchor, even for actions that should be logically in-place such as
        Open/Close/SwitchOn.  A single move_character call can also be racing the
        tail of the animation, so verify against a fresh graph and retry.
        """
        if target_position is None:
            return False

        import time as _time

        candidates = self._interaction_restore_candidates(
            list(target_position),
            avoid_object_id=avoid_object_id,
        )
        best_distance = float("inf")
        best_candidate: Optional[List[float]] = None
        best_current: Optional[List[float]] = None
        last_moved_ok = False
        last_error: Optional[Exception] = None

        def _attempt_move(candidate: List[float]) -> Optional[List[float]]:
            nonlocal best_distance, best_candidate, best_current, last_moved_ok, last_error
            try:
                last_moved_ok = bool(self.comm.move_character(0, candidate))
                last_error = None
            except Exception as exc:
                last_moved_ok = False
                last_error = exc

            _time.sleep(0.12)
            ok, graph = self._get_environment_graph_with_retry(
                retries=2,
                delay_sec=0.08,
                stage="interaction-position-restore",
            )
            if ok and graph:
                self._current_graph = graph
                self._char_ids = find_character_ids(graph)

            current_pos = self._get_char_position_from_graph()
            if current_pos is None:
                return None

            distance = self._position_distance(target_position, current_pos)
            if distance < best_distance:
                best_distance = distance
                best_candidate = list(candidate)
                best_current = list(current_pos)
            return current_pos

        for attempt in range(max(1, retries)):
            current_pos = _attempt_move(list(target_position))
            if current_pos is not None:
                distance = self._position_distance(target_position, current_pos)
                if distance <= tolerance_meters:
                    return True

        for candidate in candidates[1:]:
            current_pos = _attempt_move(candidate)
            if current_pos is None:
                continue
            distance = self._position_distance(target_position, current_pos)
            if distance <= tolerance_meters:
                return True

        if best_candidate is not None:
            # Leave Unity at the closest verified point, not at the last failed probe.
            current = _attempt_move(best_candidate)
            if current is not None:
                best_current = current
                best_distance = self._position_distance(target_position, current)

        if last_error is not None:
            print(f"  ⚠️  interaction position restore failed: {last_error}")
        print(
            "  ⚠️  interaction position restore residual="
            f"{best_distance:.3f}m (move_ok={last_moved_ok})"
        )
        if best_current is not None and best_distance < 1.0:
            print(
                "  ℹ️  kept nearest reachable post-interaction position "
                f"at residual={best_distance:.3f}m"
            )
        return False

    def _validate_and_fix_spawn(self, graph: "Dict[str, Any]") -> None:
        """Bug B fix: after reset(), check character Y. If too high, re-spawn.

        VHScene0 and some other scenes occasionally place the character outside the
        NavMesh (e.g. floating above the floor).  All WalkForward actions then return
        "Character cannot reach the destination".  We detect this by inspecting the
        character node's Y coordinate immediately after scene bind and retry
        add_character() up to 2 times if the value is suspiciously high.
        """
        import time as _time

        if not self._char_ids:
            return
        char_id = self._char_ids[0]
        for node in graph.get("nodes", []):
            if node.get("id") != char_id:
                continue
            pos = self._node_position(node)
            if pos is None:
                return
            char_y = float(pos[1])
            if char_y <= self.spawn_max_valid_y:
                print(f"  ✓ Character Y={char_y:.2f}m — spawn valid")
                return

            # Character Y is too high → off-NavMesh / floating
            print(
                f"  ⚠️  Character Y={char_y:.2f}m exceeds spawn_max_valid_y={self.spawn_max_valid_y}m "
                f"— likely off-NavMesh. Attempting re-spawn..."
            )
            for attempt in range(1, 3):
                try:
                    self.comm.add_character(self.char_resource)
                except Exception as exc:
                    print(f"    ⚠️  re-spawn attempt {attempt} failed: {exc}")
                    continue
                _time.sleep(1.5)
                ok, new_graph = self._get_environment_graph_with_retry(
                    retries=3, delay_sec=0.5, stage=f"respawn-{attempt}"
                )
                if not ok or not new_graph:
                    continue
                self._current_graph = new_graph
                self._char_ids = find_character_ids(new_graph)
                if not self._char_ids:
                    continue
                new_char_id = self._char_ids[0]
                for new_node in new_graph.get("nodes", []):
                    if new_node.get("id") != new_char_id:
                        continue
                    new_pos = self._node_position(new_node)
                    if new_pos is None:
                        break
                    new_y = float(new_pos[1])
                    print(
                        f"  ✓ Re-spawn attempt {attempt}: Y={new_y:.2f}m"
                    )
                    if new_y <= self.spawn_max_valid_y:
                        print("  ✓ Spawn Y is now valid")
                        return
                    break
            print(
                "  ⚠️  Re-spawn could not fix Y coordinate. "
                "Consider specifying char_position in the task init file."
            )
            return

    def _normalize_angle_rad(self, angle_rad: float) -> float:
        """Normalize an angle to [-pi, pi]."""
        import math

        while angle_rad > math.pi:
            angle_rad -= 2 * math.pi
        while angle_rad < -math.pi:
            angle_rad += 2 * math.pi
        return angle_rad

    def _quat_to_yaw_rad(self, quat) -> Optional[float]:
        """Convert Unity quaternion [x, y, z, w] to yaw radians."""
        import math

        try:
            qx, qy, qz, qw = (
                float(quat[0]),
                float(quat[1]),
                float(quat[2]),
                float(quat[3]),
            )
        except Exception:
            return None
        return math.atan2(2.0 * (qw * qy + qx * qz), 1.0 - 2.0 * (qy * qy + qz * qz))

    def _coerce_position_list(self, position_hint) -> Optional[List[float]]:
        """Convert position payload to [x, y, z] list."""
        if position_hint is None:
            return None
        if isinstance(position_hint, dict):
            if all(k in position_hint for k in ("x", "y", "z")):
                try:
                    return [
                        float(position_hint["x"]),
                        float(position_hint["y"]),
                        float(position_hint["z"]),
                    ]
                except Exception:
                    return None
            return None
        if isinstance(position_hint, (list, tuple)) and len(position_hint) >= 3:
            try:
                return [
                    float(position_hint[0]),
                    float(position_hint[1]),
                    float(position_hint[2]),
                ]
            except Exception:
                return None
        return None

    def _get_live_char_yaw(self) -> Optional[float]:
        """Fetch a fresh graph snapshot and return current character yaw."""
        ok, graph = self._get_environment_graph_with_retry(
            retries=3, delay_sec=0.2, stage="exact-rot-verify"
        )
        if not ok or not graph:
            return None
        self._current_graph = graph
        self._char_ids = find_character_ids(graph)
        return self._get_char_yaw()

    def _try_restore_rotation_exact(
        self,
        target_quat,
        *,
        position_hint=None,
        tolerance_deg: float = 1.0,
    ) -> bool:
        """Try restoring exact rotation via move_character; verify against live graph yaw."""
        import math
        import time as _time

        target_yaw = self._quat_to_yaw_rad(target_quat)
        if target_yaw is None:
            return False

        pos = self._coerce_position_list(position_hint)
        if pos is None:
            ok, graph = self._get_environment_graph_with_retry(
                retries=3, delay_sec=0.2, stage="exact-rot-pos"
            )
            if not ok or not graph:
                return False
            self._current_graph = graph
            self._char_ids = find_character_ids(graph)
            if not self._char_ids:
                return False
            char_id = self._char_ids[0]
            for node in graph.get("nodes", []):
                if node.get("id") != char_id:
                    continue
                node_pos = self._node_position(node)
                if node_pos is not None:
                    pos = [float(node_pos[0]), float(node_pos[1]), float(node_pos[2])]
                break
            if pos is None:
                return False

        move_character = getattr(self.comm, "move_character", None)
        if move_character is None:
            return False

        target_yaw_deg = math.degrees(target_yaw)
        target_euler = [0.0, target_yaw_deg, 0.0]

        attempts = [
            lambda: move_character(0, pos, target_quat),
            lambda: move_character(0, pos, target_euler),
            lambda: move_character(0, pos, target_yaw_deg),
            lambda: move_character(0, pos, rotation=target_quat),
            lambda: move_character(0, pos, rotation=target_euler),
            lambda: move_character(0, pos, rotation=target_yaw_deg),
            lambda: move_character(0, pos, orientation=target_quat),
            lambda: move_character(0, pos, orientation=target_euler),
            lambda: move_character(0, pos, orientation=target_yaw_deg),
            lambda: move_character(0, pos, rot=target_quat),
            lambda: move_character(0, pos, rot=target_euler),
            lambda: move_character(0, pos, rot=target_yaw_deg),
        ]

        for attempt in attempts:
            try:
                attempt()
            except TypeError:
                continue
            except Exception:
                continue

            _time.sleep(0.15)
            current_yaw = self._get_live_char_yaw()
            if current_yaw is None:
                continue
            delta_deg = abs(
                math.degrees(self._normalize_angle_rad(target_yaw - current_yaw))
            )
            if delta_deg <= tolerance_deg:
                self._char_yaw_rad = target_yaw
                print(f"  ✓ Exact init rotation restored (yaw error={delta_deg:.3f}°)")
                return True

        return False

    def _restore_rotation(
        self,
        target_quat,
        *,
        exact: bool = False,
        require_exact: bool = False,
        position_hint=None,
    ):
        """Restore character rotation using TurnLeft/TurnRight.

        Args:
            target_quat: Target quaternion [x, y, z, w] (Unity format).
        """
        import math
        import time as _time

        if exact:
            if self._try_restore_rotation_exact(
                target_quat,
                position_hint=position_hint,
                tolerance_deg=1.0,
            ):
                return
            if require_exact:
                raise RuntimeError(
                    "Exact init rotation restore failed. "
                    "This backend likely does not support setting character rotation directly "
                    "(move_character only supports position), so exact yaw cannot be guaranteed."
                )
            print(
                "  ⚠️  Exact rotation restore unavailable; fallback to best-effort discrete turns"
            )

        #    Unity         
        _time.sleep(0.5)
        #        yaw；          
        current_yaw = self._get_live_char_yaw()
        if current_yaw is None:
            current_yaw = self._char_yaw_rad
        if current_yaw is None:
            print("  ⚠️  _restore_rotation: cannot determine current yaw")
            return

        target_yaw = self._quat_to_yaw_rad(target_quat)
        if target_yaw is None:
            print("  ⚠️  _restore_rotation: invalid target quaternion")
            return

        delta = target_yaw - current_yaw
        # Normalize to [-π, π]
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi

        # Each TurnLeft/TurnRight rotates 30° = π/6
        turns = round(delta / (math.pi / 6.0))
        print(
            f"  🔄 _restore_rotation: current={math.degrees(current_yaw):.1f}° "
            f"target={math.degrees(target_yaw):.1f}° delta={math.degrees(delta):.1f}° "
            f"turns={turns}"
        )
        if turns == 0:
            # Keep a client-side yaw cache in sync with the requested pose.
            # VirtualHome graph rotation can lag briefly after reset/render_script.
            self._char_yaw_rad = target_yaw
            return

        action = "[TurnRight]" if turns > 0 else "[TurnLeft]"
        n = abs(turns)

        # Match the interactive worker: send each native turn as its own
        # render_script call so Unity has the same graph/camera update cadence.
        script_lines = [f"<char0> {action}" for _ in range(n)]
        try:
            ok = True
            msg = None
            for script_line in script_lines:
                ok, msg = self.comm.render_script(
                    [script_line],
                    skip_animation=False,
                    image_synthesis=[],
                    recording=False,
                    time_scale=VH_INTERACT_TIME_SCALE,
                )
                if not ok:
                    break
            if not ok:
                print(f"  ⚠️  _restore_rotation: render_script failed: {msg}")
            else:
                # Match interact mode: trust the requested target yaw immediately
                # instead of waiting for a potentially stale environment_graph().
                self._char_yaw_rad = target_yaw
                live_yaw = self._get_live_char_yaw()
                if live_yaw is not None:
                    residual_deg = abs(
                        math.degrees(self._normalize_angle_rad(target_yaw - live_yaw))
                    )
                    print(
                        f"  ℹ️  Discrete rotation residual yaw error={residual_deg:.3f}°"
                    )
        except Exception as exc:
            print(f"  ⚠️  _restore_rotation: render_script exception: {exc}")

        _time.sleep(0.3)

    def _restore_pitch(self, target_pitch_deg):
        """Restore pitch by switching to the nearest registered first-person camera."""
        best_idx = _VH_FP_DEFAULT_IDX
        best_diff = abs(
            _VH_FP_PITCH_CAMERAS[_VH_FP_DEFAULT_IDX][0] - float(target_pitch_deg)
        )
        for i, (angle, _name) in enumerate(_VH_FP_PITCH_CAMERAS):
            diff = abs(angle - float(target_pitch_deg))
            if diff < best_diff:
                best_idx = i
                best_diff = diff

        self._pitch_idx = best_idx
        actual_pitch = _VH_FP_PITCH_CAMERAS[self._pitch_idx][0]
        print(f"  ✓ Pitch camera restored: {actual_pitch}° (idx={self._pitch_idx})")

    def _resolve_walk_repeat_count(
        self,
        magnitude: Optional[float],
        granularity: Optional[str],
    ) -> int:
        """Map target meters (or small/medium/large) to number of [WalkForward] primitives."""
        if magnitude is not None and isinstance(magnitude, (int, float)):
            m = float(magnitude)
        elif granularity:
            g = str(granularity).strip().lower()
            if g == "small":
                m = self.move_small_magnitude
            elif g == "medium":
                m = self.move_medium_magnitude
            elif g == "large":
                m = self.move_large_magnitude
            else:
                m = self.move_small_magnitude
        else:
            m = self.move_small_magnitude
        unit = self.walk_forward_base_unit_meters
        if unit <= 0:
            unit = 0.25
        n = int(round(m / unit))
        return max(1, n)

    def step_with_action_dict(
        self, action_dict: dict
    ) -> Tuple[EnvObservation, Optional[str]]:
        """Execute one action step using an action dictionary.

        Mirrors AI2ThorEnvWrapper.step_with_action_dict.

        Args:
            action_dict: Dict with keys:
                - action_type: "navigation" | "interaction" | "task_completion"
                - action_name: VirtualHome action name (e.g. "WalkForward", "Grab", "DONE")
                - object_type: (optional) target object class name
                - object2_type: (optional) second object for two-object actions

        Returns:
            (EnvObservation, error_message) where error_message is None on success.
        """
        self.step_counter += 1

        action_type = action_dict.get("action_type")
        action_name = action_dict.get("action_name", "")
        object_type = action_dict.get("object_type")
        object2_type = action_dict.get("object2_type")

        print(f"\n--- Step {self.step_counter} ---")
        print(f"🎬 Action: {action_name}" + (f"({object_type})" if object_type else ""))

        error_message: Optional[str] = None
        last_action_success = True

        if action_type == "task_completion":
            # DONE / FAIL – no environment action, just mark
            print(f"🏁 Task completion: {action_name}")
            self.action_sequence.append(action_name)

        elif action_type in ("navigation", "interaction"):
            executed_script_line: Optional[str] = None
            # Keep evaluate aligned with interact mode:
            # LookUp/LookDown switch the active first-person camera rather than
            # relying on VirtualHome script actions.
            if action_name in ("LookUp", "LookDown"):
                old_idx = self._pitch_idx
                if action_name == "LookUp":
                    self._pitch_idx = max(0, self._pitch_idx - 1)
                else:
                    self._pitch_idx = min(
                        len(_VH_FP_PITCH_CAMERAS) - 1,
                        self._pitch_idx + 1,
                    )
                actual_pitch = _VH_FP_PITCH_CAMERAS[self._pitch_idx][0]
                self.action_sequence.append(action_name)
                if old_idx != self._pitch_idx:
                    print(
                        f"  ✓ Camera pitch -> {actual_pitch}° (idx={self._pitch_idx})"
                    )
                ok2, graph = self._get_environment_graph_with_retry(
                    retries=2, delay_sec=0.2, stage="pitch-step"
                )
                if ok2:
                    self._current_graph = graph
            else:
                char_idx = 0
                vh_nav = ACTION_NAME_MAP.get(action_name, action_name)
                script_line: Optional[str] = None
                script_lines: Optional[List[str]] = None
                build_error: Optional[str] = None
                walk_pre_yaw: Optional[float] = None
                walk_pre_pos: Optional[List[float]] = None  # Bug A: pre-walk position snapshot
                walk_yaw_restored_per_native_step = False
                turn_delta_rad: Optional[float] = None
                pre_interaction_pos: Optional[List[float]] = None
                pre_interaction_rot: Optional[List[float]] = None
                interaction_target_ids: List[int] = []

                if action_type == "navigation" and vh_nav == "WalkForward":
                    repeats = self._resolve_walk_repeat_count(
                        action_dict.get("magnitude"),
                        action_dict.get("granularity"),
                    )
                    walk_pre_yaw = (
                        self._char_yaw_rad
                        if self._char_yaw_rad is not None
                        else self._get_char_yaw()
                    )
                    # Bug A fix: snapshot position before walking for stuck detection
                    walk_pre_pos = self._get_char_position_from_graph()
                    script_lines = [f"<char{char_idx}> [WalkForward]"] * repeats
                elif action_type == "navigation" and vh_nav in (
                    "TurnLeft",
                    "TurnRight",
                ):
                    deg = action_dict.get("turn_degrees")
                    if deg is None:
                        deg = self.turn_default_degrees
                    n_turns = max(
                        1,
                        int(round(float(deg) / self.vh_turn_step_degrees)),
                    )
                    turn_step_rad = 3.141592653589793 / 6.0
                    turn_delta_rad = (
                        n_turns * turn_step_rad
                        if vh_nav == "TurnRight"
                        else -n_turns * turn_step_rad
                    )
                    script_lines = [f"<char{char_idx}> [{vh_nav}]"] * n_turns
                else:
                    step_visible_ids: Optional[set] = None
                    if action_type == "interaction":
                        step_visible_ids = self._get_visible_object_ids()
                    script_line, build_error = self._build_script_line(
                        action_name,
                        object_type,
                        object2_type,
                        char_index=char_idx,
                        action_type=action_type,
                        object_id=action_dict.get("object_id"),
                        object2_id=action_dict.get("object2_id"),
                        visible_ids=step_visible_ids,
                    )
                    if script_line and not build_error:
                        script_lines = [script_line]

                if build_error:
                    error_message = build_error
                    last_action_success = False
                    self.action_sequence.append(f"{action_name}(FAILED)")
                    print(f"⚠️  Build script failed: {error_message}")
                elif not script_lines:
                    error_message = error_message or "Failed to build script"
                    last_action_success = False
                    self.action_sequence.append(f"{action_name}(FAILED)")
                    print(f"⚠️  Build script failed: {error_message}")
                else:
                    script_line = script_lines[0]
                    #      ：            ，  “     ” 
                    # This is intentionally unconditional for VirtualHome so
                    # agent execution, golden replay, and manual evaluation all
                    # share the same interaction contract.
                    if action_type == "interaction":
                        visible_ok, visible_err = self._validate_interaction_visibility(
                            script_line,
                            object_type,
                            object2_type,
                            visible_ids=step_visible_ids,
                        )
                        if not visible_ok:
                            error_message = visible_err
                            last_action_success = False
                            self.action_sequence.append(f"{action_name}(NOT_VISIBLE)")
                            print(f"⚠️  Action blocked: {error_message}")

                    if error_message:
                        pass
                    else:
                        if (
                            action_type == "interaction"
                            and self._current_graph
                            and self._char_ids
                        ):
                            char_id = self._char_ids[0]
                            for node in self._current_graph.get("nodes", []):
                                if node.get("id") != char_id:
                                    continue
                                pos = self._node_position(node)
                                rot = node.get("obj_transform", {}).get("rotation")
                                if pos is not None:
                                    pre_interaction_pos = list(pos)
                                if rot is not None:
                                    pre_interaction_rot = list(rot)
                                break

                        # Record in stable readable format, e.g. WalkForward, PutIn(apple, fridge)
                        self.action_sequence.append(
                            self._format_action_for_sequence(
                                action_name,
                                object_type,
                                object2_type,
                                action_dict,
                            )
                        )
                        executed_script_line = (
                            script_lines[0]
                            if len(script_lines) == 1
                            else " | ".join(script_lines)
                        )
                        if action_type == "interaction":
                            interaction_target_ids = [
                                int(m)
                                for m in re.findall(
                                    r"\((\d+)\)", executed_script_line or ""
                                )
                            ]

                        # Keep interact/evaluate camera behavior aligned:
                        # Turn/Walk must animate so avatar transform + attached FP cameras update physically.
                        # PutBack/PutIn also rely on VH's held-object placement state machine;
                        # skipping their animation can make Unity intermittently reject valid scripts.
                        if vh_nav in ("TurnLeft", "TurnRight", "WalkForward", "PutBack", "PutIn"):
                            use_skip_animation = False
                        else:
                            use_skip_animation = True

                        # Execute via render_script. WalkForward and turns are
                        # executed one native step at a time to match the
                        # interactive pygame path; batching multiple turn
                        # scripts in one call can leave Unity's avatar rotation
                        # graph stale or partially applied.
                        ok = True
                        message = None
                        if action_type == "navigation" and vh_nav == "WalkForward":
                            walk_yaw_restored_per_native_step = True
                            for single_line in script_lines:
                                step_pre_yaw = walk_pre_yaw
                                step_pre_pos = None
                                ok_pre, graph_pre = self._get_environment_graph_with_retry(
                                    retries=3,
                                    delay_sec=0.2,
                                    stage="walkforward-pre-step",
                                )
                                if ok_pre and graph_pre:
                                    self._current_graph = graph_pre
                                    self._char_ids = find_character_ids(graph_pre)
                                    step_pre_yaw = self._get_char_yaw()
                                    step_pre_pos = self._get_char_position_from_graph()

                                try:
                                    ok, message = self.comm.render_script(
                                        [single_line],
                                        skip_animation=use_skip_animation,
                                        image_synthesis=[],
                                        recording=False,
                                        time_scale=VH_INTERACT_TIME_SCALE,
                                    )
                                except Exception as exc:
                                    ok, message = False, str(exc)
                                if not ok:
                                    if self.atomic_walkforward_enabled:
                                        self._rollback_character_to_walk_start(
                                            walk_pre_pos,
                                            pre_yaw=walk_pre_yaw,
                                            stage="walkforward-failed-step",
                                        )
                                        message = (
                                            f"Atomic WalkForward rollback — native substep failed: {message}; "
                                            "restored action start position."
                                        )
                                    break

                                rolled_back, rollback_message = self.rollback_if_collision_intrusion(
                                    step_pre_pos,
                                    pre_yaw=step_pre_yaw,
                                    stage="walkforward",
                                )
                                if rollback_message:
                                    message = rollback_message
                                if rolled_back:
                                    ok = False
                                    break
                                if step_pre_yaw is not None:
                                    self._restore_yaw_after_walk(step_pre_yaw)
                                self._grab_active_camera_frame_for_settle()
                        elif action_type == "navigation" and vh_nav in (
                            "TurnLeft",
                            "TurnRight",
                        ):
                            for single_line in script_lines:
                                try:
                                    ok, message = self.comm.render_script(
                                        [single_line],
                                        skip_animation=use_skip_animation,
                                        image_synthesis=[],
                                        recording=False,
                                        time_scale=VH_INTERACT_TIME_SCALE,
                                    )
                                except Exception as exc:
                                    ok, message = False, str(exc)
                                if not ok:
                                    break
                        else:
                            try:
                                render_kwargs = {
                                    "skip_animation": use_skip_animation,
                                    "image_synthesis": [],  #    Unity     ，    
                                    "recording": False,
                                }
                                if vh_nav in ("PutBack", "PutIn"):
                                    render_kwargs["time_scale"] = VH_INTERACT_TIME_SCALE
                                ok, message = self.comm.render_script(
                                    script_lines,
                                    **render_kwargs,
                                )
                            except Exception as exc:
                                ok, message = False, str(exc)
                            if (
                                not ok
                                and action_type == "interaction"
                                and vh_nav in ("PutBack", "PutIn")
                                and self._is_transient_vh_executor_error(message)
                                and self._can_retry_held_object_placement(
                                    interaction_target_ids
                                )
                            ):
                                print(
                                    "  ℹ️  Retrying held-object placement after Unity executor settle"
                                )
                                self._settle_before_interaction_retry()
                                try:
                                    ok, message = self.comm.render_script(
                                        script_lines,
                                        skip_animation=False,
                                        image_synthesis=[],
                                        recording=False,
                                        time_scale=VH_INTERACT_TIME_SCALE,
                                    )
                                except Exception as exc:
                                    ok, message = False, str(exc)
                        if not ok:
                            last_action_success = False
                            raw_err = str(message) if message else "Unknown error"
                            if raw_err.startswith(
                                (
                                    "WalkForward collision rollback",
                                    "Atomic WalkForward",
                                )
                            ):
                                error_message = raw_err
                            else:
                                error_message = self._translate_error(
                                    raw_err, action_name, object_type
                                )
                            print(f"⚠️  Action failed: {error_message}")
                        else:
                            if turn_delta_rad is not None:
                                # Match pygame interact mode: after A/D turns it
                                # refreshes the graph and trusts Unity's live yaw.
                                # Keeping a client-side accumulated yaw here can
                                # diverge from Unity and make later WalkForward
                                # restore the avatar toward the wrong heading.
                                self._sync_yaw_cache_from_live()

                            if action_type == "interaction":
                                # Sit changes the avatar pose relative to a target object.
                                # Restoring the pre-interaction anchor can undo the sitting relation.
                                preserve_posture_anchor = action_name in ("Sit",)

                                # Interaction animations may move character to an object anchor.
                                # For most actions we rollback to avoid unintended drift.
                                if pre_interaction_pos is not None and not preserve_posture_anchor:
                                    self._restore_character_position_after_interaction(
                                        pre_interaction_pos,
                                        avoid_object_id=(
                                            interaction_target_ids[0]
                                            if interaction_target_ids
                                            else None
                                        ),
                                    )

                                # Keep Grab rotation handling conservative: avoid full rotation rollback,
                                # only apply yaw correction when drift is clearly large.
                                should_restore_rotation = action_name not in (
                                    "Grab",
                                    "Sit",
                                )
                                if (
                                    should_restore_rotation
                                    and pre_interaction_rot is not None
                                ):
                                    self._restore_rotation(pre_interaction_rot)
                                if (
                                    action_name == "Grab"
                                    and pre_interaction_rot is not None
                                ):
                                    self._restore_yaw_after_interaction(
                                        pre_interaction_rot,
                                        min_delta_deg=self.grab_yaw_restore_min_delta_degrees,
                                    )
                                if action_name == "Grab":
                                    self._reset_active_pitch_camera_pose()

                            if (
                                walk_pre_yaw is not None
                                and not walk_yaw_restored_per_native_step
                            ):
                                self._restore_yaw_after_walk(walk_pre_yaw)

                            # Bug A fix: check whether character actually moved
                            if walk_pre_pos is not None:
                                walk_post_pos = self._get_char_position_from_graph()
                                if walk_post_pos is not None:
                                    moved = self._position_distance(walk_pre_pos, walk_post_pos)
                                    target_distance = min(
                                        repeats * self.walk_forward_base_unit_meters,
                                        float(
                                            action_dict.get("magnitude")
                                            or {
                                                "small": self.move_small_magnitude,
                                                "medium": self.move_medium_magnitude,
                                                "large": self.move_large_magnitude,
                                            }.get(
                                                str(action_dict.get("granularity") or "small").lower(),
                                                self.move_small_magnitude,
                                            )
                                        ),
                                    )
                                    atomic_shortfall = (
                                        self.atomic_walkforward_enabled
                                        and repeats > 1
                                        and moved
                                        < max(
                                            self.stuck_walk_position_threshold,
                                            target_distance - self.atomic_walkforward_tolerance_meters,
                                        )
                                    )
                                    if moved < self.stuck_walk_position_threshold:
                                        if self.atomic_walkforward_enabled:
                                            self._rollback_character_to_walk_start(
                                                walk_pre_pos,
                                                pre_yaw=walk_pre_yaw,
                                                stage="walkforward-stuck",
                                            )
                                        # Character did not move: treat as stuck
                                        last_action_success = False
                                        error_message = (
                                            "Agent appears stuck — WalkForward did not change position "
                                            "(moved {:.3f} m). Try TurnLeft or TurnRight to find a clear path."
                                            .format(moved)
                                        )
                                        print(f"  ⚠️  {error_message}")
                                    elif atomic_shortfall:
                                        self._rollback_character_to_walk_start(
                                            walk_pre_pos,
                                            pre_yaw=walk_pre_yaw,
                                            stage="walkforward-shortfall",
                                        )
                                        last_action_success = False
                                        error_message = (
                                            "Atomic WalkForward failed — requested {:.2f} m but moved {:.3f} m; "
                                            "restored action start position."
                                        ).format(target_distance, moved)
                                        print(f"  ⚠️  {error_message}")
                                    else:
                                        self._last_char_position = walk_post_pos

                            # Refresh graph after successful action
                            ok2, graph = self._get_environment_graph_with_retry(
                                retries=3, delay_sec=0.3, stage="post-action"
                            )
                            if ok2:
                                self._current_graph = graph

                                #                （  object_type）
                                if action_type == "interaction" and object_type:
                                    target_ids = [
                                        int(m)
                                        for m in re.findall(
                                            r"\((\d+)\)", executed_script_line or ""
                                        )
                                    ]
                                    if target_ids:
                                        self._bind_target_instance(
                                            object_type, target_ids[0]
                                        )
                                    if len(target_ids) > 1 and object2_type:
                                        self._bind_target_instance(
                                            object2_type, target_ids[1]
                                        )

        else:
            error_message = f"Unknown action_type: {action_type}"
            last_action_success = False
            self.action_sequence.append(f"{action_name}(UNKNOWN_TYPE)")

        # Capture frame and build observation
        image_path = self._save_frame(prefix=f"step_{self.step_counter}")
        metadata = self._build_metadata(
            self._current_graph, last_action_success=last_action_success
        )
        text_state = self._generate_text_state(
            self._current_graph, last_action_success=last_action_success
        )

        reward = self._compute_reward(last_action_success, error_message)
        done = self._check_done(metadata)
        if done:
            reward += self.success_reward

        observation = EnvObservation(
            image_path=image_path,
            text_state=text_state,
            reward=reward,
            done=done,
            metadata=metadata,
        )
        return observation, error_message

    def close(self):
        """Close the VirtualHome connection."""
        if hasattr(self, "comm") and self.comm is not None:
            try:
                self.comm.close()
                print("✓ VirtualHome environment closed")
            except Exception:
                pass

    # =========================================================================
    # Script building
    # =========================================================================

    def _node_matches_object_type(
        self, node_id: Optional[int], object_type: str
    ) -> bool:
        """Check whether a graph node id matches the requested object type exactly."""
        if not self._current_graph or node_id is None or not object_type:
            return False
        target = canonicalize_object_type_name(object_type)
        for node in self._current_graph.get("nodes", []):
            if int(node.get("id", -1)) != int(node_id):
                continue
            class_name = canonicalize_object_type_name(node.get("class_name", ""))
            return class_name == target
        return False

    def _get_bound_instance_id(self, object_type: Optional[str]) -> Optional[int]:
        """Return a previously bound instance id for the object type, if still valid."""
        if not object_type:
            return None
        key = canonicalize_object_type_name(object_type)
        if not key:
            return None
        bound_id = self._bound_instances.get(key)
        if bound_id is None:
            return None
        try:
            bound_id = int(bound_id)
        except (TypeError, ValueError):
            return None
        if self._node_matches_object_type(bound_id, object_type):
            return bound_id
        return None

    def _is_object_close_to_char(
        self, object_id: int, char_id: Optional[int]
    ) -> bool:
        """Check whether Unity marks the object CLOSE to the character.

        ``interaction_distance_meters`` is retained as an opt-in legacy/local
        diagnostic gate. The default value is 0, which means VirtualHome's
        Unity-side CLOSE relation is the only interaction-distance criterion.
        """
        if not self._current_graph or char_id is None:
            return False
        try:
            object_id = int(object_id)
            char_id = int(char_id)
        except (TypeError, ValueError):
            return False

        has_close_edge = False
        for edge in self._current_graph.get("edges", []):
            try:
                from_id = int(edge.get("from_id"))
                to_id = int(edge.get("to_id"))
            except (TypeError, ValueError):
                continue
            if from_id == char_id and to_id == object_id and edge.get("relation_type") == "CLOSE":
                has_close_edge = True
                break
        if not has_close_edge:
            return False

        max_distance = float(getattr(self, "interaction_distance_meters", 0.0))
        if max_distance <= 0:
            return True

        distance = self._get_object_surface_distance_to_char(object_id, char_id)
        if distance is None:
            return False
        return distance <= max_distance

    def _get_node_by_id(self, node_id: Optional[int]) -> Optional[Dict[str, Any]]:
        """Return a graph node by id from the current snapshot."""
        if node_id is None or not self._current_graph:
            return None
        try:
            node_id = int(node_id)
        except (TypeError, ValueError):
            return None
        for node in self._current_graph.get("nodes", []):
            try:
                if int(node.get("id", -1)) == node_id:
                    return node
            except (TypeError, ValueError):
                continue
        return None

    def _node_bbox_center_size(
        self, node: Dict[str, Any]
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        """Extract world-space bounding-box center and size from a node."""
        bbox = node.get("bounding_box") or {}
        center = bbox.get("center")
        size = bbox.get("size")
        if (
            isinstance(center, (list, tuple))
            and isinstance(size, (list, tuple))
            and len(center) >= 3
            and len(size) >= 3
        ):
            try:
                return (
                    (float(center[0]), float(center[1]), float(center[2])),
                    (float(size[0]), float(size[1]), float(size[2])),
                )
            except (TypeError, ValueError):
                return None, None
        return None, None

    def _horizontal_distance_to_node_surface(
        self, point: List[float], node: Dict[str, Any]
    ) -> Optional[float]:
        """Horizontal distance from a point to a node's bbox surface, with point fallback."""
        center, size = self._node_bbox_center_size(node)
        if center is not None and size is not None:
            import math

            dx = max(abs(float(point[0]) - center[0]) - max(size[0], 0.0) / 2.0, 0.0)
            dz = max(abs(float(point[2]) - center[2]) - max(size[2], 0.0) / 2.0, 0.0)
            return math.hypot(dx, dz)

        obj_pos = self._node_position(node)
        if obj_pos is None:
            return None
        return self._position_distance(list(point), list(obj_pos))

    def _get_object_surface_distance_to_char(
        self, object_id: int, char_id: Optional[int]
    ) -> Optional[float]:
        """Return horizontal distance from character position to object bbox surface."""
        if not self._current_graph or char_id is None:
            return None
        try:
            object_id = int(object_id)
            char_id = int(char_id)
        except (TypeError, ValueError):
            return None

        char_node = self._get_node_by_id(char_id)
        obj_node = self._get_node_by_id(object_id)
        char_pos = self._node_position(char_node) if char_node else None
        if obj_node is None or char_pos is None:
            return None
        return self._horizontal_distance_to_node_surface(list(char_pos), obj_node)

    def _get_object_distance_to_char(
        self, object_id: int, char_id: Optional[int]
    ) -> Optional[float]:
        """Backward-compatible alias for surface distance."""
        return self._get_object_surface_distance_to_char(object_id, char_id)

    def _point_inside_node_bbox(
        self,
        point: List[float],
        node: Dict[str, Any],
        *,
        shrink_meters: Optional[float] = None,
    ) -> bool:
        """Return True when a point is inside a node bbox after optional shrink."""
        center, size = self._node_bbox_center_size(node)
        if center is None or size is None:
            return False
        shrink = max(0.0, float(
            self.collision_bbox_shrink_meters if shrink_meters is None else shrink_meters
        ))
        half_x = max(size[0] / 2.0 - shrink, 0.0)
        half_y = max(size[1] / 2.0 - shrink, 0.0)
        half_z = max(size[2] / 2.0 - shrink, 0.0)
        return (
            abs(float(point[0]) - center[0]) <= half_x
            and abs(float(point[1]) - center[1]) <= half_y
            and abs(float(point[2]) - center[2]) <= half_z
        )

    def _is_collision_object_node(self, node: Dict[str, Any]) -> bool:
        """Return whether a graph node should block first-person walking."""
        if not self._is_scene_object_node(node):
            return False
        center, size = self._node_bbox_center_size(node)
        if center is None or size is None:
            return False
        if max(float(size[0]), float(size[1]), float(size[2])) <= 0:
            return False
        return True

    def _camera_probe_position(
        self,
        char_pos: List[float],
        yaw: Optional[float],
    ) -> List[float]:
        """Approximate active first-person camera world position from char pose."""
        import math

        cam_x, cam_y, cam_z = _VH_FP_CAMERA_OFFSET
        if yaw is None:
            yaw = self._char_yaw_rad if self._char_yaw_rad is not None else self._get_char_yaw()
        if yaw is None:
            yaw = 0.0
        world_x = float(char_pos[0]) + float(cam_x) * math.cos(yaw) + float(cam_z) * math.sin(yaw)
        world_z = float(char_pos[2]) - float(cam_x) * math.sin(yaw) + float(cam_z) * math.cos(yaw)
        return [world_x, float(char_pos[1]) + float(cam_y), world_z]

    def _find_collision_intrusion(
        self,
        graph: Optional[Dict[str, Any]] = None,
        *,
        char_id: Optional[int] = None,
        char_pos: Optional[List[float]] = None,
        yaw: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find an object bbox containing the character body/camera probe."""
        if not self.collision_rollback_enabled:
            return None
        graph = graph or self._current_graph
        if not graph:
            return None
        if char_id is None:
            char_id = self._char_ids[0] if self._char_ids else None
        if char_id is None:
            return None

        if char_pos is None:
            char_node = self._get_node_by_id(char_id)
            node_pos = self._node_position(char_node) if char_node else None
            if node_pos is None:
                return None
            char_pos = list(node_pos)

        probes = [
            ("character", list(char_pos)),
            ("camera", self._camera_probe_position(list(char_pos), yaw)),
        ]
        for node in graph.get("nodes", []):
            if not self._is_collision_object_node(node):
                continue
            for probe_name, probe_pos in probes:
                if self._point_inside_node_bbox(probe_pos, node):
                    return {
                        "object_id": int(node.get("id")),
                        "object_type": node.get("class_name", "?"),
                        "probe": probe_name,
                    }
        return None

    def rollback_if_collision_intrusion(
        self,
        pre_position: Optional[List[float]],
        *,
        pre_yaw: Optional[float] = None,
        stage: str = "walk",
    ) -> Tuple[bool, Optional[str]]:
        """Rollback to pre_position if the current char/camera is inside an object bbox."""
        if not self.collision_rollback_enabled or pre_position is None:
            return False, None

        ok, graph = self._get_environment_graph_with_retry(
            retries=3, delay_sec=0.2, stage=f"{stage}-collision-check"
        )
        if not ok or not graph:
            return False, None
        self._current_graph = graph
        self._char_ids = find_character_ids(graph)
        char_id = self._char_ids[0] if self._char_ids else None
        if char_id is None:
            return False, None

        char_node = self._get_node_by_id(char_id)
        char_pos = self._node_position(char_node) if char_node else None
        if char_pos is None:
            return False, None
        yaw = self._char_yaw_rad if self._char_yaw_rad is not None else self._get_char_yaw()
        intrusion = self._find_collision_intrusion(
            graph,
            char_id=char_id,
            char_pos=list(char_pos),
            yaw=yaw,
        )
        if not intrusion:
            return False, None

        try:
            self.comm.move_character(0, pre_position)
        except Exception as exc:
            return (
                False,
                f"Collision intrusion detected but rollback failed: {exc}",
            )

        if pre_yaw is not None:
            self._char_yaw_rad = pre_yaw
            try:
                self._restore_yaw_after_walk(pre_yaw)
            except Exception:
                pass

        ok2, graph2 = self._get_environment_graph_with_retry(
            retries=2, delay_sec=0.2, stage=f"{stage}-collision-rollback"
        )
        if ok2 and graph2:
            self._current_graph = graph2
            self._char_ids = find_character_ids(graph2)

        obj_type = intrusion.get("object_type", "?")
        obj_id = intrusion.get("object_id", "?")
        probe = intrusion.get("probe", "?")
        return (
            True,
            f"WalkForward collision rollback — {probe} entered {obj_type}({obj_id}) bbox; "
            "restored previous position.",
        )

    def _rollback_character_to_walk_start(
        self,
        pre_position: Optional[List[float]],
        *,
        pre_yaw: Optional[float] = None,
        stage: str = "walkforward",
    ) -> bool:
        """Rollback a compound WalkForward action to its starting pose."""
        if pre_position is None:
            return False

        try:
            self.comm.move_character(0, pre_position)
        except Exception:
            return False

        if pre_yaw is not None:
            self._char_yaw_rad = pre_yaw
            try:
                self._restore_yaw_after_walk(pre_yaw)
            except Exception:
                pass

        ok, graph = self._get_environment_graph_with_retry(
            retries=2, delay_sec=0.2, stage=f"{stage}-atomic-rollback"
        )
        if ok and graph:
            self._current_graph = graph
            self._char_ids = find_character_ids(graph)
        return True

    def _get_node_class_name(self, node_id: Optional[int]) -> Optional[str]:
        """Return class_name for node_id from current graph, if available."""
        if node_id is None or not self._current_graph:
            return None
        try:
            node_id = int(node_id)
        except (TypeError, ValueError):
            return None
        for node in self._current_graph.get("nodes", []):
            if int(node.get("id", -1)) == node_id:
                class_name = str(node.get("class_name", "")).strip().lower()
                return class_name or None
        return None

    def _resolve_target_object_id(
        self,
        object_type: Optional[str],
        *,
        char_id: Optional[int],
        explicit_id: Optional[int] = None,
        require_close: bool = False,
        require_visible: bool = False,
        visible_ids: Optional[set] = None,
    ) -> Optional[int]:
        """Resolve an action target id using explicit binding, replay binding, then scene lookup."""
        if not object_type or not self._current_graph:
            return None
        object_type = canonicalize_object_type_name(object_type)
        if not object_type:
            return None

        try:
            explicit_candidate = (
                int(explicit_id)
                if explicit_id is not None and explicit_id != ""
                else None
            )
        except (TypeError, ValueError):
            explicit_candidate = None

        def _candidate_held_by_char(candidate: Optional[int]) -> bool:
            if candidate is None or not self._current_graph or char_id is None:
                return False
            for edge in self._current_graph.get("edges", []):
                if (
                    edge.get("from_id") == char_id
                    and edge.get("to_id") == candidate
                    and edge.get("relation_type") in ("HOLDS_RH", "HOLDS_LH")
                ):
                    return True
            return False

        def _candidate_close_allowed(candidate: Optional[int]) -> bool:
            if candidate is None:
                return False
            if _candidate_held_by_char(candidate) or self._is_object_held(candidate):
                return True
            if require_close and not self._is_object_close_to_char(candidate, char_id):
                return False
            return True

        def _candidate_visible_allowed(candidate: Optional[int]) -> bool:
            if not _candidate_close_allowed(candidate):
                return False
            if require_visible and visible_ids and candidate not in visible_ids:
                return False
            return True

        def _iter_matching_object_ids() -> List[int]:
            ids: List[int] = []
            for node in self._current_graph.get("nodes", []):
                class_name = canonicalize_object_type_name(node.get("class_name", ""))
                if class_name != object_type:
                    continue
                try:
                    ids.append(int(node.get("id")))
                except (TypeError, ValueError):
                    continue
            if require_close and char_id is not None:
                def _distance_key(candidate: int) -> float:
                    distance = self._get_object_distance_to_char(candidate, char_id)
                    return distance if distance is not None else float("inf")

                ids.sort(
                    key=_distance_key
                )
            return ids

        if explicit_candidate is not None and self._node_matches_object_type(
            explicit_candidate, object_type
        ):
            return explicit_candidate if _candidate_close_allowed(explicit_candidate) else None

        bound_candidate = self._get_bound_instance_id(object_type)
        if bound_candidate is not None:
            if self._node_matches_object_type(bound_candidate, object_type):
                return bound_candidate if _candidate_close_allowed(bound_candidate) else None

        if require_close or require_visible:
            candidates = [
                candidate
                for candidate in _iter_matching_object_ids()
                if _candidate_close_allowed(candidate)
            ]
            for candidate in candidates:
                if _candidate_visible_allowed(candidate):
                    return candidate
            if candidates:
                return candidates[0]

        if require_close:
            return None

        return find_object_id_in_graph(
            self._current_graph,
            object_type,
            prefer_close=True,
            char_id=char_id,
            require_close=False,
        )

    def _build_script_line(
        self,
        action_name: str,
        object_type: Optional[str],
        object2_type: Optional[str] = None,
        char_index: int = 0,
        action_type: Optional[str] = None,
        object_id: Optional[int] = None,
        object2_id: Optional[int] = None,
        visible_ids: Optional[set] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Convert action parameters to a VirtualHome script string.

        Returns:
            (script_line, error_message). On success, error_message is None.
        """
        vh_action = ACTION_NAME_MAP.get(action_name, action_name)
        char_str = f"<char{char_index}>"

        # ── Navigation without object ─────────────────────────────────────
        if (
            vh_action in VH_NAVIGATION_NO_OBJECT
            or action_name in VH_NAVIGATION_NO_OBJECT
        ):
            return f"{char_str} [{vh_action}]", None

        # ── Actions where object is optional ─────────────────────────────
        if vh_action == "StandUp" and not object_type:
            return f"{char_str} [{vh_action}]", None

        # ── Object required from here ──────────────────────────────────────
        if not object_type:
            return None, f"'{action_name}' requires an object_type"
        object_type = canonicalize_object_type_name(object_type)
        if object2_type:
            object2_type = canonicalize_object_type_name(object2_type)

        char_id = self._char_ids[0] if self._char_ids else None
        # VirtualHome interactions are strict everywhere: action execution is
        # valid only for objects that are both CLOSE and visually visible.
        # The legacy require_* attributes are kept for compatibility/diagnostics
        # but no longer relax interaction prerequisites.
        require_close = bool(action_type == "interaction")
        require_visible = bool(action_type == "interaction")
        if require_visible and visible_ids is None:
            visible_ids = self._get_visible_object_ids()

        # Find primary object ID
        obj_id = self._resolve_target_object_id(
            object_type,
            char_id=char_id,
            explicit_id=object_id,
            require_close=require_close,
            require_visible=require_visible,
            visible_ids=visible_ids,
        )
        if obj_id is None:
            if require_close:
                max_distance = float(getattr(self, "interaction_distance_meters", 0.0))
                if max_distance <= 0:
                    return (
                        None,
                        f"No Unity CLOSE '{object_type}' target found; "
                        f"approach target before '{action_name}'",
                    )
                return (
                    None,
                    f"No nearby '{object_type}' found within {max_distance:.2f}m of object surface; "
                    f"approach target before '{action_name}'",
                )
            return None, f"Object '{object_type}' not found in scene"

        obj_type_for_script = self._get_node_class_name(obj_id) or object_type
        script = f"{char_str} [{vh_action}] <{obj_type_for_script}> ({obj_id})"

        # ── Two-object actions ─────────────────────────────────────────────
        if vh_action in VH_INTERACTION_TWO_OBJECTS and object2_type:
            obj2_id = self._resolve_target_object_id(
                object2_type,
                char_id=char_id,
                explicit_id=object2_id,
                require_close=require_close,
                require_visible=require_visible,
                visible_ids=visible_ids,
            )
            if obj2_id is None:
                if require_close:
                    max_distance = float(getattr(self, "interaction_distance_meters", 0.0))
                    if max_distance <= 0:
                        return (
                            None,
                            f"No Unity CLOSE '{object2_type}' target found; "
                            f"approach target before '{action_name}'",
                        )
                    return (
                        None,
                        f"No nearby '{object2_type}' found within {max_distance:.2f}m of object surface; "
                        f"approach target before '{action_name}'",
                    )
                return None, f"Target '{object2_type}' not found in scene"
            obj2_type_for_script = self._get_node_class_name(obj2_id) or object2_type
            script += f" <{obj2_type_for_script}> ({obj2_id})"

        return script, None

    # =========================================================================
    # Metadata / state helpers
    # =========================================================================

    @staticmethod
    def _is_scene_object_node(node: Dict[str, Any]) -> bool:
        """Return True for regular scene objects (exclude rooms/characters/structures)."""
        cat = str(node.get("category", "") or "")
        cn = canonicalize_object_type_name(node.get("class_name", ""))
        if cat in {"Characters", "Rooms", "Floors", "Walls", "Ceilings"}:
            return False
        if not cn:
            return False
        if cn.startswith("character"):
            return False
        if cn in {"floor", "wall", "ceiling"}:
            return False
        return True

    def _collect_scene_object_type_sets(
        self, graph: Optional[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Collect (interactable_types, all_scene_types) from a graph snapshot."""
        if not graph:
            return [], []

        all_types = set()
        interactable_types = set()

        # Property-driven interactability hints from VH graph.
        interactable_properties = {
            "GRABBABLE",
            "CAN_OPEN",
            "HAS_SWITCH",
            "SURFACES",
            "CONTAINERS",
            "RECIPIENT",
            "SITTABLE",
            "DRINKABLE",
            "EATABLE",
            "READABLE",
            "MOVABLE",
            "CUTTABLE",
        }

        for node in graph.get("nodes", []):
            if not self._is_scene_object_node(node):
                continue
            obj_type = canonicalize_object_type_name(node.get("class_name", ""))
            if not obj_type:
                continue
            all_types.add(obj_type)

            props = {str(p).upper() for p in (node.get("properties") or []) if p}
            if props & interactable_properties:
                interactable_types.add(obj_type)

        # Metadata-driven fallback for capability tags.
        if not interactable_types:
            char_id = self._char_ids[0] if self._char_ids else None
            objects, _ = build_object_metadata(graph, char_id=char_id)
            capability_keys = (
                "pickupable",
                "openable",
                "toggleable",
                "receptacle",
                "sliceable",
                "drinkable",
                "eatable",
                "readable",
                "movable",
            )
            for obj in objects:
                obj_type = canonicalize_object_type_name(obj.get("objectType", ""))
                if not obj_type:
                    continue
                if any(bool(obj.get(k)) for k in capability_keys):
                    interactable_types.add(obj_type)
                all_types.add(obj_type)

        # Ensure task targets are represented in prompt vocabulary.
        for tgt in self.target_object_types or []:
            canon = canonicalize_object_type_name(tgt)
            if canon:
                all_types.add(canon)
                interactable_types.add(canon)

        if not interactable_types:
            interactable_types = set(all_types)

        return sorted(interactable_types), sorted(all_types)

    def _refresh_scene_object_type_cache(
        self, graph: Optional[Dict[str, Any]] = None
    ) -> None:
        """Refresh cached scene object tokens used by prompt injection."""
        interactable, all_types = self._collect_scene_object_type_sets(
            graph if graph is not None else self._current_graph
        )
        self._scene_interactable_object_types = interactable
        self._scene_all_object_types = all_types

    def get_scene_interactable_object_types(self, refresh: bool = False) -> List[str]:
        """Get scene interactable object type tokens for prompt guidance."""
        if refresh or not self._scene_interactable_object_types:
            graph = self._current_graph
            if graph is None:
                ok, fetched = self._get_environment_graph_with_retry(
                    retries=3, delay_sec=0.5, stage="prompt-objects"
                )
                if ok:
                    graph = fetched
                    self._current_graph = fetched
            self._refresh_scene_object_type_cache(graph)
        return list(self._scene_interactable_object_types)

    def _build_metadata(
        self, graph: Optional[Dict[str, Any]], last_action_success: bool = True
    ) -> Dict[str, Any]:
        """Convert VirtualHome graph to an AI2THOR-compatible metadata dict.

        This allows evaluators/base.py (MultiConditionEvaluator) to work unchanged
        because it reads metadata["objects"] and metadata["inventoryObjects"].

        Extra VH-specific keys are added under 'vh_graph' and 'character_room'.
        """
        if not graph:
            return {
                "objects": [],
                "inventoryObjects": [],
                "agent": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
                "lastActionSuccess": last_action_success,
                "sceneName": f"VHScene{self.scene}",
                "vh_graph": None,
                "character_room": "unknown",
            }

        char_id = self._char_ids[0] if self._char_ids else None
        objects, inventory = build_object_metadata(graph, char_id=char_id)

        # Determine the room the character is currently in
        nodes_by_id = {n["id"]: n for n in graph.get("nodes", [])}
        char_room = "unknown"
        if char_id is not None:
            for edge in graph.get("edges", []):
                if edge["from_id"] == char_id and edge["relation_type"] == "INSIDE":
                    room_node = nodes_by_id.get(edge["to_id"], {})
                    if room_node.get("category") == "Rooms":
                        char_room = room_node.get("class_name", "unknown")
                        break

        return {
            "objects": objects,
            "inventoryObjects": inventory,
            "agent": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
            "lastActionSuccess": last_action_success,
            "sceneName": f"VHScene{self.scene}",
            "vh_graph": graph,
            "character_room": char_room,
            "bound_instances": dict(self._bound_instances),
        }

    def _bind_target_instance(self, object_type: str, object_id: int):
        """Bind successful interaction instance for an object_type.

        Policy:
            - latest (default): always update to the most recently interacted instance
            - first: keep the first bound instance and ignore later ones
        """
        if not object_type:
            return
        key = canonicalize_object_type_name(object_type)
        if not key:
            return
        new_id = int(object_id)
        existing_id = self._bound_instances.get(key)

        if existing_id is None:
            self._bound_instances[key] = new_id
            return

        if self.bound_instance_update_policy == "latest" and existing_id != new_id:
            self._bound_instances[key] = new_id

    def _generate_text_state(
        self,
        graph: Optional[Dict[str, Any]],
        last_action_success: bool = True,
    ) -> str:
        """Generate text state description for the VLM.

        In first_person mode: brief action feedback + task hint (same as AI2-THOR).
        In omniscient mode: full object inventory from the graph.
        """
        if self.text_state_mode == "first_person":
            lines = [
                " First-Person Mode Please rely primarily on the image to understand the environment.",
                f"Task: {self.task_description}",
                f"Step: {self.step_counter}",
                f"Last action success: {'Yes' if last_action_success else 'No'}",
            ]
            if not last_action_success:
                lines.append(
                    "Hint: The last action failed. "
                    "If WalkForward keeps failing or the view is unchanged, "
                    "try TurnLeft or TurnRight to reorient, then walk again. "
                    "Do NOT repeat the same action more than 2-3 times in a row."
                )
            return "\n".join(lines)

        # Omniscient mode: use utility function
        if not graph:
            return "Environment graph not available."
        char_id = self._char_ids[0] if self._char_ids else None
        return generate_text_state(
            graph, task_description=self.task_description, char_id=char_id
        )

    # =========================================================================
    # Evaluation helpers (mirrors AI2ThorEnvWrapper._check_done)
    # =========================================================================

    def _check_done(self, metadata: dict) -> bool:
        """Check if the task success condition is met using the metadata dict.

        Mirrors AI2ThorEnvWrapper._check_done logic; works on the AI2THOR-compatible
        metadata produced by _build_metadata().
        """
        if self.success_evaluator is not None:
            try:
                score = self.success_evaluator.evaluate(self, metadata)
                done = score >= 1.0
                if done:
                    cond_count = len(self.success_conditions) or (
                        1 if self.success_condition else 0
                    )
                    if cond_count > 1:
                        print(
                            f"  ✓ Multi-condition met ({self.success_logic}): {cond_count} conditions"
                        )
                return done
            except Exception as exc:
                print(f"⚠️  Evaluator-based done check failed, fallback to predicate mode: {exc}")

        if not self.target_object_types or not self.success_predicate:
            return False

        # object_in_receptacle
        if (
            self.success_condition
            and self.success_condition.get("type") == "object_in_receptacle"
        ):
            obj_type = canonicalize_object_type_name(
                self.success_condition.get("object_type", "")
            )
            rec_type = canonicalize_object_type_name(
                self.success_condition.get("receptacle_type", "")
            )
            for obj in metadata.get("objects", []):
                if (
                    canonicalize_object_type_name(obj.get("objectType", ""))
                    == obj_type
                ):
                    if self.success_predicate(obj):
                        print(f"  ✓ Condition met: {obj_type} is in {rec_type}")
                        return True
            return False

        # object_in_hand
        if (
            self.success_condition
            and self.success_condition.get("type") == "object_in_hand"
        ):
            tgt = canonicalize_object_type_name(
                self.success_condition.get("object_type", "")
            )
            for item in metadata.get("inventoryObjects", []):
                if canonicalize_object_type_name(item.get("objectType", "")) == tgt:
                    print(f"  ✓ Condition met: {tgt} is in hand")
                    return True
            return False

        # object_state (default)
        for obj in metadata.get("objects", []):
            if canonicalize_object_type_name(obj.get("objectType", "")) in set(
                canonicalize_object_type_name(t) for t in self.target_object_types
            ):
                if self.success_predicate(obj):
                    print(f"  ✓ Condition met: {obj['objectType']} satisfies predicate")
                    return True
        return False

    # =========================================================================
    # Image capture
    # =========================================================================

    def _get_active_camera_id(self) -> int:
        """Return the currently selected first-person camera id."""
        if self._char_camera_ids and 0 <= self._pitch_idx < len(self._char_camera_ids):
            return self._char_camera_ids[self._pitch_idx]
        if self._char_camera_ids:
            return self._char_camera_ids[0]
        return 0

    def _grab_active_camera_frame_for_settle(self) -> None:
        """Mirror interact worker's post-action camera grab without saving a file."""
        camera_id = self._get_active_camera_id()
        try:
            self.comm.camera_image(
                [camera_id],
                mode="normal",
                image_width=self.width,
                image_height=self.height,
            )
        except Exception:
            pass

    def _settle_before_interaction_retry(self) -> None:
        """Refresh graph/camera once before retrying a Unity-transient interaction."""
        import time as _time

        ok, graph = self._get_environment_graph_with_retry(
            retries=2,
            delay_sec=0.25,
            stage="interaction-retry-pre",
        )
        if ok and graph:
            self._current_graph = graph
            self._char_ids = find_character_ids(graph)
        self._grab_active_camera_frame_for_settle()
        _time.sleep(0.25)

    @staticmethod
    def _is_transient_vh_executor_error(message: Any) -> bool:
        """Return True for Unity executor failures that are safe to retry once."""
        text = str(message or "").lower()
        return (
            "scriptexcutor" in text
            and "execution_general" in text
            and "script is impossible to execute" in text
        )

    def _can_retry_held_object_placement(self, target_ids: List[int]) -> bool:
        """Retry PutBack/PutIn only if strict preconditions still hold."""
        if len(target_ids) < 2 or not self._current_graph or not self._char_ids:
            return False
        held_id, dest_id = int(target_ids[0]), int(target_ids[1])
        char_id = self._char_ids[0]
        return self._is_object_held(held_id) and self._is_object_close_to_char(
            dest_id,
            char_id,
        )

    def _node_position(
        self, node: Dict[str, Any]
    ) -> Optional[Tuple[float, float, float]]:
        """Extract world coordinates from a VirtualHome graph node."""
        transform = node.get("obj_transform", {})
        position = transform.get("position")
        if position is None:
            return None
        if isinstance(position, dict):
            if all(k in position for k in ("x", "y", "z")):
                return (
                    float(position["x"]),
                    float(position["y"]),
                    float(position["z"]),
                )
            return None
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            return float(position[0]), float(position[1]), float(position[2])
        return None

    def _get_char_yaw(self) -> Optional[float]:
        """Read the current character yaw from the latest graph snapshot."""
        import math

        if not self._current_graph or not self._char_ids:
            return None
        char_id = self._char_ids[0]
        for node in self._current_graph.get("nodes", []):
            if node.get("id") != char_id:
                continue
            rot = node.get("obj_transform", {}).get("rotation")
            if rot and len(rot) >= 4:
                return math.atan2(
                    2.0 * (rot[3] * rot[1] + rot[0] * rot[2]),
                    1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
                )
        return None

    def _restore_yaw_after_walk(self, target_yaw: float):
        """Match interact mode by compensating VH walk-induced yaw drift."""
        import math

        if self._skip_first_walk_yaw_restore:
            self._skip_first_walk_yaw_restore = False
            return

        # Use live graph yaw to avoid stale post-action rotation snapshots.
        current_yaw = self._get_live_char_yaw()
        if current_yaw is None:
            return

        delta = target_yaw - current_yaw
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi

        delta_deg = abs(math.degrees(delta))
        min_delta_deg = float(self.walk_yaw_restore_min_delta_degrees)
        if delta_deg < min_delta_deg:
            return

        action = "[TurnRight]" if delta > 0 else "[TurnLeft]"
        step_deg = self.vh_turn_step_degrees if self.vh_turn_step_degrees > 0 else 30.0
        n_turns = max(1, int(round(delta_deg / step_deg)))

        script_lines = [f"<char0> {action}" for _ in range(n_turns)]
        try:
            ok = True
            msg = None
            for script_line in script_lines:
                ok, msg = self.comm.render_script(
                    [script_line],
                    skip_animation=False,
                    image_synthesis=[],
                    recording=False,
                    time_scale=VH_INTERACT_TIME_SCALE,
                )
                if not ok:
                    break
            if not ok:
                print(f"  ⚠️  _restore_yaw_after_walk failed: {msg}")
                return
        except Exception as exc:
            print(f"  ⚠️  _restore_yaw_after_walk exception: {exc}")
            return

        self._char_yaw_rad = target_yaw
        ok, graph = self._get_environment_graph_with_retry(
            retries=3, delay_sec=0.3, stage="post-walk-yaw"
        )
        if ok and graph:
            self._current_graph = graph

    def _restore_yaw_after_interaction(
        self,
        target_rotation,
        *,
        min_delta_deg: float = 20.0,
    ):
        """Conservatively correct interaction-induced yaw drift.

        Some Grab animations can rotate character yaw by one or more 30-degree steps.
        Only correct when drift is clearly larger than graph lag noise.
        """
        import math

        target_yaw = self._quat_to_yaw_rad(target_rotation)
        if target_yaw is None:
            return

        # Interaction animation may update rotation asynchronously; always read live yaw.
        current_yaw = self._get_live_char_yaw()
        if current_yaw is None:
            return

        delta = target_yaw - current_yaw
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi

        delta_deg = abs(math.degrees(delta))
        if delta_deg < float(min_delta_deg):
            self._sync_yaw_cache_from_live(fallback_yaw=current_yaw)
            return

        if self._try_restore_rotation_exact(
            target_rotation,
            tolerance_deg=float(self.grab_exact_restore_tolerance_degrees),
        ):
            self._sync_yaw_cache_from_live(fallback_yaw=target_yaw)
            return

        live_yaw = self._restore_yaw_closed_loop(
            target_yaw,
            tolerance_deg=float(self.grab_yaw_target_tolerance_degrees),
            max_turns=max(1, int(self.grab_yaw_closed_loop_max_turns)),
        )
        if live_yaw is not None:
            self._char_yaw_rad = live_yaw
            residual_deg = abs(
                math.degrees(self._normalize_angle_rad(target_yaw - live_yaw))
            )
            if residual_deg >= float(self.grab_yaw_residual_warn_degrees):
                print(
                    f"  ⚠️  Grab yaw residual={residual_deg:.2f}° "
                    f"(target={math.degrees(target_yaw):.2f}°, live={math.degrees(live_yaw):.2f}°)"
                )
        else:
            self._char_yaw_rad = target_yaw

    def _sync_yaw_cache_from_live(self, fallback_yaw: Optional[float] = None):
        """Sync client yaw cache from live graph, fallback to provided yaw if needed."""
        live_yaw = self._get_live_char_yaw()
        if live_yaw is not None:
            self._char_yaw_rad = live_yaw
            return
        if fallback_yaw is not None:
            self._char_yaw_rad = fallback_yaw

    def _reset_active_pitch_camera_pose(self):
        """Refresh active first-person pitch camera pose to reduce animation-induced drift."""
        if not self.grab_reset_camera_pose or not self._char_camera_ids:
            return
        if self._pitch_idx < 0 or self._pitch_idx >= len(_VH_FP_PITCH_CAMERAS):
            return

        pitch_deg, cam_name = _VH_FP_PITCH_CAMERAS[self._pitch_idx]
        try:
            self.comm.update_character_camera(
                position=_VH_FP_CAMERA_OFFSET,
                rotation=[pitch_deg, 0, 0],
                field_view=60,
                name=cam_name,
            )
        except Exception:
            pass
        ok, graph = self._get_environment_graph_with_retry(
            retries=3, delay_sec=0.2, stage="post-interaction-yaw"
        )
        if ok and graph:
            self._current_graph = graph

    def _restore_yaw_closed_loop(
        self,
        target_yaw: float,
        *,
        tolerance_deg: float,
        max_turns: int,
    ) -> Optional[float]:
        """Closed-loop yaw recovery with one-step turns and direction self-correction."""
        import math
        import time as _time

        def _delta_deg(curr: float) -> float:
            return abs(math.degrees(self._normalize_angle_rad(target_yaw - curr)))

        current_yaw = self._get_live_char_yaw()
        if current_yaw is None:
            return None

        for _ in range(max_turns):
            delta = self._normalize_angle_rad(target_yaw - current_yaw)
            abs_delta = abs(math.degrees(delta))
            if abs_delta <= float(tolerance_deg):
                return current_yaw

            prefer_action = "[TurnRight]" if delta > 0 else "[TurnLeft]"
            try:
                ok, _msg = self.comm.render_script(
                    [f"<char0> {prefer_action}"],
                    skip_animation=False,
                    image_synthesis=[],
                    recording=False,
                    time_scale=5.0,
                )
            except Exception:
                return current_yaw

            if not ok:
                return current_yaw

            _time.sleep(0.05)
            yaw_after = self._get_live_char_yaw()
            if yaw_after is None:
                continue

            new_abs_delta = _delta_deg(yaw_after)
            if new_abs_delta > abs_delta + 1.0:
                reverse_action = (
                    "[TurnLeft]" if prefer_action == "[TurnRight]" else "[TurnRight]"
                )
                try:
                    ok2, _msg2 = self.comm.render_script(
                        [f"<char0> {reverse_action}"],
                        skip_animation=False,
                        image_synthesis=[],
                        recording=False,
                        time_scale=5.0,
                    )
                    if ok2:
                        _time.sleep(0.05)
                        yaw_rev = self._get_live_char_yaw()
                        if yaw_rev is not None and _delta_deg(yaw_rev) < new_abs_delta:
                            yaw_after = yaw_rev
                except Exception:
                    pass

            current_yaw = yaw_after

        return current_yaw

    def _filter_node_ids_in_camera_view(
        self,
        h_fov_deg: float = 60.0,
        v_fov_deg: float = 60.0,
    ) -> set:
        """Mirror interact mode's geometric FOV filtering for visibility checks."""
        import math

        if not self._current_graph or not self._char_ids:
            return set()

        char_id = self._char_ids[0]
        char_node = None
        for node in self._current_graph.get("nodes", []):
            if node.get("id") == char_id:
                char_node = node
                break
        if not char_node:
            return set()

        char_pos = self._node_position(char_node)
        char_rot = char_node.get("obj_transform", {}).get("rotation")
        if not char_pos or not char_rot or len(char_rot) < 4:
            return set()

        yaw = self._char_yaw_rad
        if yaw is None:
            yaw = math.atan2(
                2.0 * (char_rot[3] * char_rot[1] + char_rot[0] * char_rot[2]),
                1.0 - 2.0 * (char_rot[1] * char_rot[1] + char_rot[2] * char_rot[2]),
            )

        pitch_deg = _VH_FP_PITCH_CAMERAS[self._pitch_idx][0]
        cam_pitch = -math.radians(float(pitch_deg))
        cx, cy, cz = char_pos
        cam_y = cy + 1.6
        h_half = math.radians(h_fov_deg / 2.0)
        v_half = math.radians(v_fov_deg / 2.0)

        visible_ids = set()
        for node in self._current_graph.get("nodes", []):
            if node.get("id") == char_id:
                continue
            category = node.get("category", "")
            class_name = node.get("class_name", "").lower()
            if category in ("Characters", "Rooms", "Floors", "Walls", "Ceilings"):
                continue
            if class_name.startswith("character") or class_name in (
                "floor",
                "wall",
                "ceiling",
            ):
                continue

            pos = self._node_position(node)
            if not pos:
                continue
            ox, oy, oz = pos
            dx = ox - cx
            dy = oy - cam_y
            dz = oz - cz
            dist_xz = math.hypot(dx, dz)
            if dist_xz < 1e-6:
                continue

            obj_yaw = math.atan2(dx, dz)
            dyaw = obj_yaw - yaw
            while dyaw > math.pi:
                dyaw -= 2 * math.pi
            while dyaw < -math.pi:
                dyaw += 2 * math.pi

            obj_pitch = math.atan2(dy, dist_xz)
            dpitch = obj_pitch - cam_pitch
            if abs(dyaw) <= h_half and abs(dpitch) <= v_half:
                visible_ids.add(node["id"])

        return visible_ids

    def _is_visibility_object_node(self, node_id: int) -> bool:
        """Return whether a node id should count as an object visible to the agent."""
        if not self._current_graph:
            return False
        for node in self._current_graph.get("nodes", []):
            if int(node.get("id", -1)) != int(node_id):
                continue
            category = node.get("category", "")
            class_name = str(node.get("class_name", "")).lower()
            if category in ("Characters", "Rooms", "Floors", "Walls", "Ceilings"):
                return False
            if class_name.startswith("character") or class_name in (
                "floor",
                "wall",
                "ceiling",
            ):
                return False
            return True
        return False

    def _filter_visibility_object_ids(self, ids: set) -> set:
        """Remove room/character/structural ids from a visibility result."""
        return {int(obj_id) for obj_id in ids if self._is_visibility_object_node(int(obj_id))}

    def _get_instance_color_to_id(
        self,
        *,
        suppress_warnings: bool = False,
    ) -> Dict[Tuple[int, int, int], int]:
        """Map seg_inst RGB colors to graph object ids."""
        if self._instance_color_to_id is not None:
            return self._instance_color_to_id

        mapping: Dict[Tuple[int, int, int], int] = {}
        try:
            ok, raw = self.comm.instance_colors()
        except Exception as exc:
            if not suppress_warnings:
                print(f"  ⚠️  instance_colors() exception: {exc}")
            return mapping

        if not ok or not isinstance(raw, dict):
            if not suppress_warnings:
                print(f"  ⚠️  instance_colors() returned ok={ok}, type={type(raw).__name__}")
            return mapping

        for key, value in raw.items():
            try:
                object_id = int(key)
                if not isinstance(value, (list, tuple)) or len(value) < 3:
                    continue
                rgb = tuple(
                    max(0, min(255, int(round(float(channel) * 255))))
                    for channel in value[:3]
                )
            except Exception:
                continue
            mapping[rgb] = object_id

        if mapping:
            self._instance_color_to_id = mapping
        return mapping

    def _get_seg_inst_visible_object_ids(
        self,
        camera_id: int,
        *,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        min_pixels: Optional[int] = None,
        suppress_warnings: bool = False,
    ) -> Tuple[set, Dict[int, int]]:
        """Return ids with visible pixels in the instance-segmentation image."""
        color_to_id = self._get_instance_color_to_id(
            suppress_warnings=suppress_warnings
        )
        if not color_to_id:
            return set(), {}

        width = int(image_width or self.width)
        height = int(image_height or self.height)
        threshold = int(min_pixels or self.visible_seg_pixel_threshold)
        cache_key = (
            int(getattr(self, "step_counter", 0)),
            int(camera_id),
            int(width),
            int(height),
            int(threshold),
        )
        cached = self._seg_visibility_cache.get(cache_key)
        if cached is not None:
            return set(cached[0]), dict(cached[1])

        try:
            ok, images = self.comm.camera_image(
                [camera_id],
                mode="seg_inst",
                image_width=width,
                image_height=height,
            )
        except Exception as exc:
            if not suppress_warnings:
                print(f"  ⚠️  camera_image(seg_inst, camera={camera_id}) exception: {exc}")
            return set(), {}

        if not ok or not images or images[0] is None:
            if not suppress_warnings:
                print(f"  ⚠️  camera_image(seg_inst, camera={camera_id}) returned ok={ok}")
            return set(), {}

        frame = images[0]
        if not isinstance(frame, np.ndarray) or frame.ndim < 3 or frame.shape[2] < 3:
            if not suppress_warnings:
                print(
                    f"  ⚠️  camera_image(seg_inst, camera={camera_id}) returned "
                    f"unexpected frame type/shape: {type(frame).__name__}, "
                    f"{getattr(frame, 'shape', None)}"
                )
            return set(), {}

        # VirtualHome decodes images as BGR; instance_colors uses RGB.
        rgb_frame = frame[:, :, :3][:, :, ::-1].astype(np.uint8)
        colors, counts = np.unique(rgb_frame.reshape(-1, 3), axis=0, return_counts=True)

        visible_ids = set()
        pixel_counts: Dict[int, int] = {}
        for color, count in zip(colors, counts):
            rgb = tuple(int(channel) for channel in color)
            object_id = color_to_id.get(rgb)
            if object_id is None:
                continue
            if int(count) < threshold:
                continue
            if not self._is_visibility_object_node(object_id):
                continue
            visible_ids.add(int(object_id))
            pixel_counts[int(object_id)] = int(count)

        self._seg_visibility_cache[cache_key] = (set(visible_ids), dict(pixel_counts))
        return visible_ids, pixel_counts

    def _get_api_visible_object_ids(
        self,
        camera_id: int,
        *,
        suppress_warnings: bool = False,
    ) -> set:
        """Return visible object ids from Unity get_visible_objects(camera_id)."""
        try:
            ok, result = self.comm.get_visible_objects(camera_id)
            if ok and result:
                return self._filter_visibility_object_ids(
                    self._extract_visible_ids(result)
                )
            if not suppress_warnings:
                print(
                    f"  ⚠️  get_visible_objects(camera={camera_id}): ok={ok}, "
                    f"result_type={type(result).__name__}, result={str(result)[:200]}"
                )
        except Exception as exc:
            if not suppress_warnings:
                print(f"  ⚠️  get_visible_objects(camera={camera_id}) exception: {exc}")
        return set()

    def get_visual_visible_object_ids(
        self,
        camera_id: Optional[int] = None,
        *,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        min_pixels: Optional[int] = None,
        require_all_sources: bool = False,
        suppress_warnings: bool = False,
    ) -> Tuple[set, str]:
        """Return visible ids from the configured source union.

        Source values:
        - ``seg_inst+api``: union of screenshot instance pixels and Unity visible API.
        - ``seg_inst``: ids came from instance-segmentation pixels in the active camera.
        - ``api``: ids came from Unity ``get_visible_objects(camera_id)``.
        - ``fov``: ids came from graph/FOV geometry.
        - ``geometry_fallback``: FOV was used because configured sources failed.
        - ``none``: no visible ids were available.
        """
        cam = int(camera_id if camera_id is not None else self._get_active_camera_id())
        self._last_visibility_camera_id = cam

        configured_sources = list(getattr(self, "visibility_sources", ["seg_inst", "api"]))
        source_ids: Dict[str, set] = {}

        if "seg_inst" in configured_sources:
            seg_ids, _pixel_counts = self._get_seg_inst_visible_object_ids(
                cam,
                image_width=image_width,
                image_height=image_height,
                min_pixels=min_pixels,
                suppress_warnings=suppress_warnings,
            )
            if seg_ids:
                source_ids["seg_inst"] = seg_ids

        if "api" in configured_sources:
            api_ids = self._get_api_visible_object_ids(
                cam,
                suppress_warnings=suppress_warnings,
            )
            if api_ids:
                source_ids["api"] = api_ids

        if "fov" in configured_sources:
            fov_ids = self._filter_visibility_object_ids(
                self._filter_node_ids_in_camera_view()
            )
            if fov_ids:
                source_ids["fov"] = fov_ids

        union_ids = set()
        used_sources: List[str] = []
        for source_name in configured_sources:
            ids = source_ids.get(source_name)
            if not ids:
                continue
            union_ids |= ids
            used_sources.append(source_name)

        if require_all_sources:
            missing_sources = [
                source_name
                for source_name in configured_sources
                if source_name not in source_ids
            ]
            if missing_sources:
                source = "missing:" + "+".join(missing_sources)
                self._last_visibility_source = source
                return set(), source

        if union_ids:
            source = "+".join(used_sources)
            self._last_visibility_source = source
            return union_ids, source

        if require_all_sources:
            self._last_visibility_source = "none"
            return set(), "none"

        geometric_ids = self._filter_visibility_object_ids(
            self._filter_node_ids_in_camera_view()
        )
        if geometric_ids:
            if not suppress_warnings:
                print(
                    f"  ⚠️  visibility source=geometry_fallback "
                    f"(configured sources unavailable: {configured_sources}, camera={cam})"
                )
            self._last_visibility_source = "geometry_fallback"
            return geometric_ids, "geometry_fallback"

        self._last_visibility_source = "none"
        return set(), "none"

    def get_stable_visual_visible_object_ids(
        self,
        camera_id: Optional[int] = None,
        *,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        min_pixels: Optional[int] = None,
        required_ids: Optional[set] = None,
        retries: int = VH_STABLE_VISIBILITY_RETRIES,
        delay_sec: float = VH_STABLE_VISIBILITY_DELAY_SEC,
        require_all_sources: bool = True,
        suppress_warnings: bool = False,
    ) -> Tuple[set, str]:
        """Return visual ids after configured visibility sources stabilize.

        This is the shared strict visibility gate for interact menus, agent
        runtime, and golden replay. By default it requires all configured
        sources (seg_inst+api) to respond, so a transient single-source frame
        does not decide interaction validity.
        """
        import time as _time

        attempts = max(1, int(retries))
        configured_sources = list(
            getattr(self, "visibility_sources", ["seg_inst", "api"])
        )
        configured = set(configured_sources)
        required = {int(obj_id) for obj_id in (required_ids or set())}
        last_visible_ids: set = set()
        last_source = "none"
        last_sources_ready = False

        for attempt in range(attempts):
            visible_ids, source = self.get_visual_visible_object_ids(
                camera_id,
                image_width=image_width,
                image_height=image_height,
                min_pixels=min_pixels,
                require_all_sources=require_all_sources,
                suppress_warnings=suppress_warnings,
            )
            visible_ids = set(visible_ids)
            last_visible_ids = visible_ids
            last_source = source
            used = (
                set(source.split("+"))
                if source and not source.startswith(("missing:", "unstable:"))
                else set()
            )
            sources_ready = not require_all_sources or configured.issubset(used)
            last_sources_ready = sources_ready
            if visible_ids and sources_ready and (
                not required or required.issubset(visible_ids)
            ):
                return set(visible_ids), source
            if attempt < attempts - 1:
                _time.sleep(float(delay_sec))

        if last_visible_ids and last_sources_ready:
            self._last_visibility_source = last_source
            return set(last_visible_ids), last_source

        source = "unstable:" + "+".join(configured_sources)
        self._last_visibility_source = source
        return set(), source

    def _save_frame(self, prefix: str = "frame") -> str:
        """Capture a frame from VirtualHome and save it as a PNG.

        Uses the first available first-person camera, or any camera if none found.
        Falls back gracefully when camera capture fails.

        Returns:
            Absolute path to the saved image, or empty string on failure.
        """
        camera_id = self._get_active_camera_id()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)

        # Try preferred first-person camera first, then fallbacks from known character cameras.
        candidate_ids = [camera_id]
        for cid in self._char_camera_ids:
            if cid not in candidate_ids:
                candidate_ids.append(cid)

        for cid in candidate_ids:
            try:
                ok, images = self.comm.camera_image(
                    [cid],
                    mode="normal",
                    image_width=self.width,
                    image_height=self.height,
                )
                if ok and images and images[0] is not None:
                    frame = images[0]
                    if isinstance(frame, np.ndarray):
                        # camera_image    BGR（cv2.imdecode），PIL    RGB
                        img = Image.fromarray(frame[:, :, ::-1].astype(np.uint8))
                    else:
                        img = Image.open(io.BytesIO(frame))
                    img.save(filepath)
                    return filepath
                print(
                    f"  ⚠️  camera_image returned ok={ok}, images={type(images)}, "
                    f"camera_id={cid}, _char_camera_ids={self._char_camera_ids}"
                )
            except Exception as exc:
                print(f"  ⚠️  camera_image failed (camera {cid}): {exc}")

        # Fallback: save a blank placeholder so downstream code has a valid path
        try:
            placeholder = Image.new(
                "RGB", (self.width, self.height), color=(200, 200, 200)
            )
            placeholder.save(filepath)
            return filepath
        except Exception:
            return ""

    def _get_visible_object_ids(self) -> set:
        """Return IDs visible in the current first-person camera."""
        ids, _source = self.get_stable_visual_visible_object_ids()
        return ids

    def _extract_visible_ids(self, result: Any) -> set:
        """Parse visible object IDs from VirtualHome get_visible_objects result."""
        ids = set()

        def _maybe_int(v: Any) -> Optional[int]:
            """Best-effort conversion for id-like values across VH return variants."""
            if isinstance(v, bool):
                return None
            if isinstance(v, (int, float)):
                try:
                    return int(v)
                except Exception:
                    return None
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return None
                if s.isdigit():
                    return int(s)
                # Accept numeric strings like "427.0"
                try:
                    f = float(s)
                    if f.is_integer():
                        return int(f)
                except Exception:
                    return None
            return None

        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    cand = _maybe_int(item.get("id"))
                    if cand is not None:
                        ids.add(cand)
                    continue
                cand = _maybe_int(item)
                if cand is not None:
                    ids.add(cand)
            return ids

        if isinstance(result, dict):
            #     1：{"1": "character", "427": "tv", ...}
            #     2：{"visible": [1, 427, ...]} /    list   
            for key, value in result.items():
                cand = _maybe_int(key)
                if cand is not None:
                    ids.add(cand)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            cand = _maybe_int(item.get("id"))
                        else:
                            cand = _maybe_int(item)
                        if cand is not None:
                            ids.add(cand)
            return ids

        return ids

    def _find_visible_object_id(
        self, object_type: str, visible_ids: set
    ) -> Optional[int]:
        """Find a visible instance id by object class name."""
        if not self._current_graph or not object_type:
            return None
        target = canonicalize_object_type_name(object_type)
        for node in self._current_graph.get("nodes", []):
            node_id = node.get("id")
            class_name = canonicalize_object_type_name(node.get("class_name", ""))
            if class_name == target and node_id in visible_ids:
                return int(node_id)
        return None

    def _select_primary_camera_id(self, graph: Dict[str, Any], cam_count: int) -> int:
        """Select a stable camera ID that best matches first-person interaction view."""
        char_id = self._char_ids[0] if self._char_ids else None
        close_ids = set()
        if char_id is not None:
            for edge in graph.get("edges", []):
                if (
                    edge.get("from_id") == char_id
                    and edge.get("relation_type") == "CLOSE"
                ):
                    close_ids.add(edge.get("to_id"))

        best_cam = max(0, cam_count - 1)
        best_score = float("-inf")

        for cam_id in range(cam_count):
            try:
                ok, result = self.comm.get_visible_objects(cam_id)
            except Exception:
                continue
            if not ok:
                continue

            ids = self._extract_visible_ids(result)
            if not ids:
                continue

            score = 0.0
            #              ；         character   
            if char_id is not None:
                score += 1000.0 if char_id not in ids else -200.0
            #     CLOSE               
            if close_ids:
                score += 20.0 * len(ids & close_ids)
            #              （      /    ）
            score -= 0.05 * len(ids)

            if score > best_score:
                best_score = score
                best_cam = cam_id

        return best_cam

    def _ensure_first_person_cameras(self) -> List[int]:
        """Register/update dedicated first-person cameras and return their IDs.

        Returns interact-compatible camera IDs: the last N indices from
        camera_count(), where N=len(_VH_FP_PITCH_CAMERAS).
        """
        try:
            _, raw = self.comm.character_cameras()
            existing = _parse_cam_names(raw)

            for pitch_angle, cam_name in _VH_FP_PITCH_CAMERAS:
                if cam_name not in existing:
                    self.comm.add_character_camera(
                        position=_VH_FP_CAMERA_OFFSET,
                        rotation=[pitch_angle, 0, 0],
                        field_view=60,
                        name=cam_name,
                    )
                else:
                    try:
                        self.comm.update_character_camera(
                            position=_VH_FP_CAMERA_OFFSET,
                            rotation=[pitch_angle, 0, 0],
                            field_view=60,
                            name=cam_name,
                        )
                    except Exception:
                        pass

            ok_count, total = self.comm.camera_count()
            if not ok_count or total <= 0:
                return []

            n_pitch = len(_VH_FP_PITCH_CAMERAS)
            if total >= n_pitch:
                return list(range(total - n_pitch, total))
            return [0] * n_pitch
        except Exception as exc:
            print(f"  ⚠️  first-person camera setup failed: {exc}")
            return []

    def _is_object_held(self, obj_id: int) -> bool:
        """Check if the object is currently held by the first character."""
        if not self._current_graph or not self._char_ids:
            return False
        char_id = self._char_ids[0]
        for edge in self._current_graph.get("edges", []):
            if edge.get("from_id") == char_id and edge.get("to_id") == obj_id:
                if edge.get("relation_type") in ("HOLDS_RH", "HOLDS_LH"):
                    return True
        return False

    def _validate_interaction_visibility(
        self,
        script_line: str,
        object_type: Optional[str],
        object2_type: Optional[str],
        visible_ids: Optional[set] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Ensure interaction targets are visible in current camera view."""
        if visible_ids is None:
            visible_ids = self._get_visible_object_ids()
        source = self._last_visibility_source
        if not visible_ids:
            return (
                False,
                "Visibility check failed: no visible objects are currently available "
                f"(visibility_source={source}); refresh the camera view and try again",
            )

        target_ids = [int(m) for m in re.findall(r"\((\d+)\)", script_line)]
        if not target_ids:
            return True, None

        #    ：         ；      
        first_id = target_ids[0]
        if first_id not in visible_ids and not self._is_object_held(first_id):
            visible_ids, source = self.get_stable_visual_visible_object_ids(
                required_ids={first_id},
                retries=VH_STABLE_VISIBILITY_RETRIES,
                delay_sec=VH_STABLE_VISIBILITY_DELAY_SEC,
                require_all_sources=True,
                suppress_warnings=True,
            )
        if first_id not in visible_ids and not self._is_object_held(first_id):
            return (
                False,
                f"Target '{object_type}' is not visible "
                f"(visibility_source={source}); move or rotate before interacting",
            )

        #     （PutIn/PutBack  ）    
        if len(target_ids) > 1 and target_ids[1] not in visible_ids:
            visible_ids, source = self.get_stable_visual_visible_object_ids(
                required_ids={target_ids[1]},
                retries=VH_STABLE_VISIBILITY_RETRIES,
                delay_sec=VH_STABLE_VISIBILITY_DELAY_SEC,
                require_all_sources=True,
                suppress_warnings=True,
            )
        if len(target_ids) > 1 and target_ids[1] not in visible_ids:
            return (
                False,
                f"Second target '{object2_type}' is not visible "
                f"(visibility_source={source}); move or rotate before placing",
            )

        return True, None

    # =========================================================================
    # Reward calculation
    # =========================================================================

    def _compute_reward(
        self, last_action_success: bool, error_message: Optional[str]
    ) -> float:
        """Return a step reward (small bonus/penalty); task completion reward is added by caller."""
        if error_message or not last_action_success:
            return self.step_failure_penalty
        return self.step_success_bonus

    # =========================================================================
    # Error translation
    # =========================================================================

    def _translate_error(
        self, raw_error: str, action_name: str, object_type: Optional[str]
    ) -> str:
        """Convert VirtualHome raw error messages to user-friendly English."""
        err = raw_error.lower()
        obj = object_type or "object"

        if "not found" in err or "no node" in err:
            return f"'{obj}' not found in the scene graph"
        if "can't reach" in err or "not reachable" in err or "not close" in err:
            return f"Cannot reach '{obj}'; move closer first (e.g., WalkForward + TurnLeft/TurnRight)"
        if "not grabbable" in err or "can't grab" in err:
            return f"'{obj}' is not pickupable"
        if "not openable" in err or "can't open" in err:
            return f"'{obj}' cannot be opened"
        if "already open" in err:
            return f"'{obj}' is already open"
        if "already closed" in err:
            return f"'{obj}' is already closed"
        if "already holding" in err or "hands are full" in err:
            return "Hands are full; drop or put down the current item first"
        if "hand is empty" in err:
            return "Hands are empty; nothing to drop or place"
        if "not switchable" in err or "no switch" in err:
            return f"'{obj}' has no switch (not toggleable)"
        if "character not found" in err:
            return "Character not initialised; call reset() first"
        if "failed" in err:
            return f"Action '{action_name}' failed: {raw_error}"
        return f"[VH Error] {raw_error}"

    # =========================================================================
    # Utility helpers
    # =========================================================================

    def _format_action_for_sequence(
        self,
        action_name: str,
        object_type: Optional[str] = None,
        object2_type: Optional[str] = None,
        action_dict: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format a compact human-readable action string for episode logs."""
        merged = dict(action_dict or {})
        merged.setdefault("action_name", action_name)
        if object_type is not None:
            merged.setdefault("object_type", object_type)
        if object2_type is not None:
            merged.setdefault("object2_type", object2_type)
        return format_vh_action_dict(merged)

    def get_action_sequence(self) -> str:
        """Return a human-readable action sequence string."""
        if not self.action_sequence:
            return "(no actions recorded)"
        return " -> ".join(self.action_sequence)

    def get_current_graph(self) -> Optional[Dict[str, Any]]:
        """Return the most recently fetched environment graph (raw VH format)."""
        return self._current_graph

    def get_character_room(self) -> str:
        """Return the name of the room the first character is currently in."""
        if not self._current_graph or not self._char_ids:
            return "unknown"
        char_id = self._char_ids[0]
        nodes_by_id = {n["id"]: n for n in self._current_graph.get("nodes", [])}
        for edge in self._current_graph.get("edges", []):
            if edge["from_id"] == char_id and edge["relation_type"] == "INSIDE":
                room = nodes_by_id.get(edge["to_id"], {})
                if room.get("category") == "Rooms":
                    return room.get("class_name", "unknown")
        return "unknown"

    # =========================================================================
    # Golden action evaluation  —  canonical entry point for evaluate scripts
    # =========================================================================

    @staticmethod
    def merge_recorded_user_actions(
        action_sequence: List[Dict[str, Any]],
        recorded_user_actions: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Inject interact-recorded object IDs into a golden action sequence.

        During an interact session the controller records which specific object
        instance was used for each step (object_id / object2_id).  Injecting
        these IDs during evaluation guarantees the *same* instance is used,
        preventing ambiguity when multiple objects of the same class exist.

        Matching is done by action signature (action_type + action_name +
        object_type + object2_type + granularity/turn key) in order.  Only
        identical signatures trigger injection; mismatches are skipped with a
        warning so that partial recordings degrade gracefully.

        Args:
            action_sequence: Golden action dicts from task.json.
            recorded_user_actions: List of action dicts from init.json
                                   ``recorded_user_actions`` field.

        Returns:
            New list with object IDs injected where available.
        """
        if not action_sequence or not recorded_user_actions:
            return action_sequence

        def _normalize(action: Dict[str, Any]) -> Dict[str, Any]:
            """Round-trip through format/parse to canonicalise field names."""
            try:
                base = parse_vh_action_string(format_vh_action_dict(action))
            except Exception:
                base = dict(action)
            for key in ("object_id", "object2_id", "turn_modifier",
                        "turn_degrees", "granularity", "magnitude"):
                if key not in action:
                    continue
                value = action[key]
                if key in ("object_id", "object2_id") and value not in (None, ""):
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        pass
                base[key] = value
            return base

        def _signature(action: Dict[str, Any]):
            name = str(action.get("action_name", "")).strip()
            gran = action.get("granularity")
            t_mod = action.get("turn_modifier")
            t_deg = action.get("turn_degrees")
            if name == "WalkForward" and not gran and action.get("magnitude") is None:
                gran = "small"
            if name in ("TurnLeft", "TurnRight"):
                if t_mod:
                    turn_key = str(t_mod).lower()
                elif t_deg is not None:
                    try:
                        turn_key = "small" if abs(float(t_deg) - 30.0) < 1e-6 else "normal"
                    except (TypeError, ValueError):
                        turn_key = "normal"
                else:
                    turn_key = "normal"
            else:
                turn_key = None
            return (
                str(action.get("action_type", "")).strip(),
                name,
                action.get("object_type"),
                action.get("object2_type"),
                gran,
                turn_key,
            )

        normalized_recorded: List[Dict[str, Any]] = []
        for idx, item in enumerate(recorded_user_actions, 1):
            try:
                normalized_recorded.append(_normalize(item))
            except Exception as exc:
                print(f"  ⚠️  skip recorded_user_actions[{idx}]: {exc}")

        if not normalized_recorded:
            return action_sequence

        merged: List[Dict[str, Any]] = []
        inject_count = 0
        compare_len = min(len(action_sequence), len(normalized_recorded))

        for idx, action in enumerate(action_sequence):
            merged_action = dict(action)
            if idx < compare_len:
                recorded = normalized_recorded[idx]
                if _signature(merged_action) == _signature(recorded):
                    for key in ("object_id", "object2_id", "turn_modifier",
                                "turn_degrees", "granularity", "magnitude"):
                        if key in recorded:
                            merged_action[key] = recorded[key]
                    inject_count += 1
                else:
                    print(
                        f"  ⚠️  recorded_user_actions[{idx+1}] signature mismatch — "
                        f"skipping injection: "
                        f"task={format_vh_action_dict(merged_action)} | "
                        f"recorded={format_vh_action_dict(recorded)}"
                    )
            merged.append(merged_action)

        if inject_count:
            print(f"  ✓ Injected {inject_count} recorded object-IDs from init.json")
        return merged

    def execute_golden_action_sequence(
        self,
        action_sequence: List[Dict[str, Any]],
        *,
        recorded_user_actions: Optional[List[Dict[str, Any]]] = None,
        step_callback=None,
    ) -> Dict[str, Any]:
        """Canonical entry point for replaying a known-good (golden) action sequence.

        Key differences vs. calling ``step_with_action_dict`` in a plain loop:

        1. **object_id injection** — instance IDs recorded during the original
           interact session are merged in via ``merge_recorded_user_actions``
           so the evaluation uses the exact same objects.
        2. **Strict interaction checks** — replay uses the same visible+CLOSE
           prerequisites as the agent runtime.
        3. **Structured result** — returns a uniform dict consumed by all
           evaluate scripts so there is no duplicated loop logic.

        Args:
            action_sequence: List of action dicts (from task.json
                             ``golden_actions``) parsed by
                             ``parse_vh_action_string``.
            recorded_user_actions: Optional list from init.json
                                   ``recorded_user_actions`` field.  When
                                   supplied, object IDs are injected before
                                   execution.
            step_callback: Optional callable(step_i, action_dict, obs, error)
                           called after each step — use for step-by-step
                           interactive / debugging mode.

        Returns:
            Dict with keys:
                - ``steps`` (int): number of non-completion actions executed.
                - ``errors`` (List[str]): per-step error strings (empty = all ok).
                - ``last_observation`` (EnvObservation | None): observation
                  after the last executed step.
        """
        # ── 1. Inject recorded object IDs ────────────────────────────────────
        merged = self.merge_recorded_user_actions(action_sequence, recorded_user_actions)

        errors: List[str] = []
        executed_actions: List[Dict[str, Any]] = []
        last_observation = None
        step_count = 0
        for i, action_dict in enumerate(merged, 1):
            action_name = action_dict.get("action_name", "")
            action_type = action_dict.get("action_type", "")

            # task_completion (Done/Fail) — stop, don't count as a step
            if action_type == "task_completion":
                print(f"  🏁 Task completion reached: {action_name}")
                break

            obs, error_msg = self.step_with_action_dict(action_dict)
            last_observation = obs
            step_count = i

            step_info = {
                "step": i,
                "action": dict(action_dict),
                "error_message": error_msg,
                "reward": obs.reward if obs else None,
                "done": obs.done if obs else False,
                "last_action_success": (
                    obs.metadata.get("lastActionSuccess")
                    if obs and obs.metadata
                    else None
                ),
                "bound_instances": (
                    dict(obs.metadata.get("bound_instances", {}))
                    if obs and obs.metadata
                    else {}
                ),
            }
            executed_actions.append(step_info)

            if error_msg:
                errors.append(f"Step {i} ({action_name}): {error_msg}")

            if step_callback is not None:
                try:
                    step_callback(i, action_dict, obs, error_msg)
                except Exception:
                    pass

        return {
            "steps": step_count,
            "errors": errors,
            "last_observation": last_observation,
            "executed_actions": executed_actions,
        }
