"""
CARLA Environment Wrapper - Dual-System Architecture
System 2 (VLM, Slow): High-level Decision - "Intent"
System 1 (LocalPlanner, Fast): Low-level Execution - "Implementation"
"""

import os
import math
import time
import queue
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image

try:
    import carla
except ImportError:
    carla = None
    print("Warning: carla not installed, please run 'pip install carla==0.9.15'")

from core.llm.schemas import EnvObservation
from envs.base import BaseEnv
from envs.carla.utils import (
    get_speed_kmh,
    calculate_2d_distance,
    calculate_angle_to_target,
    format_location,
    get_nearest_waypoint,
    get_lane_center_offset,
)
from envs.carla.local_planner import System1Executor, InterruptReason
from envs.carla.traffic_lights import TrafficLightController

#   scripts/carla/interact_carla.py                  
# (W/A/D/Q/E/S/Space)      ；   Nudge/SpeedUp/SlowDown    VLM          
_INTERACT_STYLE_TL_STEP_ACTIONS = frozenset(
    {
        "FollowLane",
        "TeleportForward",
        "GoStraight",
        "TeleportTurnLeft",
        "TurnLeft",
        "TeleportTurnRight",
        "TurnRight",
        "ChangeLaneLeft",
        "ChangeLaneRight",
        "Stop",
    }
)

# ==============================================================================
# Helper Functions for Geometry
# ==============================================================================


def point_to_segment_dist(px, py, x1, y1, x2, y2):
    """    (px,py)     (x1,y1)-(x2,y2)      """
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def is_point_in_polygon(px, py, polygon):
    """            (Ray Casting Algorithm)

    Args:
        px, py:    
        polygon:        ，   [x, y]   [x, y, z]   
    """
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        #   2D [x,y]   3D [x,y,z]   
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]

        intersect = ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def distance_to_polygon(px, py, polygon):
    """           (   0，          )

    Args:
        px, py:    
        polygon:        ，   [x, y]   [x, y, z]   
    """
    if is_point_in_polygon(px, py, polygon):
        return 0.0

    min_dist = float("inf")
    for i in range(len(polygon)):
        #   2D 3D  ，  x,y
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        d = point_to_segment_dist(px, py, p1[0], p1[1], p2[0], p2[1])
        if d < min_dist:
            min_dist = d
    return min_dist


def load_world_with_retry(
    client, scene_name: str, base_timeout: float, retries: int = 2
):
    last_error = None
    original_timeout = float(base_timeout)
    extended_timeout = max(original_timeout, 60.0)

    for _ in range(max(1, retries)):
        try:
            client.set_timeout(extended_timeout)
            return client.load_world(scene_name)
        except Exception as e:
            last_error = e
        finally:
            client.set_timeout(original_timeout)

    raise RuntimeError(last_error)


def _apply_world_weather(world, override_name: Optional[str], fallback_name: str) -> None:
    """Set CARLA weather; override_name wins, else fallback_name."""
    if world is None or carla is None:
        return
    name = override_name or fallback_name
    preset = getattr(carla.WeatherParameters, str(name), None)
    if preset is None:
        preset = carla.WeatherParameters.ClearNoon
    world.set_weather(preset)


# ==============================================================================
# Walker Environment Wrapper (New Implementation)
# ==============================================================================


class WalkerEnvWrapper(BaseEnv):
    """CARLA Walker Environment Wrapper"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config)
        self.config = config or {}
        env_config = self.config.get("env", {})
        self.output_dir = kwargs.get("output_dir", env_config.get("output_dir", "outputs"))
        os.makedirs(self.output_dir, exist_ok=True)

        # Connection settings
        self.host = kwargs.get("host", env_config.get("host", "localhost")) or "localhost"
        self.port = kwargs.get("port", env_config.get("port", 2000))
        self.timeout = kwargs.get("timeout", env_config.get("timeout", 10.0))

        #     ：   sensors.rgb_camera_walker（  scripts/carla/interact_walker.py   ）
        sensors = env_config.get("sensors", {})
        walker_rgb = sensors.get("rgb_camera_walker")
        if walker_rgb:
            self.camera_width = walker_rgb.get("width", 1280)
            self.camera_height = walker_rgb.get("height", 720)
            self.camera_fov = walker_rgb.get("fov", 60.0)
            self._walker_cam_attach = walker_rgb.get("attach") or {}
        else:
            self.camera_width = 1280
            self.camera_height = 720
            self.camera_fov = 60.0
            self._walker_cam_attach = {}

        self.map_name = kwargs.get("map", env_config.get("map", "Town01"))

        # State
        self.client = None
        self.world = None
        self.walker = None
        self.camera = None
        self.collision_sensor = None
        self.collision_count = 0
        self.last_collision_frame = 0
        self.image_queue = queue.Queue()
        self.step_counter = 0
        self.last_image = None
        self.total_distance = 0.0

        # Task state
        self.target_location = None
        self.target_polygon = None
        self.success_condition = {}

        self._connect()

    def _connect(self):
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            try:
                current_world = self.client.get_world()
                current_map = current_world.get_map().name.split("/")[-1]
                if current_map == self.map_name:
                    self.world = current_world
                else:
                    self.world = load_world_with_retry(
                        self.client, self.map_name, self.timeout
                    )
            except Exception as e:
                print(
                    f"Warning: Failed to load map '{self.map_name}', fallback to current world: {e}"
                )
                self.world = self.client.get_world()

            if self.world is None:
                raise RuntimeError("CARLA world is None after connect")

            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            _apply_world_weather(
                self.world,
                None,
                self.config.get("env", {}).get("weather", "ClearNoon"),
            )

        except Exception as e:
            self.world = None
            raise RuntimeError(f"Error connecting to CARLA walker world: {e}")

    def reset(
        self,
        task_description: str,
        scene: Optional[str] = None,
        weather: Optional[str] = None,
    ) -> EnvObservation:
        self.action_sequence = []
        self.step_counter = 0
        self.collision_count = 0
        self.last_collision_frame = 0
        self._cleanup()

        if self.world is None:
            self._connect()

        if scene and scene != self.map_name:
            self.map_name = scene
            try:
                self.world = load_world_with_retry(self.client, scene, self.timeout)
            except Exception as e:
                print(f"Warning: Failed to switch to scene '{scene}', keep current world: {e}")

            if self.world is None:
                raise RuntimeError("CARLA walker world is not available during reset")

            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)

        env_cfg = self.config.get("env", {})
        _apply_world_weather(
            self.world, weather, env_cfg.get("weather", "ClearNoon")
        )

        # Parse Success Conditions
        task_conf = self.config.get("task", {})
        self.success_condition = task_conf.get("success_conditions", {})

        # Setup targets
        self.target_location = None
        self.target_polygon = None

        cond_type = self.success_condition.get("type")
        if cond_type == "distance_to_waypoint":
            loc = self.success_condition.get("target_location")
            if loc and len(loc) >= 2:
                self.target_location = carla.Location(
                    x=loc[0], y=loc[1], z=loc[2] if len(loc) > 2 else 0
                )
                print(
                    f"Goal: Reach location ({self.target_location.x:.1f}, {self.target_location.y:.1f})"
                )

        elif cond_type == "polygon_area":
            poly = self.success_condition.get("target_polygon")
            if poly:
                # Convert [[x,y,z],...] to list of (x,y)
                self.target_polygon = [(p[0], p[1]) for p in poly]
                print(
                    f"Goal: Reach building area ({len(self.target_polygon)} points)"
                )

        self._spawn_walker()
        self._setup_camera()
        self._setup_collision_sensor()

        for _ in range(10):
            self.world.tick()
        image = self._get_latest_image()

        return EnvObservation(
            image_path=self._save_frame(image, "reset"),
            text_state=self._generate_text_state(),
            reward=0.0,
            done=False,
            metadata=self._build_metadata(success=False),
        )

    def _check_success(self) -> bool:
        """         (      )"""
        if not self.walker:
            return False

        curr_loc = self.walker.get_location()
        threshold_label = self.success_condition.get("threshold_label", "medium")
        # Strict=3m, Medium=5m, Loose=10m
        threshold_map = {"strict": 3.0, "medium": 5.0, "loose": 10.0}
        threshold = threshold_map.get(threshold_label, 5.0)

        dist = float("inf")
        cond_type = self.success_condition.get("type")

        if cond_type == "distance_to_waypoint" and self.target_location:
            dist = math.hypot(
                curr_loc.x - self.target_location.x, curr_loc.y - self.target_location.y
            )

        elif cond_type == "polygon_area" and self.target_polygon:
            dist = distance_to_polygon(curr_loc.x, curr_loc.y, self.target_polygon)

        if dist <= threshold:
            print(f"Success Condition Met! Distance {dist:.2f}m <= {threshold}m")
            return True

        return False

    def step_with_action_dict(
        self, action_dict: dict
    ) -> Tuple[EnvObservation, Optional[str]]:
        if self.world is None or self.walker is None:
            err = "Walker environment not initialized (world/walker missing)"
            return (
                EnvObservation(
                    image_path="",
                    text_state="",
                    reward=0.0,
                    done=False,
                    metadata=self._build_metadata(success=False, error=err),
                ),
                err,
            )

        action_name = action_dict.get("action_name")
        params = action_dict.get("parameters", {})
        self.step_counter += 1

        # Record action sequence for episode logs.
        action_label = str(action_name) if action_name else "UnknownAction"
        if isinstance(params, dict) and params:
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            self.action_sequence.append(f"{action_label}({params_str})")
        else:
            self.action_sequence.append(f"{action_label}()")

        # Execute Action
        #      （  prompt / interact     ）
        # -     : 2.5 / 5 / 10  
        # -       : medium=30° / large=90°
        try:
            if action_name == "WalkForward":
                dist = self._snap_walk_distance(params.get("distance", 10.0))
                self._apply_walk(dist, forward=True)
            elif action_name == "WalkBackward":
                dist = self._snap_walk_distance(params.get("distance", 10.0))
                self._apply_walk(dist, forward=False)
            elif action_name == "TurnLeft":
                deg = self._snap_turn_degrees(params.get("degrees", 30.0))
                self._apply_turn(-deg)
            elif action_name == "TurnRight":
                deg = self._snap_turn_degrees(params.get("degrees", 30.0))
                self._apply_turn(deg)
        except Exception as e:
            print(f"Action Error: {e}")

        # Tick world
        for _ in range(5):
            self.world.tick()
        image = self._get_latest_image()

        #     ：      Done        ，     (done=True)
        #       Done     ，    (done=False,            result=Fail)
        #          done=True   reward  ，    evaluator    
        #        "It should fail if not reached"，    done=True         
        #     replay        done=True，evaluate        

        is_success = self._check_success()
        real_done = False

        if action_name == "Done":
            if is_success:
                real_done = True  # Task Successfully Completed
            else:
                print("Agent called Done but target not reached!")
                # real_done = False # Keep False so evaluator knows it wasn't a success termination

        return EnvObservation(
            image_path=self._save_frame(image, f"step_{self.step_counter}"),
            text_state=self._generate_text_state(),
            reward=10.0 if real_done else 0.0,
            done=real_done,
            metadata=self._build_metadata(success=real_done),
        ), None

    def get_distance_to_target(self) -> float:
        """        2D  ；       inf """
        if not self.walker:
            return float("inf")

        curr_loc = self.walker.get_location()
        cond_type = self.success_condition.get("type")

        if cond_type == "distance_to_waypoint" and self.target_location:
            return math.hypot(
                curr_loc.x - self.target_location.x, curr_loc.y - self.target_location.y
            )
        if cond_type == "polygon_area" and self.target_polygon:
            return distance_to_polygon(curr_loc.x, curr_loc.y, self.target_polygon)
        return float("inf")

    def _generate_text_state(self, info=None) -> str:
        if not self.walker:
            return ""
        loc = self.walker.get_location()
        yaw = self.walker.get_transform().rotation.yaw
        return f"Pos: ({loc.x:.1f}, {loc.y:.1f}), Heading: {yaw:.1f}, Collisions: {self.collision_count}"

    def _build_metadata(
        self, success: bool = False, error: Optional[str] = None
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "success": bool(success),
            "sceneName": self.map_name,
            "objects": [],
        }

        if self.walker:
            transform = self.walker.get_transform()
            loc = transform.location
            rot = transform.rotation
            metadata["agent"] = {
                "position": {"x": loc.x, "y": loc.y, "z": loc.z},
                "rotation": {"pitch": rot.pitch, "yaw": rot.yaw, "roll": rot.roll},
            }

        if error:
            metadata["error"] = error

        metadata["collisions"] = int(self.collision_count)

        return metadata

    def close(self):
        self._cleanup()

    def _cleanup(self):
        if self.collision_sensor:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            except Exception:
                pass
            self.collision_sensor = None
        if self.camera:
            try:
                self.camera.destroy()
            except Exception:
                pass
            self.camera = None
        if self.walker:
            try:
                self.walker.destroy()
            except Exception:
                pass
            self.walker = None

    def _spawn_walker(self):
        """    （        ）"""
        bp = self.world.get_blueprint_library().filter("walker.pedestrian.*")[0]

        #      10  
        for i in range(10):
            # 1.        
            try:
                #     World      
                loc = self.world.get_random_location_from_navigation()
            except AttributeError:
                #      API，  Map   
                loc = self.world.get_map().get_random_location_from_navigation()

            spawn_point = None
            if loc:
                spawn_point = carla.Transform(loc)
            else:
                #   ：    Spawn Points   
                spawn_points = self.world.get_map().get_spawn_points()
                if spawn_points:
                    spawn_point = spawn_points[i % len(spawn_points)]

            if spawn_point:
                #     ：        ，       
                spawn_point.location.z += 1.0

                #    try_spawn_actor     
                self.walker = self.world.try_spawn_actor(bp, spawn_point)
                if self.walker:
                    #     ，    
                    print(f"Walker spawned at {spawn_point.location}")
                    return

        #    10     ，    
        raise RuntimeError("Failed to spawn walker after 10 attempts (Collision).")

    def _setup_camera(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.camera_width))
        bp.set_attribute("image_size_y", str(self.camera_height))
        bp.set_attribute("fov", str(self.camera_fov))
        a = self._walker_cam_attach
        cam_tf = carla.Transform(
            carla.Location(
                x=float(a.get("x", 0.0)),
                y=float(a.get("y", 0.0)),
                z=float(a.get("z", 1.7)),
            ),
            carla.Rotation(
                pitch=float(a.get("pitch", 0.0)),
                yaw=float(a.get("yaw", 0.0)),
                roll=float(a.get("roll", 0.0)),
            ),
        )
        self.camera = self.world.spawn_actor(bp, cam_tf, attach_to=self.walker)
        self.camera.listen(lambda i: self.image_queue.put(i))

    def _setup_collision_sensor(self) -> None:
        """Attach collision sensor to walker (aligned with CarlaEnvWrapper / interact_carla)."""
        if not self.walker or not self.world:
            return
        try:
            bp = self.world.get_blueprint_library().find("sensor.other.collision")
            self.collision_sensor = self.world.spawn_actor(
                bp, carla.Transform(), attach_to=self.walker
            )
            self.collision_sensor.listen(self._on_collision_event)
        except Exception as e:
            print(f"Warning: walker collision sensor setup failed: {e}")
            self.collision_sensor = None

    def _on_collision_event(self, event) -> None:
        """Collision callback with frame dedup (same logic as CarlaEnvWrapper)."""
        if event.frame - self.last_collision_frame > 20:
            self.collision_count += 1
            self.last_collision_frame = event.frame

    def _get_latest_image(self):
        latest_frame = None

        try:
            latest_frame = self.image_queue.get(timeout=2.0)
            while not self.image_queue.empty():
                latest_frame = self.image_queue.get_nowait()
        except queue.Empty:
            pass

        if latest_frame is not None:
            array = np.frombuffer(latest_frame.raw_data, dtype=np.uint8)
            array = array.reshape((latest_frame.height, latest_frame.width, 4))
            image = array[:, :, :3][:, :, ::-1].copy()
            self.last_image = image
            return image

        if self.last_image is not None:
            return self.last_image.copy()

        return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)

    def _save_frame(self, img, prefix):
        if img is None:
            return ""

        filename = f"{prefix}.png"
        frame_path = os.path.join(self.output_dir, filename)
        Image.fromarray(img).save(frame_path)
        return frame_path

    def _get_current_observation(self) -> EnvObservation:
        for _ in range(5):
            self.world.tick()

        image = self._get_latest_image()
        return EnvObservation(
            image_path=self._save_frame(image, f"step_{self.step_counter}_current"),
            text_state=self._generate_text_state(),
            reward=0.0,
            done=self._check_success(),
            metadata=self._build_metadata(success=False),
        )

    def _apply_walk(self, dist, forward):
        """Continuous multi-frame walk aligned with interact_walker.py."""
        WALKER_SPEED = 8.0

        start_location = self.walker.get_transform().location
        target_distance = float(dist)

        walker_transform = self.walker.get_transform()
        forward_vec = walker_transform.get_forward_vector()

        direction = carla.Vector3D(
            x=forward_vec.x * (1.0 if forward else -1.0),
            y=forward_vec.y * (1.0 if forward else -1.0),
            z=0.0,
        )
        length = (direction.x ** 2 + direction.y ** 2) ** 0.5
        if length > 0.0:
            direction.x /= length
            direction.y /= length

        max_frames = int(target_distance / (WALKER_SPEED * 0.05)) + 50
        traveled = 0.0
        for _ in range(max_frames):
            control = carla.WalkerControl()
            control.direction = direction
            control.speed = WALKER_SPEED
            control.jump = False
            self.walker.apply_control(control)

            self.world.tick()

            current_location = self.walker.get_transform().location
            traveled = (
                (current_location.x - start_location.x) ** 2
                + (current_location.y - start_location.y) ** 2
            ) ** 0.5

            if traveled >= target_distance:
                break

        stop_control = carla.WalkerControl()
        stop_control.direction = carla.Vector3D(0, 0, 0)
        stop_control.speed = 0.0
        stop_control.jump = False
        self.walker.apply_control(stop_control)

        overshoot = traveled - target_distance
        if overshoot > 0.01:
            corrected_transform = self.walker.get_transform()
            corrected_transform.location.x -= direction.x * overshoot
            corrected_transform.location.y -= direction.y * overshoot
            self.walker.set_transform(corrected_transform)

        self.world.tick()
        self.total_distance += min(traveled, target_distance)

    def _apply_turn(self, angle):
        tf = self.walker.get_transform()
        tf.rotation.yaw += angle
        self.walker.set_transform(tf)
        self.world.tick()

    @staticmethod
    def _snap_walk_distance(value) -> float:
        """   /replay              {2.5, 5.0, 10.0}        """
        try:
            from envs.carla.step_sizes import resolve_step_distance
        except ModuleNotFoundError:
            from mllm_base_agent.environments.carla.step_sizes import (
                resolve_step_distance,
            )

        return resolve_step_distance(value)

    @staticmethod
    def _snap_turn_degrees(value) -> float:
        """   /replay              {30.0, 90.0}        """
        try:
            from envs.carla.step_sizes import resolve_turn_degrees
        except ModuleNotFoundError:
            from mllm_base_agent.environments.carla.step_sizes import (
                resolve_turn_degrees,
            )

        return resolve_turn_degrees(value)


# ==============================================================================
# Carla Vehicle Environment Wrapper (Original)
# ==============================================================================


class CarlaEnvWrapper(BaseEnv):
    """CARLA Environment Wrapper (Vehicle)"""

    def __init__(
        self,
        host="localhost",
        port=2000,
        timeout=10.0,
        config=None,
        output_dir="outputs",
    ):
        if carla is None:
            raise ImportError("carla module not installed")
        super().__init__(config)
        self.config = config or {}
        env_config = self.config.get("env", {})
        self.host = env_config.get("host", host) or host or "localhost"
        self.port = env_config.get("port", port)
        self.timeout = env_config.get("timeout", timeout)

        self.map_name = env_config.get("map", "Town01")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        rgb_config = env_config.get("sensors", {}).get("rgb_camera", {})
        self.camera_width = rgb_config.get("width", 800)
        self.camera_height = rgb_config.get("height", 600)
        self.camera_fov = rgb_config.get("fov", 90)
        self._vehicle_cam_attach = rgb_config.get("attach") or {}

        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.system1 = None
        self.image_queue = queue.Queue()
        self.last_image = None
        self.step_counter = 0
        self.success_condition = {}
        self.target_polygon = None
        self.traffic_light_controller: Optional[TrafficLightController] = None
        #   scripts/carla/interact_carla.py   ：  N                 
        self._tl_action_counter = 0
        self._tl_change_every_n = 3
        #   interact_carla.py   ：       
        self.collision_sensor = None
        self.collision_count = 0
        self.last_collision_frame = 0
        self._connect()

    def _connect(self):
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)

            try:
                current_world = self.client.get_world()
                current_map = current_world.get_map().name.split("/")[-1]
                if current_map == self.map_name:
                    self.world = current_world
                else:
                    self.world = load_world_with_retry(
                        self.client, self.map_name, self.timeout
                    )
            except Exception as e:
                print(
                    f"Warning: Failed to load map '{self.map_name}', fallback to current world: {e}"
                )
                self.world = self.client.get_world()

            if self.world is None:
                raise RuntimeError("CARLA world is None after connect")

            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            env_cfg = self.config.get("env", {})
            _apply_world_weather(
                self.world, None, env_cfg.get("weather", "ClearNoon")
            )
        except Exception as e:
            self.world = None
            raise RuntimeError(f"Failed to initialize CARLA vehicle world: {e}")

    def reset(
        self,
        task_description: str,
        scene: Optional[str] = None,
        weather: Optional[str] = None,
    ) -> EnvObservation:
        self.step_counter = 0
        self.action_sequence = []
        self.collision_count = 0
        self.last_collision_frame = 0
        self._cleanup()

        if self.world is None:
            self._connect()

        # Update Scene
        if scene and scene != self.map_name:
            try:
                self.world = load_world_with_retry(self.client, scene, self.timeout)
                self.map_name = scene
            except Exception as e:
                print(f"Warning: Failed to switch to scene '{scene}', keep current world: {e}")

        if self.world is None:
            raise RuntimeError("CARLA world is not available during reset")

        env_cfg = self.config.get("env", {})
        _apply_world_weather(
            self.world, weather, env_cfg.get("weather", "ClearNoon")
        )

        # Determine success conditions
        task_config = self.config.get("task", {})
        self.success_condition = task_config.get("success_conditions", {})
        self.target_polygon = self.success_condition.get("target_polygon")

        # Spawn Vehicle
        try:
            bp_lib = self.world.get_blueprint_library()
            bp = bp_lib.find("vehicle.tesla.model3")
            spawn_points = self.world.get_map().get_spawn_points()
            start_pose = spawn_points[0] if spawn_points else carla.Transform()

            self.vehicle = self.world.try_spawn_actor(bp, start_pose)
            if not self.vehicle:
                # Fallback to a safe location if spawn point blocked
                self.vehicle = self.world.try_spawn_actor(
                    bp, carla.Transform(carla.Location(z=2))
                )
        except Exception as e:
            print(f"Spawn Error: {e}")

        if self.vehicle is None:
            print("Vehicle spawn failed, environment is not executable")
            self.system1 = None
            return EnvObservation(
                image_path=self._save_frame(self._blank_image(), "reset_error"),
                text_state="",
                reward=0.0,
                done=False,
                metadata=self._build_metadata(
                    success=False, error="vehicle_spawn_failed"
                ),
            )

        # Init System1（    ，    executor，    interact_carla              ）
        if self.vehicle:
            from envs.carla.local_planner import System1Executor

            self._setup_camera()
            self.system1 = System1Executor(
                self.vehicle,
                self.world,
                self.world.get_map(),
                rgb_camera=self.camera,
            )
            self._setup_traffic_lights()
            if self.traffic_light_controller:
                self.system1.traffic_light_controller = self.traffic_light_controller
            self._setup_collision_sensor()
            self.world.tick()

        return self._get_current_observation(prefix="reset")

    def step_with_action_dict(
        self, action_dict: dict
    ) -> Tuple[EnvObservation, Optional[str]]:
        action_name = action_dict.get("action_name")
        params = action_dict.get("parameters", {})
        self.step_counter += 1

        # Record action sequence for episode logs.
        action_label = str(action_name) if action_name else "UnknownAction"
        if isinstance(params, dict) and params:
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            self.action_sequence.append(f"{action_label}({params_str})")
        else:
            self.action_sequence.append(f"{action_label}()")

        if self.world is None:
            err = "Vehicle environment not initialized (world missing)"
            return (
                EnvObservation(
                    image_path="",
                    text_state="",
                    reward=0.0,
                    done=False,
                    metadata=self._build_metadata(success=False, error=err),
                ),
                err,
            )

        if self.vehicle is None:
            err = "Vehicle actor not available (spawn failed or destroyed)"
            return (
                EnvObservation(
                    image_path="",
                    text_state="",
                    reward=0.0,
                    done=False,
                    metadata=self._build_metadata(success=False, error=err),
                ),
                err,
            )

        # Handle explicitly checking success
        if action_name == "Done":
            is_success = self._check_success()
            return (
                self._get_current_observation(
                    prefix=f"step_{self.step_counter}_done",
                    reward=10.0 if is_success else 0.0,
                    done=is_success,
                    success=is_success,
                ),
                None,
            )

        # Execute Action via System 1
        if self.system1:
            try:
                action_result = self.system1.execute_action(action_name, params)
            except Exception as e:
                return EnvObservation(
                    image_path=self._save_frame(
                        self._blank_image(), f"step_{self.step_counter}_error"
                    ),
                    text_state=self._generate_text_state(),
                    reward=0.0,
                    done=False,
                    metadata=self._build_metadata(success=False, error=str(e)),
                ), str(e)
        else:
            err = "System1 executor is not initialized"
            return (
                EnvObservation(
                    image_path="",
                    text_state=self._generate_text_state(),
                    reward=0.0,
                    done=False,
                    metadata=self._build_metadata(success=False, error=err),
                ),
                err,
            )

        #   interact_carla   ：         ； “     ”     ；  _tl_change_every_n    advance
        blocked = isinstance(action_result, dict) and action_result.get("blocked_by_red_light", False)
        counts_toward_tl = action_name in _INTERACT_STYLE_TL_STEP_ACTIONS
        if (
            self.traffic_light_controller
            and self.traffic_light_controller.enabled
            and not blocked
            and counts_toward_tl
        ):
            self._tl_action_counter += 1
            if self._tl_action_counter >= self._tl_change_every_n:
                self._tl_action_counter = 0
                self.traffic_light_controller.advance()

        return (
            self._get_current_observation(prefix=f"step_{self.step_counter}"),
            None,
        )

    def _build_metadata(
        self, success: bool = False, error: Optional[str] = None
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "success": bool(success),
            "sceneName": self.map_name,
            "objects": [],
        }

        if self.vehicle:
            transform = self.vehicle.get_transform()
            loc = transform.location
            rot = transform.rotation
            metadata["agent"] = {
                "position": {"x": loc.x, "y": loc.y, "z": loc.z},
                "rotation": {"pitch": rot.pitch, "yaw": rot.yaw, "roll": rot.roll},
            }

        if error:
            metadata["error"] = error

        if self.traffic_light_controller:
            metadata.update(self.traffic_light_controller.build_report())

        metadata["collisions"] = int(self.collision_count)

        return metadata

    def _check_success(self) -> bool:
        if not self.vehicle:
            return False

        # Handle threshold_label (strict=3m, medium=5m, loose=10m)
        threshold_label = self.success_condition.get("threshold_label", "medium")
        threshold_map = {"strict": 3.0, "medium": 5.0, "loose": 10.0}
        tolerance = self.success_condition.get(
            "tolerance", threshold_map.get(threshold_label, 5.0)
        )

        dist = self.get_distance_to_target()
        if math.isfinite(dist):
            print(
                f"Distance to target: {dist:.2f}m (threshold: {tolerance}m, label: {threshold_label})"
            )
            if dist <= tolerance:
                print(f"Success Condition Met! Distance {dist:.2f}m <= {tolerance}m")
                return True
            else:
                print(f"Not yet reached: {dist:.2f}m > {tolerance}m")

        else:
            print("Warning: Success condition unsupported or target not configured")

        return False

    def get_distance_to_target(self) -> float:
        """        2D  ；       inf """
        if not self.vehicle:
            return float("inf")

        loc = self.vehicle.get_location()
        cond_type = self.success_condition.get("type")
        if cond_type == "polygon_area" and self.target_polygon:
            return distance_to_polygon(loc.x, loc.y, self.target_polygon)
        return float("inf")

    def close(self):
        self._cleanup()

    def _generate_text_state(self, metadata: Any = None) -> str:
        """Generate text state description"""
        if not self.vehicle:
            return ""

        transform = self.vehicle.get_transform()
        loc = transform.location
        rot = transform.rotation

        state_str = (
            f"Pos: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}), Heading: {rot.yaw:.1f}"
        )
        if self.traffic_light_controller and self.traffic_light_controller.enabled:
            state_str += (
                f", TrafficLight: {self.traffic_light_controller.current_state}"
            )
        state_str += f", Collisions: {self.collision_count}"
        return state_str

    def _cleanup(self):
        if self.traffic_light_controller:
            self.traffic_light_controller.release()
            self.traffic_light_controller = None
        self.last_image = None
        self.image_queue = queue.Queue()
        if self.collision_sensor:
            try:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
            except Exception:
                pass
            self.collision_sensor = None
        if self.camera:
            try:
                self.camera.destroy()
            except Exception:
                pass
            self.camera = None
        if self.vehicle:
            try:
                self.vehicle.destroy()
            except Exception:
                pass
            self.vehicle = None

    def _setup_camera(self):
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.camera_width))
        bp.set_attribute("image_size_y", str(self.camera_height))
        bp.set_attribute("fov", str(self.camera_fov))
        a = self._vehicle_cam_attach
        camera_transform = carla.Transform(
            carla.Location(
                x=float(a.get("x", 1.5)),
                y=float(a.get("y", 0.0)),
                z=float(a.get("z", 2.4)),
            ),
            carla.Rotation(
                pitch=float(a.get("pitch", -10.0)),
                yaw=float(a.get("yaw", 0.0)),
                roll=float(a.get("roll", 0.0)),
            ),
        )
        self.camera = self.world.spawn_actor(
            bp, camera_transform, attach_to=self.vehicle
        )
        self.camera.listen(lambda i: self.image_queue.put(i))

    def _setup_collision_sensor(self) -> None:
        """  scripts/carla/interact_carla.py   ：sensor.other.collision     """
        if not self.vehicle or not self.world:
            return
        try:
            bp = self.world.get_blueprint_library().find("sensor.other.collision")
            self.collision_sensor = self.world.spawn_actor(
                bp, carla.Transform(), attach_to=self.vehicle
            )
            self.collision_sensor.listen(self._on_collision_sensor_event)
            print("✓          (  interact_carla   )")
        except Exception as e:
            print(f"⚠️          : {e}")
            self.collision_sensor = None

    def _on_collision_sensor_event(self, event) -> None:
        """  interact_carla._collision_callback   ：    +    +    System1 """
        if event.frame - self.last_collision_frame > 20:
            self.collision_count += 1
            self.last_collision_frame = event.frame
            print(f"⚠️    ! (  : {self.collision_count})")
        if self.system1 is not None:
            try:
                self.system1.set_collision_flag()
            except Exception:
                pass

    def _setup_traffic_lights(self):
        self.traffic_light_controller = None
        self._tl_action_counter = 0
        self._tl_change_every_n = 3
        controller = TrafficLightController(self.world, carla_module=carla)
        if controller.initialize():
            self.traffic_light_controller = controller

    def _get_latest_image(self):
        latest_frame = None
        try:
            latest_frame = self.image_queue.get(timeout=2.0)
            while not self.image_queue.empty():
                latest_frame = self.image_queue.get_nowait()
        except queue.Empty:
            pass

        if latest_frame is not None:
            array = np.frombuffer(latest_frame.raw_data, dtype=np.uint8)
            array = array.reshape((latest_frame.height, latest_frame.width, 4))
            image = array[:, :, :3][:, :, ::-1].copy()
            self.last_image = image
            return image

        if self.last_image is not None:
            return self.last_image.copy()

        return self._blank_image()

    def _blank_image(self):
        return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)

    def _save_frame(self, img, prefix):
        filename = f"{prefix}.png"
        frame_path = os.path.join(self.output_dir, filename)
        Image.fromarray(img).save(frame_path)
        return frame_path

    def _get_current_observation(
        self,
        prefix: str = "obs",
        reward: float = 0.0,
        done: bool = False,
        success: bool = False,
        error: Optional[str] = None,
    ) -> EnvObservation:
        if self.world:
            for _ in range(5):
                self.world.tick()

        image = self._get_latest_image()
        return EnvObservation(
            image_path=self._save_frame(image, prefix),
            text_state=self._generate_text_state(),
            reward=reward,
            done=done,
            metadata=self._build_metadata(success=success, error=error),
        )
