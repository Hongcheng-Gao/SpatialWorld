"""
CARLA Local Planner (System 1: Fast System)
Responsible for low-level tactical execution: lane keeping, speed control, obstacle detection

Design Philosophy:
- VLM (System 2, Slow): Responsible for "Intent" - High-level decisions
- LocalPlanner (System 1, Fast): Responsible for "Implementation" - Per-frame execution

Reference: CARLA official agents.navigation.local_planner
"""

import math
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
from enum import Enum

try:
    import carla
    from agents.navigation.local_planner import LocalPlanner as CarlaLocalPlanner
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    CARLA_AGENTS_AVAILABLE = True
except ImportError:
    CARLA_AGENTS_AVAILABLE = False
    CarlaLocalPlanner = None
    GlobalRoutePlanner = None


class InterruptReason(Enum):
    """Interrupt reason enumeration"""

    NONE = "none"
    RED_LIGHT = "red_light"
    PEDESTRIAN = "pedestrian"
    VEHICLE_AHEAD = "vehicle_ahead"
    COLLISION = "collision"
    LANE_INVASION = "lane_invasion"
    REACHED_WAYPOINT = "reached_waypoint"
    MAX_FRAMES = "max_frames"
    JUNCTION_ENTRY = "junction_entry"


class PIDController:
    """PID Controller"""

    def __init__(self, Kp: float = 1.0, Ki: float = 0.1, Kd: float = 0.05):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self._integral = 0.0
        self._previous_error = 0.0

    def step(self, error: float, dt: float = 0.05) -> float:
        self._integral += error * dt
        self._integral = max(-10.0, min(10.0, self._integral))  # Anti-windup

        derivative = (error - self._previous_error) / dt if dt > 0 else 0.0
        self._previous_error = error

        return self.Kp * error + self.Ki * self._integral + self.Kd * derivative

    def reset(self):
        self._integral = 0.0
        self._previous_error = 0.0


class LocalPlannerSystem1:
    """
    System 1: Local Planner (Fast System)

    Functions:
    1. Lane following: Automatically maintain lane center
    2. Speed control: Maintain target speed
    3. Obstacle detection: Detect vehicles and pedestrians ahead
    4. Traffic light response: Automatically detect and respond to red lights
    5. Interrupt triggering: Trigger interrupts in dangerous situations
    """

    def __init__(
        self,
        vehicle: "carla.Vehicle",
        world: "carla.World",
        world_map: "carla.Map",
        dt: float = 0.05,
    ):
        self.vehicle = vehicle
        self.world = world
        self.map = world_map
        self.dt = dt

        # PID controllers
        self.speed_pid = PIDController(Kp=1.0, Ki=0.1, Kd=0.05)
        self.steer_pid = PIDController(Kp=1.5, Ki=0.0, Kd=0.3)

        # Current task
        self.target_speed: float = 0.0  # km/h
        self.current_mode: str = (
            "idle"  # idle, follow_lane, turn_left, turn_right, stop
        )

        # Detection parameters
        self.detection_range = 30.0  # Forward detection distance (m)
        self.safe_distance = 10.0  # Safe following distance (m)
        self.pedestrian_range = 15.0  # Pedestrian detection distance (m)

        # State
        self.frames_executed = 0
        self.last_interrupt_reason = InterruptReason.NONE

        # Collision flag (set by external wrapper)
        self.collision_flag = False
        self.lane_invasion_flag = False

    def set_task(
        self,
        mode: str,
        target_speed: float = 30.0,
        turn_direction: Optional[str] = None,
    ):
        """Set current task

        Args:
            mode: Control mode (follow_lane, turn_left, turn_right, stop)
            target_speed: Target speed (km/h)
            turn_direction: Turn direction (left, right, straight)
        """
        self.current_mode = mode
        self.target_speed = target_speed
        self.turn_direction = turn_direction
        self.frames_executed = 0
        self.last_interrupt_reason = InterruptReason.NONE

        # Reset PID
        self.speed_pid.reset()
        self.steer_pid.reset()

    def get_current_speed(self) -> float:
        """Get current speed (km/h)"""
        v = self.vehicle.get_velocity()
        return math.sqrt(v.x**2 + v.y**2 + v.z**2) * 3.6

    def get_current_waypoint(self) -> Optional["carla.Waypoint"]:
        """Get current waypoint"""
        loc = self.vehicle.get_transform().location
        return self.map.get_waypoint(loc, project_to_road=True)

    def get_target_waypoint(self, distance: float = 5.0) -> Optional["carla.Waypoint"]:
        """Get forward target waypoint"""
        current_wp = self.get_current_waypoint()
        if current_wp is None:
            return None

        # Select waypoint based on mode
        if self.current_mode in ["turn_left", "turn_right"]:
            # Junction turning: select corresponding direction
            if current_wp.is_junction:
                next_wps = current_wp.next(distance)
                if next_wps:
                    # Simple strategy: select waypoint with most suitable angle
                    return self._select_turn_waypoint(current_wp, next_wps)

        # Default: follow lane
        next_wps = current_wp.next(distance)
        return next_wps[0] if next_wps else None

    def _select_turn_waypoint(
        self, current_wp: "carla.Waypoint", next_wps: List["carla.Waypoint"]
    ) -> "carla.Waypoint":
        """Select turning waypoint.

        CARLA coordinate convention (UE4 left-handed):
          positive yaw change => clockwise => RIGHT turn
          negative yaw change => counter-clockwise => LEFT turn
        """
        current_yaw = current_wp.transform.rotation.yaw

        best_wp = next_wps[0]

        for wp in next_wps:
            next_yaw = wp.transform.rotation.yaw
            angle_diff = next_yaw - current_yaw

            # Normalize to [-180, 180]
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360

            # Negative angle_diff => left turn; positive => right turn
            if self.current_mode == "turn_left" and angle_diff < -20:
                best_wp = wp
                break
            elif self.current_mode == "turn_right" and angle_diff > 20:
                best_wp = wp
                break

        return best_wp

    def check_traffic_light(self) -> bool:
        """Check if traffic light is red

        Returns:
            True if need to stop (red light)
        """
        if self.vehicle.is_at_traffic_light():
            tl = self.vehicle.get_traffic_light()
            if tl and tl.get_state() == carla.TrafficLightState.Red:
                return True
        return False

    def check_vehicle_ahead(self) -> Tuple[bool, float]:
        """Check if there is a vehicle ahead

        Returns:
            (has_vehicle, distance) tuple
        """
        vehicle_loc = self.vehicle.get_transform().location
        vehicle_fwd = self.vehicle.get_transform().get_forward_vector()

        # Get all vehicles
        vehicles = self.world.get_actors().filter("vehicle.*")

        min_distance = float("inf")
        has_vehicle = False

        for other in vehicles:
            if other.id == self.vehicle.id:
                continue

            other_loc = other.get_transform().location

            # Calculate distance
            dx = other_loc.x - vehicle_loc.x
            dy = other_loc.y - vehicle_loc.y
            distance = math.sqrt(dx**2 + dy**2)

            if distance > self.detection_range:
                continue

            # Check if ahead (dot product)
            dot = dx * vehicle_fwd.x + dy * vehicle_fwd.y
            if dot > 0 and distance < min_distance:
                min_distance = distance
                has_vehicle = True

        return has_vehicle, min_distance

    def check_pedestrian_ahead(self) -> Tuple[bool, float]:
        """Check if there is a pedestrian ahead

        Returns:
            (has_pedestrian, distance) tuple
        """
        vehicle_loc = self.vehicle.get_transform().location
        vehicle_fwd = self.vehicle.get_transform().get_forward_vector()

        # Get all pedestrians
        walkers = self.world.get_actors().filter("walker.*")

        min_distance = float("inf")
        has_pedestrian = False

        for walker in walkers:
            walker_loc = walker.get_transform().location

            dx = walker_loc.x - vehicle_loc.x
            dy = walker_loc.y - vehicle_loc.y
            distance = math.sqrt(dx**2 + dy**2)

            if distance > self.pedestrian_range:
                continue

            # Check if ahead
            dot = dx * vehicle_fwd.x + dy * vehicle_fwd.y
            if dot > 0 and distance < min_distance:
                min_distance = distance
                has_pedestrian = True

        return has_pedestrian, min_distance

    def check_interrupts(self) -> InterruptReason:
        """Check if current execution needs to be interrupted

        Returns:
            Interrupt reason
        """
        # 1. Collision detection (highest priority)
        if self.collision_flag:
            self.collision_flag = False
            return InterruptReason.COLLISION

        # 2. Red light detection
        if self.check_traffic_light():
            return InterruptReason.RED_LIGHT

        # 3. Pedestrian detection
        has_pedestrian, ped_dist = self.check_pedestrian_ahead()
        if has_pedestrian and ped_dist < 8.0:
            return InterruptReason.PEDESTRIAN

        # 4. Vehicle ahead detection
        has_vehicle, veh_dist = self.check_vehicle_ahead()
        if has_vehicle and veh_dist < self.safe_distance:
            return InterruptReason.VEHICLE_AHEAD

        # 5. Lane invasion
        if self.lane_invasion_flag:
            self.lane_invasion_flag = False
            # Lane invasion does not interrupt, just record

        return InterruptReason.NONE

    def compute_control(self) -> "carla.VehicleControl":
        """Compute single frame control command

        Returns:
            carla.VehicleControl
        """
        control = carla.VehicleControl()

        if self.current_mode == "idle":
            control.throttle = 0.0
            control.brake = 0.3
            control.steer = 0.0
            return control

        if self.current_mode == "stop":
            speed = self.get_current_speed()
            control.throttle = 0.0
            control.brake = min(1.0, speed / 10.0 + 0.5)
            control.steer = 0.0
            return control

        # === Longitudinal control (speed) ===
        current_speed = self.get_current_speed()

        # Adjust target speed based on ahead conditions
        effective_target_speed = self.target_speed

        # Red light deceleration
        if self.check_traffic_light():
            effective_target_speed = 0.0

        # Vehicle ahead deceleration
        has_vehicle, veh_dist = self.check_vehicle_ahead()
        if has_vehicle and veh_dist < self.safe_distance * 2:
            # Dynamically adjust speed based on distance
            speed_factor = max(
                0.2, (veh_dist - self.safe_distance) / self.safe_distance
            )
            effective_target_speed = min(
                effective_target_speed, current_speed * speed_factor
            )

        # Pedestrian ahead deceleration
        has_pedestrian, ped_dist = self.check_pedestrian_ahead()
        if has_pedestrian:
            effective_target_speed = min(effective_target_speed, 10.0)

        # PID speed control
        speed_error = effective_target_speed - current_speed
        throttle_cmd = self.speed_pid.step(speed_error, self.dt)

        if throttle_cmd >= 0:
            control.throttle = min(0.8, max(0.0, throttle_cmd * 0.1))
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(1.0, max(0.0, abs(throttle_cmd) * 0.15))

        # === Lateral control (steering) ===
        # Dynamic look-ahead distance
        look_ahead = max(5.0, current_speed * 0.5)
        target_wp = self.get_target_waypoint(look_ahead)

        if target_wp is None:
            control.steer = 0.0
            return control

        # Calculate steering angle
        vehicle_transform = self.vehicle.get_transform()
        vehicle_loc = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

        target_loc = target_wp.transform.location

        dx = target_loc.x - vehicle_loc.x
        dy = target_loc.y - vehicle_loc.y
        target_yaw = math.atan2(dy, dx)

        yaw_error = target_yaw - vehicle_yaw
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        # Turn mode increases steering
        if self.current_mode == "turn_left":
            yaw_error -= 0.2  # Bias toward left turn
        elif self.current_mode == "turn_right":
            yaw_error += 0.2  # Bias toward right turn

        steer_cmd = self.steer_pid.step(yaw_error, self.dt)
        control.steer = max(-1.0, min(1.0, steer_cmd))

        return control

    def run_one_frame(self) -> Tuple["carla.VehicleControl", InterruptReason]:
        """Execute single frame control

        Returns:
            (control, interrupt_reason) tuple
        """
        self.frames_executed += 1

        # Check interrupts
        interrupt = self.check_interrupts()
        if interrupt != InterruptReason.NONE:
            self.last_interrupt_reason = interrupt
            # Execute emergency brake on interrupt
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 1.0
            control.steer = 0.0
            return control, interrupt

        # Normal control
        control = self.compute_control()
        return control, InterruptReason.NONE


class System1Executor:
    """
    System 1 Executor

    Responsible for executing VLM high-level commands, supports long steps and interrupt mechanism
    """

    # Default configuration
    DEFAULT_MAX_FRAMES = 200  # Default maximum frames
    DEFAULT_MIN_FRAMES = 20  # Minimum frames (ensure basic execution)
    INTERRUPT_COOLDOWN = 10  # Cooldown frames after interrupt

    def __init__(
        self,
        vehicle: "carla.Vehicle",
        world: "carla.World",
        world_map: "carla.Map",
        rgb_camera: Any = None,
    ):
        self.vehicle = vehicle
        self.world = world
        self.map = world_map
        self._rgb_camera = rgb_camera

        # Create local planner
        self.local_planner = LocalPlannerSystem1(vehicle, world, world_map)

        # Execution statistics
        self.total_frames_executed = 0
        self.execution_history: List[Dict[str, Any]] = []
        self.traffic_light_controller = None
        self._last_action_blocked_by_red = False

    def _is_red_light(self) -> bool:
        """Check if the traffic light controller currently indicates red."""
        if self.traffic_light_controller is None:
            return False
        return getattr(self.traffic_light_controller, "is_red", False)

    def _find_nearest_red_light_ahead(self) -> Optional[Tuple[float, Any]]:
        """    (XY)          ；  interact_carla   ：  RGB             """
        if not (self.traffic_light_controller and self.traffic_light_controller.enabled):
            return None
        if self._rgb_camera is not None:
            transform = self._rgb_camera.get_transform()
        else:
            transform = self.vehicle.get_transform()
        ref_loc = transform.location
        fwd = transform.get_forward_vector()
        hlen = math.sqrt(fwd.x * fwd.x + fwd.y * fwd.y) or 1.0
        fwd_x, fwd_y = fwd.x / hlen, fwd.y / hlen

        min_dist = float("inf")
        nearest_loc = None
        for tl in self.traffic_light_controller.traffic_lights:
            tl_loc = tl.get_transform().location
            dx = tl_loc.x - ref_loc.x
            dy = tl_loc.y - ref_loc.y
            forward_dist = dx * fwd_x + dy * fwd_y
            if forward_dist <= 0:
                continue
            if forward_dist < min_dist:
                min_dist = forward_dist
                nearest_loc = tl_loc
        if nearest_loc is None:
            return None
        return min_dist, nearest_loc

    def execute_action(
        self,
        action_name: str,
        parameters: Dict[str, Any],
        collision_callback: callable = None,
        lane_invasion_callback: callable = None,
    ) -> Dict[str, Any]:
        """Execute semantic action

        Args:
            action_name: Action name (FollowLane, TurnLeft, Stop, etc.)
            parameters: Action parameters (distance, etc.)
            collision_callback: Collision callback (for setting collision_flag)
            lane_invasion_callback: Lane invasion callback

        Returns:
            Execution result dictionary
        """
        start_position = self.vehicle.get_transform().location
        completed = True
        self._last_action_blocked_by_red = False

        # Align runtime action support with interact/evaluate action space.
        #       : 2.5 / 5 / 10  （  prompt   interact_carla   ） 
        if action_name in {"FollowLane", "TeleportForward"}:
            distance = self._snap_lane_distance(parameters.get("distance", 10.0))
            print(f"\n🚗 System 1 Executing: {action_name}({distance}m)")
            print(f"   Before: [{start_position.x:.2f}, {start_position.y:.2f}]")
            self._teleport_forward(distance)

        elif action_name == "GoStraight":
            # Straight-through behavior: prefer short forward hop when crossing junction.
            distance = self._snap_lane_distance(parameters.get("distance", 10.0))
            print(f"\n🚗 System 1 Executing: {action_name}({distance}m)")
            self._teleport_forward(distance)

        elif action_name == "Nudge":
            distance = float(parameters.get("distance", 3.0))
            print(f"\n🚗 System 1 Executing: {action_name}({distance:.0f}m)")
            self._teleport_forward(distance)

        elif action_name == "SpeedUp":
            distance = float(parameters.get("distance", 10.0))
            print(f"\n🚗 System 1 Executing: {action_name}({distance:.0f}m)")
            self._teleport_forward(distance)

        elif action_name == "SlowDown":
            distance = float(parameters.get("distance", 3.0))
            print(f"\n🚗 System 1 Executing: {action_name}({distance:.0f}m)")
            self._teleport_forward(distance)

        elif action_name in {"TeleportTurnLeft", "TurnLeft"}:
            print(f"\n🚗 System 1 Executing: {action_name}")
            self._teleport_turn_at_junction("left")

        elif action_name in {"TeleportTurnRight", "TurnRight"}:
            print(f"\n🚗 System 1 Executing: {action_name}")
            self._teleport_turn_at_junction("right")

        elif action_name == "ChangeLaneLeft":
            print(f"\n🚗 System 1 Executing: {action_name}")
            completed = self._teleport_change_lane("left")

        elif action_name == "ChangeLaneRight":
            print(f"\n🚗 System 1 Executing: {action_name}")
            completed = self._teleport_change_lane("right")

        elif action_name == "Stop":
            print(f"\n🚗 System 1 Executing: {action_name}")
            self._stop_vehicle()

        else:
            print(f"\n⚠️ Unknown action: {action_name}")
            completed = False

        end_position = self.vehicle.get_transform().location
        distance_traveled = math.sqrt(
            (end_position.x - start_position.x) ** 2
            + (end_position.y - start_position.y) ** 2
        )

        if self._last_action_blocked_by_red:
            completed = False

        result = {
            "action": action_name,
            "distance_traveled": distance_traveled,
            "completed": completed,
            "blocked_by_red_light": self._last_action_blocked_by_red,
        }
        self.execution_history.append(result)
        print(f"   ✓ Complete: {distance_traveled:.1f}m (completed={completed})")

        return result

    @staticmethod
    def _snap_lane_distance(value) -> float:
        """   /replay              {2.5, 5.0, 10.0}        """
        from envs.carla.step_sizes import resolve_step_distance

        return resolve_step_distance(value)

    def _teleport_forward(self, distance: float = 10.0):
        """              scripts/carla/interact_carla.py teleport_forward   ：
            hop（      2m）         ， **    **，    “   8m   ”      
        """
        start_transform = self.vehicle.get_transform()
        start_location = start_transform.location

        current_wp = self.map.get_waypoint(
            start_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        if current_wp is None:
            print("❌ Cannot get current waypoint")
            return

        effective_distance = math.ceil(distance / 2.0) * 2.0
        red_light = self._is_red_light()
        already_in_junction = current_wp.is_junction

        #   interact_carla.teleport_forward   ：          ，    hop          ，
        #                effective_distance       stop_margin 
        if red_light and not already_in_junction:
            half_length = self.vehicle.bounding_box.extent.x
            stop_margin = 2.0
            scan_range = effective_distance + half_length + stop_margin
            scan_wp = current_wp
            scan_dist = 0.0
            has_junction_ahead = False
            while scan_dist < scan_range:
                _nwps = scan_wp.next(2.0)
                if not _nwps:
                    break
                scan_wp = _nwps[0]
                scan_dist += 2.0
                if scan_wp.is_junction:
                    has_junction_ahead = True
                    break

            if has_junction_ahead:
                light_info = self._find_nearest_red_light_ahead()
                if light_info is not None:
                    light_dist, _ = light_info
                    if light_dist > stop_margin:
                        remaining_to_stop = light_dist - stop_margin
                        if effective_distance > remaining_to_stop:
                            print(
                                "🛑 Red light: hop exceeds safe margin before light — "
                                "holding position (same rule as interact_carla)"
                            )
                            self._last_action_blocked_by_red = True
                            return
                else:
                    print(
                        "🛑 Red light: junction ahead, no light ahead — "
                        "holding position (same rule as interact_carla)"
                    )
                    self._last_action_blocked_by_red = True
                    return

        accumulated = 0.0
        target_wp = current_wp

        while accumulated < distance:
            next_wps = target_wp.next(2.0)
            if not next_wps:
                break
            candidate = next_wps[0]
            target_wp = candidate
            accumulated += 2.0

        target_transform = target_wp.transform
        target_transform.location.z += 0.5
        self.vehicle.set_transform(target_transform)

        for _ in range(5):
            self.world.tick()

    def _teleport_turn_at_junction(self, direction: str):
        """Teleport turn at junction. Blocked when traffic light is red."""
        if self._is_red_light():
            stop_margin = 2.0
            light_info = self._find_nearest_red_light_ahead()
            if not (
                light_info is not None and light_info[0] <= stop_margin
            ):
                print(f"Red light: cannot turn {direction} at junction")
                self._last_action_blocked_by_red = True
                return

        start_transform = self.vehicle.get_transform()
        start_location = start_transform.location
        current_yaw = start_transform.rotation.yaw
        # Normalize to [-180, 180]
        while current_yaw > 180:
            current_yaw -= 360
        while current_yaw < -180:
            current_yaw += 360

        # Get current waypoint
        current_wp = self.map.get_waypoint(
            start_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if current_wp is None:
            print("❌ Cannot get current waypoint")
            return

        # ── Step 1: find the junction ────────────────────────────────────────
        if current_wp.is_junction:
            # Already inside the junction
            junction = current_wp.get_junction()
        else:
            junction = None
            search_wp = current_wp
            accumulated = 0.0
            while accumulated < 50.0:
                next_wps = search_wp.next(2.0)
                if not next_wps:
                    break
                search_wp = next_wps[0]
                accumulated += 2.0
                if search_wp.is_junction:
                    junction = search_wp.get_junction()
                    break

        if junction is None:
            print("❌ No junction found in the next 50 m")
            return

        # ── Step 2: enumerate all (entry, exit) pairs ────────────────────────
        waypoint_pairs = junction.get_waypoints(carla.LaneType.Driving)
        if not waypoint_pairs:
            print("❌ Junction returned no waypoint pairs")
            return

        # ── Step 3: find the entry closest to our current position ───────────
        min_dist = float("inf")
        best_entry_wp = None
        for entry_wp, _ in waypoint_pairs:
            d = math.hypot(
                start_location.x - entry_wp.transform.location.x,
                start_location.y - entry_wp.transform.location.y,
            )
            if d < min_dist:
                min_dist = d
                best_entry_wp = entry_wp

        if best_entry_wp is None:
            print("❌ Cannot determine junction entry point")
            return

        # ── Step 4: pick the exit that matches the desired turn direction ─────
        # left  turn => negative yaw delta  => target range (-135, -30)
        # right turn => positive yaw delta  => target range (  30, 135)
        if direction == "left":
            angle_lo, angle_hi = -135, -30
            ideal = -60
        elif direction == "right":
            angle_lo, angle_hi = 30, 135
            ideal = 60
        else:
            # Straight through the junction
            angle_lo, angle_hi = -45, 45
            ideal = 0

        candidates = []
        for entry_wp, exit_wp in waypoint_pairs:
            if (
                math.hypot(
                    entry_wp.transform.location.x - best_entry_wp.transform.location.x,
                    entry_wp.transform.location.y - best_entry_wp.transform.location.y,
                )
                > 5.0
            ):
                continue

            exit_yaw = exit_wp.transform.rotation.yaw
            while exit_yaw > 180:
                exit_yaw -= 360
            while exit_yaw < -180:
                exit_yaw += 360

            angle_diff = exit_yaw - current_yaw
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360

            if angle_lo <= angle_diff <= angle_hi:
                candidates.append(
                    {"exit_wp": exit_wp, "score": abs(angle_diff - ideal)}
                )

        if not candidates:
            print(f"❌ No suitable {direction} exit found at junction")
            return

        candidates.sort(key=lambda x: x["score"])
        target_wp = candidates[0]["exit_wp"]

        # ── Step 5: advance until we leave the junction area ─────────────────
        final_wp = target_wp
        for _ in range(20):
            if not final_wp.is_junction:
                break
            next_wps = final_wp.next(2.0)
            if not next_wps:
                break
            final_wp = next_wps[0]

        # ── Step 6: teleport ─────────────────────────────────────────────────
        target_transform = final_wp.transform
        target_transform.location.z += 0.5
        self.vehicle.set_transform(target_transform)

        for _ in range(5):
            self.world.tick()

    def _teleport_change_lane(self, direction: str) -> bool:
        """Teleport to adjacent driving lane in the same direction."""
        transform = self.vehicle.get_transform()
        location = transform.location

        current_wp = self.map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if current_wp is None:
            print("❌ Cannot get current waypoint for lane change")
            return False

        target_wp = (
            current_wp.get_left_lane() if direction == "left" else current_wp.get_right_lane()
        )
        if target_wp is None:
            print(f"❌ No available {direction} lane")
            return False

        if target_wp.lane_type != carla.LaneType.Driving:
            print(f"❌ Target {direction} lane is not driving lane")
            return False

        # Prevent switching to opposite-direction lane
        current_sign = 1 if current_wp.lane_id > 0 else -1
        target_sign = 1 if target_wp.lane_id > 0 else -1
        if current_sign != target_sign:
            print("❌ Target lane is opposite direction, lane change canceled")
            return False

        target_transform = target_wp.transform
        target_transform.location.z += 0.5
        self.vehicle.set_transform(target_transform)
        self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))

        for _ in range(5):
            self.world.tick()
        return True

    def _stop_vehicle(self):
        """Force stop and tick several frames for stabilization."""
        self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        self.vehicle.apply_control(control)
        for _ in range(5):
            self.world.tick()

    def set_collision_flag(self):
        """Set collision flag"""
        self.local_planner.collision_flag = True

    def set_lane_invasion_flag(self):
        """Set lane invasion flag"""
        self.local_planner.lane_invasion_flag = True
