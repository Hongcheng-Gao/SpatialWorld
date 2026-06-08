"""
CARLA Vehicle Controller
Implements PID controller for lane keeping and speed control

Reference: CARLA official PID controller implementation
"""

import math
from typing import Optional, Tuple
from collections import deque

try:
    import carla
except ImportError:
    carla = None


class PIDController:
    """PID Controller Base Class"""

    def __init__(self, Kp: float, Ki: float, Kd: float, dt: float = 0.05):
        """
        Args:
            Kp: Proportional coefficient
            Ki: Integral coefficient
            Kd: Derivative coefficient
            dt: Time step
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt

        self._error_buffer = deque(maxlen=30)
        self._integral = 0.0
        self._previous_error = 0.0

    def step(self, error: float) -> float:
        """Execute one step of PID calculation

        Args:
            error: Current error

        Returns:
            Control output
        """
        self._error_buffer.append(error)

        # Integral term (with anti-windup)
        self._integral += error * self.dt
        self._integral = max(-10.0, min(10.0, self._integral))

        # Derivative term
        if len(self._error_buffer) >= 2:
            derivative = (error - self._previous_error) / self.dt
        else:
            derivative = 0.0

        self._previous_error = error

        # PID output
        output = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        return output

    def reset(self):
        """Reset controller state"""
        self._error_buffer.clear()
        self._integral = 0.0
        self._previous_error = 0.0


class VehicleController:
    """CARLA Vehicle Controller

    Integrates longitudinal control (speed) and lateral control (steering)
    """

    def __init__(
        self, vehicle: "carla.Vehicle", world_map: "carla.Map", dt: float = 0.05
    ):
        """
        Args:
            vehicle: CARLA vehicle object
            world_map: CARLA map object
            dt: Control period
        """
        self.vehicle = vehicle
        self.map = world_map
        self.dt = dt

        # Longitudinal PID (speed control)
        self.speed_controller = PIDController(Kp=1.0, Ki=0.1, Kd=0.05, dt=dt)

        # Lateral PID (lane keeping - steering control)
        self.steering_controller = PIDController(Kp=1.5, Ki=0.0, Kd=0.3, dt=dt)

        # Target state
        self.target_speed: float = 0.0  # km/h
        self.target_waypoint: Optional["carla.Waypoint"] = None

        # Control mode
        self.mode: str = "idle"  # idle, follow_lane, turn, stop, nudge
        self.turn_direction: Optional[str] = None  # left, right, straight

    def set_target_speed(self, speed_kmh: float):
        """Set target speed"""
        self.target_speed = max(0.0, min(120.0, speed_kmh))

    def set_mode(self, mode: str, **kwargs):
        """Set control mode

        Args:
            mode: Control mode (idle, follow_lane, turn, stop, nudge)
            **kwargs: Mode parameters
        """
        self.mode = mode

        if mode == "turn":
            self.turn_direction = kwargs.get("direction", "straight")
        elif mode == "follow_lane":
            self.target_speed = kwargs.get("speed", 30.0)
        elif mode == "stop":
            self.target_speed = 0.0
        elif mode == "nudge":
            # Creep mode: low speed forward
            self.target_speed = kwargs.get("speed", 10.0)

    def get_current_speed(self) -> float:
        """Get current speed (km/h)"""
        velocity = self.vehicle.get_velocity()
        speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed_ms * 3.6

    def get_current_waypoint(self) -> Optional["carla.Waypoint"]:
        """Get nearest waypoint at current location"""
        location = self.vehicle.get_transform().location
        return self.map.get_waypoint(location, project_to_road=True)

    def get_next_waypoint(self, distance: float = 5.0) -> Optional["carla.Waypoint"]:
        """Get waypoint at specified distance ahead

        Args:
            distance: Forward distance (meters)

        Returns:
            Waypoint
        """
        current_wp = self.get_current_waypoint()
        if current_wp is None:
            return None

        # Select next waypoint based on mode
        if self.mode == "turn" and self.turn_direction:
            # Junction turning
            next_wps = current_wp.next(distance)
            if not next_wps:
                return None

            # Check if in junction
            if current_wp.is_junction:
                # In junction, find waypoint in corresponding direction
                for wp in next_wps:
                    lane_change = self._get_turn_direction(current_wp, wp)
                    if lane_change == self.turn_direction:
                        return wp

            # Not in junction or didn't find corresponding direction, return first
            return next_wps[0]
        else:
            # Default: follow current lane
            next_wps = current_wp.next(distance)
            return next_wps[0] if next_wps else None

    def _get_turn_direction(
        self, current_wp: "carla.Waypoint", next_wp: "carla.Waypoint"
    ) -> str:
        """Determine turn direction"""
        current_yaw = current_wp.transform.rotation.yaw
        next_yaw = next_wp.transform.rotation.yaw

        angle_diff = next_yaw - current_yaw
        # Normalize to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360

        # CARLA / UE4 left-handed coordinate system:
        #   positive yaw delta  => clockwise from above => RIGHT turn
        #   negative yaw delta  => counter-clockwise    => LEFT  turn
        if angle_diff > 30:
            return "right"
        elif angle_diff < -30:
            return "left"
        else:
            return "straight"

    def compute_control(self) -> "carla.VehicleControl":
        """Compute vehicle control command

        Returns:
            carla.VehicleControl object
        """
        control = carla.VehicleControl()

        # Get current state
        current_speed = self.get_current_speed()

        if self.mode == "idle":
            # Idle mode: maintain stationary
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 0.3
            return control

        elif self.mode == "stop":
            # Stop mode
            if current_speed > 1.0:
                control.throttle = 0.0
                control.steer = 0.0
                control.brake = min(
                    1.0, current_speed / 10.0
                )  # Adjust brake force based on speed
            else:
                control.throttle = 0.0
                control.steer = 0.0
                control.brake = 1.0
                control.hand_brake = True
            return control

        # === Longitudinal control (speed) ===
        speed_error = self.target_speed - current_speed
        throttle_cmd = self.speed_controller.step(speed_error)

        if throttle_cmd >= 0:
            control.throttle = min(1.0, throttle_cmd * 0.1)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(1.0, abs(throttle_cmd) * 0.1)

        # === Lateral control (steering) ===
        # Get target waypoint
        look_ahead_distance = max(
            5.0, current_speed * 0.5
        )  # Faster speed, look farther ahead
        target_wp = self.get_next_waypoint(look_ahead_distance)

        if target_wp is None:
            control.steer = 0.0
            return control

        # Calculate steering error (angle between vehicle heading and target direction)
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

        target_location = target_wp.transform.location

        # Calculate target direction
        dx = target_location.x - vehicle_location.x
        dy = target_location.y - vehicle_location.y
        target_yaw = math.atan2(dy, dx)

        # Calculate heading error
        yaw_error = target_yaw - vehicle_yaw
        # Normalize to [-pi, pi]
        while yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        while yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        # PID calculate steering
        steer_cmd = self.steering_controller.step(yaw_error)
        control.steer = max(-1.0, min(1.0, steer_cmd))

        return control

    def reset(self):
        """Reset controller"""
        self.speed_controller.reset()
        self.steering_controller.reset()
        self.mode = "idle"
        self.target_speed = 0.0
        self.turn_direction = None


class SemanticAction:
    """Semantic action definition

    Converts high-level commands to controller modes
    """

    # Mapping from action names to control modes
    ACTION_MAP = {
        "FollowLane": {
            "mode": "follow_lane",
            "default_speed": 30.0,
            "description": "Drive along current lane",
        },
        "ChangeLaneLeft": {
            "mode": "follow_lane",
            "lane_change": "left",
            "default_speed": 25.0,
            "description": "Change lane to left",
        },
        "ChangeLaneRight": {
            "mode": "follow_lane",
            "lane_change": "right",
            "default_speed": 25.0,
            "description": "Change lane to right",
        },
        "TurnLeft": {
            "mode": "turn",
            "direction": "left",
            "default_speed": 15.0,
            "description": "Turn left at junction",
        },
        "TurnRight": {
            "mode": "turn",
            "direction": "right",
            "default_speed": 15.0,
            "description": "Turn right at junction",
        },
        "GoStraight": {
            "mode": "turn",
            "direction": "straight",
            "default_speed": 25.0,
            "description": "Go straight at junction",
        },
        "Stop": {"mode": "stop", "default_speed": 0.0, "description": "Stop"},
        "Nudge": {"mode": "nudge", "default_speed": 5.0, "description": "Creep/adjust"},
        "SpeedUp": {
            "mode": "follow_lane",
            "speed_delta": 10.0,
            "description": "Accelerate",
        },
        "SlowDown": {
            "mode": "follow_lane",
            "speed_delta": -10.0,
            "description": "Decelerate",
        },
    }

    @classmethod
    def get_action_info(cls, action_name: str) -> dict:
        """Get action information"""
        return cls.ACTION_MAP.get(
            action_name, {"mode": "idle", "description": "Unknown action"}
        )

    @classmethod
    def get_all_actions(cls) -> list:
        """Get all available actions"""
        return list(cls.ACTION_MAP.keys())
