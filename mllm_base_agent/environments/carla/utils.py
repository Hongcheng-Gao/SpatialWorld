"""
CARLA Environment Utility Functions
Provides auxiliary functions such as distance calculation, collision detection, lane invasion detection
"""

import math
from typing import Tuple, Optional

try:
    import carla
except ImportError:
    carla = None


def calculate_distance_to_waypoint(
    vehicle_location: "carla.Location", waypoint: "carla.Waypoint"
) -> float:
    """Calculate distance from vehicle to waypoint

    Args:
        vehicle_location: Vehicle location (carla.Location)
        waypoint: Target waypoint (carla.Waypoint)

    Returns:
        Distance (meters)
    """
    if carla is None:
        raise ImportError("carla module not installed")

    waypoint_location = waypoint.transform.location

    dx = vehicle_location.x - waypoint_location.x
    dy = vehicle_location.y - waypoint_location.y
    dz = vehicle_location.z - waypoint_location.z

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_distance_between_locations(
    location1: "carla.Location", location2: "carla.Location"
) -> float:
    """Calculate Euclidean distance between two locations

    Args:
        location1: First location
        location2: Second location

    Returns:
        Distance (meters)
    """
    dx = location1.x - location2.x
    dy = location1.y - location2.y
    dz = location1.z - location2.z

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_2d_distance(
    location1: "carla.Location", location2: "carla.Location"
) -> float:
    """Calculate 2D distance between two locations (ignoring height)

    Args:
        location1: First location
        location2: Second location

    Returns:
        2D distance (meters)
    """
    dx = location1.x - location2.x
    dy = location1.y - location2.y

    return math.sqrt(dx * dx + dy * dy)


def detect_collision(vehicle: "carla.Vehicle", collision_history: list) -> bool:
    """Detect vehicle collision

    Args:
        vehicle: Vehicle object
        collision_history: Collision history record list

    Returns:
        Whether collision occurred
    """
    return len(collision_history) > 0


def check_lane_invasion(lane_invasion_history: list) -> bool:
    """Detect lane invasion

    Args:
        lane_invasion_history: Lane invasion history record list

    Returns:
        Whether lane invasion occurred
    """
    return len(lane_invasion_history) > 0


def get_speed_kmh(vehicle: "carla.Vehicle") -> float:
    """Get vehicle speed (km/h)

    Args:
        vehicle: Vehicle object

    Returns:
        Speed (km/h)
    """
    velocity = vehicle.get_velocity()
    speed_ms = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed_ms * 3.6  # Convert to km/h


def get_forward_vector(rotation: "carla.Rotation") -> Tuple[float, float, float]:
    """Get forward vector from rotation

    Args:
        rotation: Rotation angle (carla.Rotation)

    Returns:
        Normalized forward vector (x, y, z)
    """
    yaw_rad = math.radians(rotation.yaw)
    pitch_rad = math.radians(rotation.pitch)

    x = math.cos(pitch_rad) * math.cos(yaw_rad)
    y = math.cos(pitch_rad) * math.sin(yaw_rad)
    z = math.sin(pitch_rad)

    return (x, y, z)


def calculate_angle_to_target(
    vehicle_location: "carla.Location",
    vehicle_rotation: "carla.Rotation",
    target_location: "carla.Location",
) -> float:
    """Calculate angle difference between vehicle heading and target point

    Args:
        vehicle_location: Vehicle location
        vehicle_rotation: Vehicle rotation
        target_location: Target location

    Returns:
        Angle difference (degrees), range [-180, 180]
    """
    # Calculate target direction vector
    dx = target_location.x - vehicle_location.x
    dy = target_location.y - vehicle_location.y

    # Calculate target angle
    target_angle = math.degrees(math.atan2(dy, dx))

    # Get vehicle current heading angle
    vehicle_angle = vehicle_rotation.yaw

    # Calculate angle difference
    angle_diff = target_angle - vehicle_angle

    # Normalize to [-180, 180]
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360

    return angle_diff


def is_vehicle_stopped(vehicle: "carla.Vehicle", threshold: float = 0.1) -> bool:
    """Check if vehicle is stopped

    Args:
        vehicle: Vehicle object
        threshold: Speed threshold (m/s)

    Returns:
        Whether stopped
    """
    velocity = vehicle.get_velocity()
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    return speed < threshold


def get_nearest_waypoint(
    world_map: "carla.Map", location: "carla.Location"
) -> Optional["carla.Waypoint"]:
    """Get nearest waypoint

    Args:
        world_map: CARLA map object
        location: Location

    Returns:
        Nearest waypoint, returns None if failed
    """
    try:
        waypoint = world_map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        return waypoint
    except Exception as e:
        print(f"Failed to get waypoint: {e}")
        return None


def get_lane_center_offset(
    vehicle_location: "carla.Location", waypoint: "carla.Waypoint"
) -> float:
    """Calculate vehicle offset relative to lane center

    Args:
        vehicle_location: Vehicle location
        waypoint: Lane waypoint

    Returns:
        Offset (meters), positive value means right offset, negative means left offset
    """
    # Get waypoint location
    waypoint_location = waypoint.transform.location
    waypoint_rotation = waypoint.transform.rotation

    # Calculate vector from vehicle to waypoint
    dx = vehicle_location.x - waypoint_location.x
    dy = vehicle_location.y - waypoint_location.y

    # Calculate waypoint right vector
    yaw_rad = math.radians(waypoint_rotation.yaw + 90)  # +90 degrees to get right vector
    right_x = math.cos(yaw_rad)
    right_y = math.sin(yaw_rad)

    # Calculate projection (dot product)
    offset = dx * right_x + dy * right_y

    return offset


def calculate_steering_angle(
    vehicle_location: "carla.Location",
    vehicle_rotation: "carla.Rotation",
    target_location: "carla.Location",
    max_steer: float = 1.0,
) -> float:
    """Calculate steering angle based on target position

    This is a simple P controller

    Args:
        vehicle_location: Vehicle location
        vehicle_rotation: Vehicle rotation
        target_location: Target location
        max_steer: Maximum steering value

    Returns:
        Steering value [-max_steer, max_steer]
    """
    # Calculate angle difference
    angle_diff = calculate_angle_to_target(
        vehicle_location, vehicle_rotation, target_location
    )

    # Simple proportional control
    # Assume ±30 degrees corresponds to maximum steering
    steer = angle_diff / 30.0

    # Limit to [-max_steer, max_steer]
    steer = max(-max_steer, min(max_steer, steer))

    return steer


def get_vehicle_bounding_box_corners(vehicle: "carla.Vehicle") -> list:
    """Get 8 corner coordinates of vehicle bounding box

    Args:
        vehicle: Vehicle object

    Returns:
        List of 8 corner world coordinates
    """
    bbox = vehicle.bounding_box
    vehicle_transform = vehicle.get_transform()

    # Half size of bounding box
    extent = bbox.extent

    # Relative positions of 8 corners (vehicle coordinate system)
    corners_local = [
        carla.Location(x=extent.x, y=extent.y, z=extent.z),
        carla.Location(x=extent.x, y=extent.y, z=-extent.z),
        carla.Location(x=extent.x, y=-extent.y, z=extent.z),
        carla.Location(x=extent.x, y=-extent.y, z=-extent.z),
        carla.Location(x=-extent.x, y=extent.y, z=extent.z),
        carla.Location(x=-extent.x, y=extent.y, z=-extent.z),
        carla.Location(x=-extent.x, y=-extent.y, z=extent.z),
        carla.Location(x=-extent.x, y=-extent.y, z=-extent.z),
    ]

    # Transform to world coordinate system
    corners_world = []
    for corner in corners_local:
        # Apply bounding box relative position
        corner = carla.Location(
            x=corner.x + bbox.location.x,
            y=corner.y + bbox.location.y,
            z=corner.z + bbox.location.z,
        )
        # Apply vehicle transform
        world_point = vehicle_transform.transform(corner)
        corners_world.append(world_point)

    return corners_world


def format_location(location: "carla.Location", precision: int = 2) -> str:
    """Format location information as string

    Args:
        location: CARLA Location object
        precision: Decimal precision

    Returns:
        Formatted string
    """
    return f"({location.x:.{precision}f}, {location.y:.{precision}f}, {location.z:.{precision}f})"


def format_rotation(rotation: "carla.Rotation", precision: int = 1) -> str:
    """Format rotation information as string

    Args:
        rotation: CARLA Rotation object
        precision: Decimal precision

    Returns:
        Formatted string
    """
    return f"(pitch={rotation.pitch:.{precision}f}°, yaw={rotation.yaw:.{precision}f}°, roll={rotation.roll:.{precision}f}°)"


def create_vehicle_control(
    throttle: float = 0.0,
    steer: float = 0.0,
    brake: float = 0.0,
    hand_brake: bool = False,
    reverse: bool = False,
) -> "carla.VehicleControl":
    """Create vehicle control object

    Args:
        throttle: Throttle [0, 1]
        steer: Steering [-1, 1]
        brake: Brake [0, 1]
        hand_brake: Hand brake
        reverse: Reverse

    Returns:
        carla.VehicleControl object
    """
    if carla is None:
        raise ImportError("carla module not installed")

    return carla.VehicleControl(
        throttle=float(throttle),
        steer=float(steer),
        brake=float(brake),
        hand_brake=hand_brake,
        reverse=reverse,
        manual_gear_shift=False,
    )
