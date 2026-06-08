def snap_init_location_z(world, location, carla_module, z_above_nav=1.0):
    carla = carla_module
    try:
        lane_any = carla.LaneType.Any
    except AttributeError:
        lane_any = carla.LaneType.Driving
    map_ = world.get_map()
    try:
        probe = carla.Location(location.x, location.y, location.z + 50.0)
        wp = map_.get_waypoint(probe, project_to_road=True, lane_type=lane_any)
    except Exception:
        return location
    if wp is None:
        return location
    nav_z = wp.transform.location.z
    target_z = nav_z + z_above_nav
    if location.z < nav_z + 0.35:
        return carla.Location(location.x, location.y, target_z)
    return location
