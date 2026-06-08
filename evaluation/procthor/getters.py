"""
Data Retrieval Functions
Extract data needed for evaluation from environment
"""
from typing import Dict, Any, List, Optional, Tuple
import math


def _point_in_polygon(point_x: float, point_z: float, polygon: List[Dict[str, float]]) -> bool:
    """          （     /    ）
    
    Args:
        point_x:   X  
        point_z:   Z  
        polygon:        ，     {"x": float, "y": int, "z": float}
        
    Returns:
        bool:         
    """
    if not polygon or len(polygon) < 3:
        return False
    
    #      （Ray Casting Algorithm）
    #     （X   ）      ，           
    #      ，      ；     ，      
    
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi = polygon[i].get("x", 0)
        zi = polygon[i].get("z", 0)
        xj = polygon[j].get("x", 0)
        zj = polygon[j].get("z", 0)
        
        #         (Pi, Pj)  
        #    (point_x, point_z)  （X   ）  
        
        #            Z  
        if ((zi > point_z) != (zj > point_z)):
            #       Z  ，      
            
            #      （xi == xj）     
            if xi == xj:
                #    ：        ，       
                if point_x < xi:
                    inside = not inside
            else:
                #     ：     X  
                x_intersection = (point_z - zi) * (xj - xi) / (zj - zi) + xi
                #          （X  ），      
                if point_x < x_intersection:
                    inside = not inside
        
        j = i
    
    return inside


def _build_room_boundaries_from_house_scene(env: Any) -> Optional[Dict[str, List[List[Dict[str, float]]]]]:
    """ ProcTHOR house              （      ）
    
    ProcTHOR house    rooms  ，  room floorPolygon  ，
               ，           
    
    Args:
        env: Environment instance (should have scene attribute)
        
    Returns:
        {room_type: [polygons]}   None（    house  ）
    """
    #   env   scene  （ProcTHOR house  ）
    if not hasattr(env, 'scene') or not env.scene:
        return None
    
    house = env.scene
    
    #   house   rooms 
    if not isinstance(house, dict) or 'rooms' not in house:
        return None
    
    rooms = house.get('rooms', [])
    if not rooms:
        return None
    
    #              
    room_polygons = {}
    
    for room in rooms:
        room_type = room.get('roomType')
        floor_polygon = room.get('floorPolygon')
        
        if room_type and floor_polygon and isinstance(floor_polygon, list):
            #               ，      （               ）
            if room_type not in room_polygons:
                room_polygons[room_type] = []
            room_polygons[room_type].append(floor_polygon)
    
    return room_polygons if room_polygons else None


def get_object_state(env: Any, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get state list of target objects
    
    Args:
        env: Environment instance
        metadata: Environment metadata
        
    Returns:
        Target object state list
    """
    target_types = getattr(env, "target_object_types", [])
    objects = metadata.get("objects", [])
    
    target_objects = [
        obj for obj in objects
        if obj["objectType"] in target_types
    ]
    
    return target_objects


def get_agent_position(env: Any, metadata: Dict[str, Any]) -> Dict[str, float]:
    """Get agent position
    
    Args:
        env: Environment instance
        metadata: Environment metadata
        
    Returns:
        Position dictionary {x, y, z}
    """
    return metadata.get("agent", {}).get("position", {})


def get_inventory_objects(env: Any, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get objects held in hand
    
    Args:
        env: Environment instance
        metadata: Environment metadata
        
    Returns:
        Objects in hand list
    """
    return metadata.get("inventoryObjects", [])


def get_agent_room(env: Any, metadata: Dict[str, Any]) -> Optional[str]:
    """Get the room type where the agent is currently located
    
      ProcTHOR floorPolygon（     ）            
            ，        ，          
    
    Args:
        env: Environment instance (must have scene attribute with house object)
        metadata: Environment metadata
        
    Returns:
        Room type string (e.g., "Kitchen", "Bathroom", "Bedroom"), or None if cannot determine
    """
    agent = metadata.get("agent", {})
    agent_pos = agent.get("position", {})
    
    if not agent_pos:
        return None
    
    agent_x = agent_pos.get("x", 0)
    agent_z = agent_pos.get("z", 0)
    
    #   ProcTHOR floorPolygon（   ，        ）
    room_polygons = _build_room_boundaries_from_house_scene(env)
    
    if not room_polygons:
        return None
    
    #        ：               
    matching_rooms = []
    for room_type, polygons in room_polygons.items():
        #               （        ）
        for polygon in polygons:
            if _point_in_polygon(agent_x, agent_z, polygon):
                matching_rooms.append(room_type)
                break  #             
    
    if matching_rooms:
        #           （     ，     ），     
        return matching_rooms[0]
    
    return None


def get_room_boundaries(env: Any, metadata: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """            （     ）
    
      ProcTHOR floorPolygon          （min_x, max_x, min_z, max_z） 
           ，       （            ） 
    
    Args:
        env: Environment instance
        metadata: Environment metadata (not used, but kept for compatibility)
        
    Returns:
        {room_type: {"min_x": float, "max_x": float, "min_z": float, "max_z": float, "polygon_count": int}}
    """
    room_polygons = _build_room_boundaries_from_house_scene(env)
    
    if not room_polygons:
        return {}
    
    room_boundaries = {}
    
    for room_type, polygons in room_polygons.items():
        #            
        all_x = []
        all_z = []
        
        for polygon in polygons:
            for vertex in polygon:
                all_x.append(vertex.get("x", 0))
                all_z.append(vertex.get("z", 0))
        
        if all_x and all_z:
            room_boundaries[room_type] = {
                "min_x": min(all_x),
                "max_x": max(all_x),
                "min_z": min(all_z),
                "max_z": max(all_z),
                "polygon_count": len(polygons)
            }
    
    return room_boundaries


def get_object_room(env: Any, metadata: Dict[str, Any]) -> Optional[str]:
    """Get the room type where a specific object is located
    
    Args:
        env: Environment instance (should have target_object_types or condition info)
        metadata: Environment metadata
        
    Returns:
        Room type string, or None if object not found
    """
    #   env   metadata          
    #           ，      condition     object_type
    #                 
    target_types = getattr(env, "target_object_types", [])
    all_objects = metadata.get("objects", [])
    
    #          
    for obj in all_objects:
        obj_type = obj.get("objectType", "")
        if obj_type in target_types or any(obj_type.startswith(t + "|") for t in target_types):
            room_type = obj.get("roomType")
            if room_type:
                return room_type
    
    return None
