"""
Evaluation Functions
Compare actual results with expected results
"""
from typing import Dict, Any, List, Optional


def check_object_state(
    result: List[Dict[str, Any]],
    expected: Dict[str, Any],
    **options
) -> float:
    """Check if object state matches expected
    
    Args:
        result: Object state list
        expected: Expected state {field, value, any}
        **options: Additional options
        
    Returns:
        Score 0.0 or 1.0
    """
    if not result:
        return 0.0
    
    field = expected.get("field", "isOpen")
    value = expected.get("value", True)
    any_match = expected.get("any", True)
    
    matches = [
        obj for obj in result
        if obj.get(field) == value
    ]
    
    if any_match:
        # Any one match is sufficient
        return 1.0 if matches else 0.0
    else:
        # All need to match
        return 1.0 if len(matches) == len(result) else 0.0


def check_object_in_hand(
    result: List[Dict[str, Any]],
    expected: Dict[str, Any],
    **options
) -> float:
    """Check if holding specified object
    
    Args:
        result: Objects in hand list
        expected: Expected object {object_type}
        **options: Additional options
        
    Returns:
        Score 0.0 or 1.0
    """
    target_type = expected.get("object_type")
    if not target_type:
        return 0.0
    
    for obj in result:
        if obj.get("objectType") == target_type:
            return 1.0
    
    return 0.0


def check_object_in_receptacle(
    result: List[Dict[str, Any]],
    expected: Dict[str, Any],
    **options
) -> float:
    """Check if object is in specified receptacle
    
    Args:
        result: Object state list
        expected: Expected {object_type, receptacle_type, count, value}
            - object_type: Target object type
            - receptacle_type: Target receptacle type
            - count: Required number of objects (default 1)
            - value: True means should be in receptacle, False means should NOT be in receptacle (default True)
        **options: Additional options
        
    Returns:
        Score 0.0 or 1.0
    """
    object_type = expected.get("object_type")
    receptacle_type = expected.get("receptacle_type")
    required_count = expected.get("count", 1)
    expected_value = expected.get("value", True)  # Default True for backward compatibility
    
    if not object_type or not receptacle_type:
        return 0.0
    
    # Find target objects
    target_objects = [
        obj for obj in result
        if obj.get("objectType") == object_type
    ]
    
    if not target_objects:
        # If no target objects found and we expect them NOT to be in receptacle, that's success
        return 1.0 if not expected_value else 0.0
    
    # Count objects in specified receptacle
    count_in_receptacle = 0
    for obj in target_objects:
        parent_receptacles = obj.get("parentReceptacles", [])
        for parent_id in parent_receptacles:
            parent_type = parent_id.split('|')[0] if '|' in parent_id else parent_id
            if parent_type == receptacle_type:
                count_in_receptacle += 1
                break  # Each object only counts once
    
    # Check based on expected_value
    if expected_value:
        # True: should be in receptacle
        return 1.0 if count_in_receptacle >= required_count else 0.0
    else:
        # False: should NOT be in receptacle
        return 1.0 if count_in_receptacle == 0 else 0.0


def check_agent_in_room(
    result: Optional[str],
    expected: Dict[str, Any],
    **options
) -> float:
    """Check if agent is in specified room
    
    Args:
        result: Current room type (from get_agent_room)
        expected: Expected {room_type}
            - room_type: Target room type (e.g., "Kitchen", "Bathroom", "Bedroom")
        **options: Additional options
        
    Returns:
        Score 0.0 or 1.0
    """
    expected_room = expected.get("room_type")
    
    if not expected_room:
        return 0.0
    
    if not result:
        return 0.0
    
    #           
    return 1.0 if result.lower() == expected_room.lower() else 0.0


def check_object_in_room(
    result: Optional[str],
    expected: Dict[str, Any],
    **options
) -> float:
    """Check if object is in specified room
    
    Args:
        result: Object's room type (from get_object_room, but we'll get it from metadata directly)
        expected: Expected {object_type, room_type}
            - object_type: Target object type
            - room_type: Target room type (e.g., "Kitchen", "Bathroom", "Bedroom")
        **options: Additional options (may contain metadata)
        
    Returns:
        Score 0.0 or 1.0
    """
    expected_room = expected.get("room_type")
    object_type = expected.get("object_type")
    
    if not expected_room or not object_type:
        return 0.0
    
    #   options     metadata（    ）
    metadata = options.get("metadata", {})
    if not metadata:
        return 0.0
    
    #     metadata      
    from .semantic_mapping import get_semantic_variants
    object_type_variants = get_semantic_variants(object_type)
    
    all_objects = metadata.get("objects", [])
    for obj in all_objects:
        obj_type = obj.get("objectType")
        if obj_type in object_type_variants:
            obj_room = obj.get("roomType")
            if obj_room and obj_room.lower() == expected_room.lower():
                return 1.0
    
    return 0.0


def check_agent_near_object(
    result: Optional[Dict[str, float]],
    expected: Dict[str, Any],
    **options
) -> float:
    """Check if agent is near specified object
    
    Args:
        result: Agent position {x, y, z} (from get_agent_position)
        expected: Expected {object_type, distance}
            - object_type: Target object type (e.g., "Bed", "Fridge", "Sofa")
            - distance: Maximum distance in meters (default 1.0)
        **options: Additional options (may contain metadata)
        
    Returns:
        Score 0.0 or 1.0
    """
    object_type = expected.get("object_type")
    max_distance = expected.get("distance", 1.0)  # Default 1 meter
    
    if not object_type:
        return 0.0
    
    if not result:
        return 0.0
    
    #   options     metadata（    ）
    metadata = options.get("metadata", {})
    if not metadata:
        return 0.0
    
    #        
    agent_x = result.get("x", 0)
    agent_z = result.get("z", 0)
    
    #   metadata        
    from .semantic_mapping import get_semantic_variants
    object_type_variants = get_semantic_variants(object_type)
    
    all_objects = metadata.get("objects", [])
    for obj in all_objects:
        obj_type = obj.get("objectType")
        if obj_type in object_type_variants:
            obj_pos = obj.get("position", {})
            if not obj_pos:
                continue
            
            obj_x = obj_pos.get("x", 0)
            obj_z = obj_pos.get("z", 0)
            
            #     （2D  ，  Y ）
            dx = agent_x - obj_x
            dz = agent_z - obj_z
            distance = (dx ** 2 + dz ** 2) ** 0.5
            
            if distance <= max_distance:
                return 1.0
    
    return 0.0
