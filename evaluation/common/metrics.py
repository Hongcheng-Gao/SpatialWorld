"""
Evaluation Functions
Compare actual results with expected results
"""
from typing import Dict, Any, List


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


def check_object_state_condition(
    result: List[Dict[str, Any]],
    expected: Dict[str, Any],
    **options
) -> float:
    """Check if object has specific state (isHot, isCool, isClean, etc.)
    
    Args:
        result: Object state list
        expected: Expected {object_type, state, value}
            - object_type: Target object type
            - state: State field name (e.g., "isHot", "isCool", "isClean")
            - value: Expected value (default True)
        **options: Additional options
        
    Returns:
        Score 0.0 or 1.0
    """
    object_type = expected.get("object_type")
    state_field = expected.get("state")
    expected_value = expected.get("value", True)
    
    if not object_type or not state_field:
        return 0.0
    
    # Find target objects
    target_objects = [
        obj for obj in result
        if obj.get("objectType") == object_type
    ]
    
    if not target_objects:
        return 0.0
    
    # Check if any object has the expected state
    for obj in target_objects:
        if obj.get(state_field) == expected_value:
            return 1.0
    
    return 0.0


def check_multiple_conditions(
    result: List[Dict[str, Any]],
    expected: Dict[str, Any],
    **options
) -> float:
    """Check multiple conditions with AND/OR logic
    
    Args:
        result: Object state list (all objects in scene)
        expected: Expected {conditions, logic}
            - conditions: List of condition dicts
            - logic: "AND" or "OR" (default "AND")
        **options: Additional options
        
    Returns:
        Score 0.0 or 1.0
    """
    conditions = expected.get("conditions", [])
    logic = expected.get("logic", "AND").upper()
    
    if not conditions:
        return 0.0
    
    # Evaluate each condition
    scores = []
    for condition in conditions:
        condition_type = condition.get("type", "object_state")
        
        if condition_type == "object_in_receptacle":
            score = check_object_in_receptacle(result, condition, **options)
        elif condition_type == "object_state":
            score = check_object_state_condition(result, condition, **options)
        elif condition_type == "object_in_hand":
            # For object_in_hand, we need inventory objects, not scene objects
            # This should be handled separately
            score = 0.0  # Will be handled by caller
        else:
            score = 0.0
        
        scores.append(score)
    
    # Apply logic
    if logic == "AND":
        return 1.0 if all(s == 1.0 for s in scores) else 0.0
    elif logic == "OR":
        return 1.0 if any(s == 1.0 for s in scores) else 0.0
    else:
        return 0.0
