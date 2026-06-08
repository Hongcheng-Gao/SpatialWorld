"""
Data Retrieval Functions
Extract data needed for evaluation from environment
"""
from typing import Dict, Any, List


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
