"""
Evaluator Base Class
Supports both single success_condition and multiple success_conditions with AND/OR logic
   spatial-planning/evaluators/base.py
"""

from typing import Dict, Any, Callable, List
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Task evaluator base class"""

    def __init__(
        self, getter: Callable, metric: Callable, expected: Any = None, **options
    ):
        """Initialize evaluator

        Args:
            getter: Data retrieval function
            metric: Evaluation function
            expected: Expected value (optional)
            **options: Other evaluation options
        """
        self.getter = getter
        self.metric = metric
        self.expected = expected
        self.options = options

    def evaluate(self, env: Any, metadata: Dict[str, Any]) -> float:
        """Evaluate task completion

        Args:
            env: Environment instance
            metadata: Environment metadata

        Returns:
            Score (0.0 - 1.0)
        """
        try:
            # Get actual result
            result = self.getter(env, metadata)

            # Execute evaluation
            if self.expected is not None:
                score = self.metric(result, self.expected, **self.options)
            else:
                score = self.metric(result, **self.options)

            logger.info(f"Evaluation result: {score:.2f}")
            return float(score)

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0


class MultiConditionEvaluator:
    """Evaluator that supports multiple success conditions with AND/OR logic"""

    def __init__(
        self,
        conditions: List[Dict[str, Any]],
        logic: str = "AND",
        target_objects: List[str] = None,
    ):
        """Initialize multi-condition evaluator

        Args:
            conditions: List of success condition configurations
            logic: Logic operator ("AND" or "OR")
            target_objects: List of target object types
        """
        self.conditions = conditions
        self.logic = logic.upper()
        self.target_objects = target_objects or []

    def evaluate(self, env: Any, metadata: Dict[str, Any]) -> float:
        """Evaluate all conditions with specified logic

        Args:
            env: Environment instance
            metadata: Environment metadata

        Returns:
            Score (0.0 - 1.0)
        """

        if not self.conditions:
            logger.warning("No conditions to evaluate")
            return 0.0

        scores = []

        for i, condition in enumerate(self.conditions):
            try:
                score = self._evaluate_single_condition(condition, env, metadata)
                scores.append(score)
                condition_desc = condition.get('description', f"{condition.get('type')} - {condition}")
                print(f"  ✓ Condition {i + 1}: {condition_desc}")
                print(f"    Score: {score:.2f} {'✅' if score >= 1.0 else '❌'}")
                logger.info(
                    f"Condition {i + 1} ({condition.get('type', 'unknown')}): {score:.2f}"
                )
            except Exception as e:
                print(f"  ❌ Condition {i + 1} evaluation failed: {e}")
                logger.error(f"Condition {i + 1} evaluation failed: {e}")
                scores.append(0.0)

        # Apply logic
        if self.logic == "AND":
            final_score = 1.0 if all(s >= 1.0 for s in scores) else 0.0
        elif self.logic == "OR":
            final_score = 1.0 if any(s >= 1.0 for s in scores) else 0.0
        else:
            logger.warning(f"Unknown logic '{self.logic}', defaulting to AND")
            final_score = 1.0 if all(s >= 1.0 for s in scores) else 0.0

        print(f"\n📊 Multi-condition evaluation ({self.logic}):")
        print(f"   Individual scores: {scores}")
        print(f"   Final score: {final_score:.2f} {'✅ success' if final_score >= 1.0 else '❌ failed'}")
        logger.info(
            f"Multi-condition evaluation ({self.logic}): {final_score:.2f} (individual scores: {scores})"
        )
        return final_score

    def _evaluate_single_condition(
        self, condition: Dict[str, Any], env: Any, metadata: Dict[str, Any]
    ) -> float:
        """Evaluate a single condition

        Args:
            condition: Single condition configuration
            env: Environment instance
            metadata: Environment metadata

        Returns:
            Score (0.0 or 1.0)
        """

        condition_type = condition.get("type", "object_state")

        # Get all objects from metadata
        all_objects = metadata.get("objects", [])

        if condition_type == "object_state":
            # Check object state (isOpen, isToggled, isSliced, etc.)
            # Supports semantic mapping: checking for "Apple" will also match "AppleSliced"
            object_type = condition.get("object_type")
            state_field = condition.get("state") or condition.get("field", "isOpen")
            expected_value = condition.get("value", True)
            expected_liquid = condition.get("liquid")  #   ：       

            #        object_type，        
            if not object_type:
                raise ValueError(
                    f"object_state condition requires 'object_type' field. "
                    f"Got condition: {condition}"
                )

            #     ：isPickedUp        inventoryObjects
            if state_field == "isPickedUp":
                inventory_objects = metadata.get("inventoryObjects", [])
                
                #       
                print(f"\n  🔍 isPickedUp check:")
                print(f"    Target object: {object_type}")
                print(f"    Expected value: {expected_value}")
                print(f"    inventoryObjects count: {len(inventory_objects)}")
                if inventory_objects:
                    print(f"    Inventory objects: {[obj.get('objectType') for obj in inventory_objects]}")
                else:
                    print("    Inventory objects: empty")
                
                # Import semantic mapping
                from .semantic_mapping import get_semantic_variants
                object_type_variants = get_semantic_variants(object_type)
                print(f"    Semantic variants: {object_type_variants}")
                
                #    inventoryObjects         
                for inv_obj in inventory_objects:
                    inv_obj_type = inv_obj.get("objectType")
                    print(f"    Checking inventory object: {inv_obj_type}")
                    if inv_obj_type in object_type_variants:
                        #      ，isPickedUp     True
                        if expected_value:
                            print(f"    ✅ Found matching object in inventory: {inv_obj_type} (isPickedUp=True)")
                            logger.info(f"Found {object_type} in inventory (isPickedUp=True)")
                            return 1.0
                        else:
                            #    isPickedUp=False，      
                            print("    ❌ Object is in inventory, expected isPickedUp=False")
                            return 0.0
                
                #       
                if expected_value:
                    #    isPickedUp=True，       
                    print(f"    ❌ Expected {object_type} in inventory, but it was not found")
                    logger.debug(f"{object_type} not in inventory (isPickedUp=False)")
                    return 0.0
                else:
                    #    isPickedUp=False，      ，    
                    print("    ✅ Object is not in inventory as expected (isPickedUp=False)")
                    logger.info(f"{object_type} not in inventory (isPickedUp=False)")
                    return 1.0

            # Import semantic mapping
            from .semantic_mapping import get_semantic_variants

            # Get all possible object type variants
            object_type_variants = get_semantic_variants(object_type)

            # Filter target objects by any variant type
            target_objects = [
                obj
                for obj in all_objects
                if obj.get("objectType") in object_type_variants
            ]

            # Check if any object matches the expected state
            for obj in target_objects:
                #       
                if obj.get(state_field) != expected_value:
                    continue

                #                  ，      
                if (
                    expected_liquid
                    and state_field == "isFilledWithLiquid"
                    and expected_value
                ):
                    obj_liquid = obj.get("fillLiquid", "")
                    if obj_liquid.lower() == expected_liquid.lower():
                        logger.info(
                            f"Found {object_type} filled with {expected_liquid}"
                        )
                        return 1.0
                    else:
                        logger.debug(
                            f"{object_type} filled with '{obj_liquid}', expected '{expected_liquid}'"
                        )
                        continue
                else:
                    #     ，        
                    return 1.0

            return 0.0

        elif condition_type == "object_in_receptacle":
            # Check if object is in specified receptacle
            # Supports semantic mapping: checking for "Apple" will also match "AppleSliced"
            object_type = condition.get("object_type")
            receptacle_type = condition.get("receptacle_type")
            required_count = condition.get("count", 1)
            expected_value = condition.get(
                "value", True
            )  # Default True for backward compatibility

            if not object_type or not receptacle_type:
                logger.warning(
                    "object_in_receptacle requires both object_type and receptacle_type"
                )
                return 0.0

            # Import semantic mapping
            from .semantic_mapping import get_semantic_variants

            # Get all possible object type variants (e.g., Apple -> [Apple, AppleSliced])
            object_type_variants = get_semantic_variants(object_type)

            # Find target objects matching any variant
            target_objects = [
                obj
                for obj in all_objects
                if obj.get("objectType") in object_type_variants
            ]

            # Count objects in specified receptacle
            count_in_receptacle = 0
            for obj in target_objects:
                parent_receptacles = obj.get("parentReceptacles", [])
                for parent_id in parent_receptacles:
                    # Extract type from object ID (e.g., "Pan|1|2|3" -> "Pan")
                    parent_type = (
                        parent_id.split("|")[0] if "|" in parent_id else parent_id
                    )
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

        elif condition_type == "object_in_hand":
            # Check if holding specified object
            # Supports semantic mapping: checking for "Apple" will also match "AppleSliced"
            object_type = condition.get("object_type")
            inventory_objects = metadata.get("inventoryObjects", [])

            if not object_type:
                return 1.0 if inventory_objects else 0.0

            # Import semantic mapping
            from .semantic_mapping import get_semantic_variants

            # Get all possible object type variants
            object_type_variants = get_semantic_variants(object_type)

            for obj in inventory_objects:
                if obj.get("objectType") in object_type_variants:
                    return 1.0
            return 0.0

        elif condition_type == "agent_in_room":
            # Check if agent is in specified room
            room_type = condition.get("room_type")
            
            if not room_type:
                logger.warning("agent_in_room condition requires 'room_type' field")
                return 0.0
            
            from .getters import get_agent_room
            
            #              （    ）
            agent = metadata.get("agent", {})
            agent_pos = agent.get("position", {})
            all_objects = metadata.get("objects", [])
            nearby_info = []
            room_vote_count = {}  #            
            if agent_pos:
                for obj in all_objects:
                    obj_pos = obj.get("position", {})
                    if not obj_pos:
                        continue
                    #             
                    if obj.get("pickupable", False):
                        continue
                    dx = obj_pos.get("x", 0) - agent_pos.get("x", 0)
                    dz = obj_pos.get("z", 0) - agent_pos.get("z", 0)
                    distance = (dx ** 2 + dz ** 2) ** 0.5
                    if distance <= 3.0:
                        obj_room = obj.get("roomType")
                        if obj_room:
                            nearby_info.append((obj.get("objectType", "Unknown"), obj_room, distance))
                            #         
                            room_vote_count[obj_room] = room_vote_count.get(obj_room, 0) + 1
            
            current_room = get_agent_room(env, metadata)
            
            print(f"\n  🔍 Agent room check:")
            print(f"    Expected room: {room_type}")
            if nearby_info:
                print("    Nearby objects with room assignments:")
                for obj_type, obj_room, dist in nearby_info[:5]:  #     5 
                    print(f"      - {obj_type}: {obj_room} (distance: {dist:.2f}m)")
                #           
                if len(room_vote_count) > 1:
                    print("    Room estimates:")
                    for rtype, count in sorted(room_vote_count.items(), key=lambda x: -x[1]):
                        marker = "✅" if rtype == current_room else "  "
                        print(f"      {marker} {rtype}: {count} votes")
                    if len(room_vote_count) > 1 and max(room_vote_count.values()) - min(room_vote_count.values()) <= 1:
                        print("    ⚠️  Room estimate is ambiguous; using nearest object votes")
            else:
                print("    Nearby objects: none within 3m with roomType metadata")
            print(f"    Current room: {current_room if current_room else 'unknown'}")
            
            if not current_room:
                print("    ❌ Could not determine current room")
                logger.debug(f"Cannot determine agent's current room")
                return 0.0
            
            #           
            if current_room.lower() == room_type.lower():
                print(f"    ✅ Agent is in expected room: {room_type}")
                logger.info(f"Agent is in {room_type} (current: {current_room})")
                return 1.0
            else:
                print(f"    ❌ Agent is in {current_room}, expected {room_type}")
                logger.debug(f"Agent is in {current_room}, expected {room_type}")
                return 0.0

        elif condition_type == "object_in_room":
            # Check if object is in specified room (using floorPolygon for accurate detection)
            object_type = condition.get("object_type")
            room_type = condition.get("room_type")
            
            if not object_type or not room_type:
                logger.warning("object_in_room condition requires both 'object_type' and 'room_type' fields")
                return 0.0
            
            from .semantic_mapping import get_semantic_variants
            from .getters import _build_room_boundaries_from_house_scene, _point_in_polygon
            
            # Get all possible object type variants
            object_type_variants = get_semantic_variants(object_type)
            
            #      floorPolygon   （   ）
            room_polygons = _build_room_boundaries_from_house_scene(env)
            
            #          
            all_objects = metadata.get("objects", [])
            for obj in all_objects:
                obj_type = obj.get("objectType")
                if obj_type in object_type_variants:
                    obj_pos = obj.get("position", {})
                    if not obj_pos:
                        continue
                    
                    obj_x = obj_pos.get("x", 0)
                    obj_z = obj_pos.get("z", 0)
                    
                    #      floorPolygon   
                    if room_polygons:
                        #               
                        for room_name, polygons in room_polygons.items():
                            for polygon in polygons:
                                if _point_in_polygon(obj_x, obj_z, polygon):
                                    print(
                                        f"    Object room check: {obj_type}, "
                                        f"position=({obj_x:.2f}, {obj_z:.2f}), "
                                        f"room={room_name}, expected={room_type}"
                                    )
                                    if room_name.lower() == room_type.lower():
                                        print(
                                            f"    ✅ Found {obj_type} in {room_type} "
                                            "(matched by floorPolygon)"
                                        )
                                        logger.info(f"Found {object_type} in {room_type} (object: {obj_type}, position: ({obj_x}, {obj_z}), room: {room_name})")
                                        return 1.0
                                    break  #         ，            
                    else:
                        #          roomType   （    ）
                        obj_room = obj.get("roomType")
                        if obj_room:
                            print(
                                f"    Object room check: {obj_type}, "
                                f"room={obj_room}, expected={room_type}"
                            )
                            if obj_room.lower() == room_type.lower():
                                print(
                                    f"    ✅ Found {obj_type} in {room_type} "
                                    "(matched by roomType metadata)"
                                )
                                logger.info(f"Found {object_type} in {room_type} (object: {obj_type}, room: {obj_room})")
                                return 1.0
            
            print(f"    ❌ {object_type} was not found in {room_type}")
            logger.debug(f"{object_type} not found in {room_type}")
            return 0.0

        elif condition_type == "object_not_in_room":
            #     ：                （              ）
            object_type = condition.get("object_type")
            room_type = condition.get("room_type")

            if not object_type or not room_type:
                logger.warning("object_not_in_room condition requires both 'object_type' and 'room_type' fields")
                return 0.0

            from .semantic_mapping import get_semantic_variants
            from .getters import _build_room_boundaries_from_house_scene, _point_in_polygon

            object_type_variants = get_semantic_variants(object_type)
            room_polygons = _build_room_boundaries_from_house_scene(env)
            all_objects = metadata.get("objects", [])

            for obj in all_objects:
                obj_type = obj.get("objectType")
                if obj_type not in object_type_variants:
                    continue
                obj_pos = obj.get("position", {})
                if not obj_pos:
                    continue
                obj_x = obj_pos.get("x", 0)
                obj_z = obj_pos.get("z", 0)
                if room_polygons:
                    for room_name, polygons in room_polygons.items():
                        for polygon in polygons:
                            if _point_in_polygon(obj_x, obj_z, polygon):
                                if room_name.lower() == room_type.lower():
                                    print(f"    ❌ Found {obj_type} in {room_type}; condition failed")
                                    logger.debug(f"{object_type} found in {room_type}, object_not_in_room fails")
                                    return 0.0
                                break
                else:
                    obj_room = obj.get("roomType")
                    if obj_room and obj_room.lower() == room_type.lower():
                        print(f"    ❌ Found {obj_type} in {room_type}; condition failed")
                        logger.debug(f"{object_type} found in {room_type}, object_not_in_room fails")
                        return 0.0

            print(f"    ✅    {object_type}     {room_type}")
            logger.info(f"All {object_type} are not in {room_type}")
            return 1.0

        elif condition_type == "agent_near_object":
            # Check if agent is near specified object
            object_type = condition.get("object_type")
            max_distance = condition.get("distance", 1.0)  # Default 1 meter
            
            if not object_type:
                logger.warning("agent_near_object condition requires 'object_type' field")
                return 0.0
            
            from .getters import get_agent_position
            from .semantic_mapping import get_semantic_variants
            
            #        
            agent_pos = get_agent_position(env, metadata)
            if not agent_pos:
                print(f"    ❌          ")
                logger.debug("Cannot get agent position")
                return 0.0
            
            agent_x = agent_pos.get("x", 0)
            agent_z = agent_pos.get("z", 0)
            
            #              
            object_type_variants = get_semantic_variants(object_type)
            
            #            
            all_objects = metadata.get("objects", [])
            closest_distance = float('inf')
            closest_obj = None
            
            for obj in all_objects:
                obj_type = obj.get("objectType")
                if obj_type in object_type_variants:
                    obj_pos = obj.get("position", {})
                    if not obj_pos:
                        continue
                    
                    obj_x = obj_pos.get("x", 0)
                    obj_z = obj_pos.get("z", 0)
                    
                    #   2D  （  Y ）
                    dx = agent_x - obj_x
                    dz = agent_z - obj_z
                    distance = (dx ** 2 + dz ** 2) ** 0.5
                    
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_obj = obj_type
            
            if closest_obj:
                print(f"    Near-object check for {object_type}:")
                print(f"      Agent position: ({agent_x:.2f}, {agent_z:.2f})")
                print(
                    f"      Closest object: {closest_obj}, "
                    f"distance: {closest_distance:.2f}m, "
                    f"threshold: {max_distance:.2f}m"
                )
                
                if closest_distance <= max_distance:
                    print(
                        f"    ✅ Agent is near {object_type} "
                        f"(distance: {closest_distance:.2f}m <= {max_distance:.2f}m)"
                    )
                    logger.info(f"Agent is near {object_type} (distance: {closest_distance:.2f}m, threshold: {max_distance:.2f}m)")
                    return 1.0
                else:
                    print(
                        f"    ❌ Agent is too far from {object_type} "
                        f"(distance: {closest_distance:.2f}m > {max_distance:.2f}m)"
                    )
                    logger.debug(f"Agent is too far from {object_type} (distance: {closest_distance:.2f}m, threshold: {max_distance:.2f}m)")
                    return 0.0
            else:
                print(f"    ❌ Object not found: {object_type}")
                logger.debug(f"Object {object_type} not found")
                return 0.0

        else:
            logger.warning(f"Unknown condition type: {condition_type}")
            return 0.0


def create_evaluator_from_config(config: Dict[str, Any]) -> Evaluator:
    """Create evaluator from configuration

    Supports both:
    1. Single success_condition (legacy format)
    2. Multiple success_conditions with success_logic (new format)

    Args:
        config: Task configuration, contains success_condition or success_conditions

    Returns:
        Evaluator or MultiConditionEvaluator instance
    """
    from . import getters, metrics

    # Check for new format: success_conditions array
    success_conditions = config.get("success_conditions")
    if success_conditions and isinstance(success_conditions, list):
        success_logic = config.get("success_logic", "AND")
        target_objects = config.get("target_object_types", [])

        logger.info(
            f"Creating MultiConditionEvaluator with {len(success_conditions)} conditions, logic={success_logic}"
        )
        return MultiConditionEvaluator(
            conditions=success_conditions,
            logic=success_logic,
            target_objects=target_objects,
        )

    # Legacy format: single success_condition
    success_condition = config.get("success_condition", {})
    condition_type = success_condition.get("type", "object_state")

    # Select getter and metric based on type
    if condition_type == "object_state":
        getter = getters.get_object_state
        metric = metrics.check_object_state
        expected = {
            "field": success_condition.get("field", "isOpen"),
            "value": success_condition.get("value", True),
            "any": success_condition.get("any", True),
        }

    elif condition_type == "object_in_hand":
        getter = getters.get_inventory_objects
        metric = metrics.check_object_in_hand
        expected = {"object_type": success_condition.get("object_type")}

    elif condition_type == "object_in_receptacle":
        getter = getters.get_object_state
        metric = metrics.check_object_in_receptacle
        expected = {
            "receptacle_type": success_condition.get("receptacle_type"),
            "object_type": success_condition.get("object_type"),
            "value": success_condition.get(
                "value", True
            ),  # Default True for backward compatibility
        }

    else:
        raise ValueError(f"Unsupported evaluation type: {condition_type}")

    # Get target object types
    target_objects = config.get("target_object_types", [])

    return Evaluator(
        getter=getter, metric=metric, expected=expected, target_objects=target_objects
    )
