"""
Evaluator Base Class
Supports both single success_condition and multiple success_conditions with AND/OR logic
"""

from typing import Dict, Any, Callable, List
import logging

logger = logging.getLogger(__name__)


import math

# ==============================================================================
# Helper Functions for Geometry
# ==============================================================================


def point_to_segment_dist(px, py, x1, y1, x2, y2):
    """Calculate distance from point to segment"""
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
    """Check if point is in polygon (Ray Casting)"""
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i][:2]  # Support 2D or 3D points
        xj, yj = polygon[j][:2]

        intersect = ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-9) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def distance_to_polygon(px, py, polygon):
    """Calculate distance to polygon (0 if inside)"""
    if is_point_in_polygon(px, py, polygon):
        return 0.0

    min_dist = float("inf")
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        d = point_to_segment_dist(px, py, p1[0], p1[1], p2[0], p2[1])
        if d < min_dist:
            min_dist = d
    return min_dist


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
                logger.info(
                    f"Condition {i + 1} ({condition.get('type', 'unknown')}): {score:.2f}"
                )
            except Exception as e:
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
        bound_instances = metadata.get("bound_instances", {}) or {}

        def _get_bound_id(obj_type: str):
            if not obj_type:
                return None
            #       ，      
            if obj_type in bound_instances:
                return str(bound_instances[obj_type])
            lower_key = obj_type.lower()
            if lower_key in bound_instances:
                return str(bound_instances[lower_key])
            return None

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

            #        ，         
            bound_id = _get_bound_id(object_type)
            if bound_id is not None:
                target_objects = [
                    obj
                    for obj in target_objects
                    if str(obj.get("objectId")) == bound_id
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

            #        ，         
            bound_id = _get_bound_id(object_type)
            if bound_id is not None:
                target_objects = [
                    obj
                    for obj in target_objects
                    if str(obj.get("objectId")) == bound_id
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

            bound_id = _get_bound_id(object_type)

            for obj in inventory_objects:
                if obj.get("objectType") in object_type_variants and (
                    bound_id is None or str(obj.get("objectId")) == bound_id
                ):
                    return 1.0
            return 0.0

        elif condition_type == "polygon_area":
            # Check if agent is in specified polygon area
            # Supports tolerance (e.g., for vehicle "in front of building")
            polygon = condition.get("polygon") or condition.get("target_polygon")

            # polygon format check: should be list of [x, y] or [x, y, z]
            if not polygon or not isinstance(polygon, list) or len(polygon) < 3:
                logger.warning(f"Invalid polygon in condition: {polygon}")
                return 0.0

            agent_pos = metadata.get("agent", {}).get("position")
            if not agent_pos:
                logger.warning("Agent position not found in metadata")
                return 0.0

            px, py = agent_pos.get("x"), agent_pos.get("y")
            if px is None or py is None:
                return 0.0

            # 1. Exact inclusion check
            if is_point_in_polygon(px, py, polygon):
                logger.info("Agent inside polygon")
                return 1.0

            # 2. Tolerance check (distance)
            # Default tolerance 8.0 meters (approx 2 lanes) if not specified
            # This covers the "vehicle on the road in front of building" case
            tolerance = condition.get("tolerance", 8.0)

            dist = distance_to_polygon(px, py, polygon)
            if dist <= tolerance:
                logger.info(
                    f"Agent outside polygon but within tolerance {tolerance}m (dist={dist:.2f}m)"
                )
                return 1.0

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

    # Handle case where success_conditions is a single dict (common mistake or variant)
    if success_conditions and isinstance(success_conditions, dict):
        success_conditions = [success_conditions]

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
