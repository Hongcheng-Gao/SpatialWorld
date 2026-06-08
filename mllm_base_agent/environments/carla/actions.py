from enum import Enum
from typing import List, Dict, Any, Optional
import re


class ExecutorType(Enum):
    VEHICLE = "vehicle"
    WALKER = "walker"


class ActionType(Enum):
    # Common
    DONE = "Done"
    FAIL = "Fail"
    PASS = "Pass"  # Fallback

    # Vehicle Actions
    TELEPORT_FORWARD = "TeleportForward"
    TELEPORT_TURN_LEFT = "TeleportTurnLeft"
    TELEPORT_TURN_RIGHT = "TeleportTurnRight"
    FOLLOW_LANE = "FollowLane"
    CHANGE_LANE_LEFT = "ChangeLaneLeft"
    CHANGE_LANE_RIGHT = "ChangeLaneRight"
    STOP = "Stop"
    GO_STRAIGHT = "GoStraight"  # Legacy
    NUDGE = "Nudge"  # Legacy
    SPEED_UP = "SpeedUp"  # Legacy
    SLOW_DOWN = "SlowDown"  # Legacy

    # Walker Actions
    WALK_FORWARD = "WalkForward"
    WALK_BACKWARD = "WalkBackward"
    TURN_LEFT = "TurnLeft"
    TURN_RIGHT = "TurnRight"


class CarlaUnifiedAction:
    def __init__(self, action_type: ActionType, parameters: Dict[str, Any] = None):
        self.action_type = action_type
        self.parameters = parameters or {}

    def __repr__(self):
        params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        if params_str:
            return f"{self.action_type.value}({params_str})"
        return self.action_type.value

    @staticmethod
    def from_string(
        action_str: str, executor_type: ExecutorType
    ) -> "CarlaUnifiedAction":
        """
        Parses an action string like "MoveForward(distance=10)" or "TurnLeft"
        """
        action_str = action_str.strip()

        # Regex to parse Name(key=value, key2=value2) or just Name
        pattern = r"([a-zA-Z0-9_]+)(?:\((.*)\))?"
        match = re.match(pattern, action_str)

        if not match:
            raise ValueError(f"Invalid action string format: {action_str}")

        action_name = match.group(1)
        params_str = match.group(2)

        # Find matching ActionType
        try:
            # Case insensitive search could be implemented, but strict for now or try title case
            # Common case: "MoveForward" -> ActionType.MOVE_FORWARD (if strictly mapped)
            # But here values are exact strings.

            # Helper to find enum by value string
            action_enum = None
            for act in ActionType:
                if act.value.lower() == action_name.lower():
                    action_enum = act
                    break

            if not action_enum:
                # Fallback for aliases / legacy
                if executor_type == ExecutorType.VEHICLE:
                    if (
                        action_name == "MoveForward"
                    ):  # Alias for TeleportForward or FollowLane?
                        action_enum = ActionType.TELEPORT_FORWARD
                    elif action_name == "MoveLeft":
                        action_enum = ActionType.CHANGE_LANE_LEFT
                    elif action_name == "MoveRight":
                        action_enum = ActionType.CHANGE_LANE_RIGHT
                elif executor_type == ExecutorType.WALKER:
                    if action_name == "MoveAhead":
                        action_enum = ActionType.WALK_FORWARD
                    elif action_name == "MoveBack":
                        action_enum = ActionType.WALK_BACKWARD
                    elif action_name == "RotateLeft":
                        action_enum = ActionType.TURN_LEFT
                    elif action_name == "RotateRight":
                        action_enum = ActionType.TURN_RIGHT

            if not action_enum:
                # If still not found, treat as unknown or raise
                # The usage in code assumes action_obj.action_type.value is valid
                raise ValueError(f"Unknown action: {action_name}")

        except Exception as e:
            raise ValueError(f"Error parsing action name '{action_name}': {e}")

        # Restrict available actions by executor type to align with interact/evaluate
        vehicle_allowed = {
            ActionType.DONE,
            ActionType.FAIL,
            ActionType.PASS,
            ActionType.TELEPORT_FORWARD,
            ActionType.TELEPORT_TURN_LEFT,
            ActionType.TELEPORT_TURN_RIGHT,
            ActionType.FOLLOW_LANE,
            ActionType.CHANGE_LANE_LEFT,
            ActionType.CHANGE_LANE_RIGHT,
            ActionType.STOP,
            ActionType.GO_STRAIGHT,
            ActionType.NUDGE,
            ActionType.SPEED_UP,
            ActionType.SLOW_DOWN,
            # Legacy aliases
            ActionType.TURN_LEFT,
            ActionType.TURN_RIGHT,
        }
        walker_allowed = {
            ActionType.DONE,
            ActionType.FAIL,
            ActionType.PASS,
            ActionType.WALK_FORWARD,
            ActionType.WALK_BACKWARD,
            ActionType.TURN_LEFT,
            ActionType.TURN_RIGHT,
        }

        allowed = vehicle_allowed if executor_type == ExecutorType.VEHICLE else walker_allowed
        if action_enum not in allowed:
            raise ValueError(
                f"Action '{action_enum.value}' is not allowed for executor '{executor_type.value}'"
            )

        parameters = {}
        if params_str:
            # splitting by comma, allowing for spaces
            # Simple split by comma might fail on nested things, but for now simple params
            param_parts = params_str.split(",")
            for part in param_parts:
                part = part.strip()
                if not part:
                    continue
                if "=" in part:
                    k, v = part.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    # Try to convert to float/int/bool
                    if v.lower() == "true":
                        v = True
                    elif v.lower() == "false":
                        v = False
                    else:
                        try:
                            if "." in v:
                                v = float(v)
                            else:
                                v = int(v)
                        except:
                            pass  # keep as string
                    parameters[k] = v
                else:
                    # Positional args - handle if needed, or just warn
                    # Attempt to infer key based on action
                    if action_enum in [
                        ActionType.TELEPORT_FORWARD,
                        ActionType.WALK_FORWARD,
                        ActionType.WALK_BACKWARD,
                        ActionType.FOLLOW_LANE,
                    ]:
                        # Assume first positional is 'distance'
                        try:
                            val = float(part)
                            parameters["distance"] = val
                        except:
                            pass
                    elif action_enum in [
                        ActionType.TELEPORT_TURN_LEFT,
                        ActionType.TELEPORT_TURN_RIGHT,
                        ActionType.TURN_LEFT,
                        ActionType.TURN_RIGHT,
                    ]:
                        try:
                            val = float(part)
                            parameters["degrees"] = val
                        except:
                            pass

        return CarlaUnifiedAction(action_enum, parameters)


def parse_action_sequence(
    action_strs: List[str], executor_type: ExecutorType
) -> List[CarlaUnifiedAction]:
    actions = []
    for s in action_strs:
        s = s.strip()
        if not s:
            continue
        try:
            action = CarlaUnifiedAction.from_string(s, executor_type)
            actions.append(action)
        except Exception as e:
            print(f"Warning: Failed to parse action '{s}': {e}")
    return actions
