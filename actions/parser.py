"""Unified action parsing utilities.

Model-facing actions follow the SpatialWorld Table 9 interface:
- Move(direction, [granularity])
- Rotate(direction, [angle])
- Tilt(direction, [angle])
- ChangePosture(pose)
- Pick(obj), Place(obj, [target])
- ChangeState(obj, state)
- Manipulate(obj, action)
- EndTask(status), Communicate(msg)

The parser translates these names into existing backend action dictionaries so
legacy environment wrappers do not need to change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_TERMINAL_ACTIONS = {"DONE", "FAIL"}
_GRANULARITIES = {"small": "Small", "medium": "Medium", "large": "Large", "0": "0"}


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _split_args(params: str) -> List[str]:
    args: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    depth = 0
    for ch in params:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
            current.append(ch)
            continue
        if ch in "([{" :
            depth += 1
            current.append(ch)
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            current.append(ch)
            continue
        if ch == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                args.append(_strip_quotes(item))
            current = []
            continue
        current.append(ch)
    item = "".join(current).strip()
    if item:
        args.append(_strip_quotes(item))
    return args


def _parse_call(action_string: str) -> tuple[str, List[str]]:
    action_string = action_string.strip()
    if "(" not in action_string or not action_string.endswith(")"):
        return action_string, []
    name = action_string[: action_string.index("(")].strip()
    params = action_string[action_string.index("(") + 1 : -1].strip()
    return name, _split_args(params)


def _normalize_direction(direction: str) -> str:
    aliases = {
        "front": "forward", "ahead": "forward", "forward": "forward",
        "back": "backward", "backward": "backward",
        "left": "left", "right": "right", "up": "up", "down": "down",
    }
    key = str(direction or "").strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported direction: {direction}")
    return aliases[key]


def _normalize_granularity(value: Optional[str]) -> Optional[str]:
    if value is None or str(value).strip() == "":
        return None
    raw = str(value).strip()
    low = raw.lower()
    return _GRANULARITIES.get(low, raw)


def _task_completion(status: str) -> Dict[str, Any]:
    status = str(status or "").strip().upper()
    if status not in _TERMINAL_ACTIONS:
        raise ValueError(f"EndTask status must be DONE or FAIL, got {status!r}")
    return {"action_type": "task_completion", "action_name": status}


def _distance_params(granularity: Optional[str]) -> Dict[str, Any]:
    if not granularity:
        return {}
    value = str(granularity).lower() if str(granularity).lower() in {"small", "medium", "large"} else granularity
    return {"distance": value}


def _degree_params(angle: Optional[str]) -> Dict[str, Any]:
    if not angle:
        return {}
    value = str(angle).lower() if str(angle).lower() in {"small", "medium", "large"} else angle
    return {"degrees": value}


def _vh_obj(obj: str) -> str:
    return str(obj or "").strip().lower().replace(" ", "")


def _map_move(direction: str, granularity: Optional[str], env_type: str, executor_type: Optional[str]) -> Dict[str, Any]:
    if granularity == "0":
        if env_type == "carla":
            return {"action_type": "carla_semantic", "action_name": "Stop", "parameters": {}}
        return {"action_type": "navigation", "action_name": "Wait", "granularity": "0"}

    if env_type == "virtualhome":
        if direction != "forward":
            raise ValueError("VirtualHome supports Move(forward, ...) only; rotate first for other directions")
        result = {"action_type": "navigation", "action_name": "WalkForward"}
        if granularity:
            result["granularity"] = str(granularity).lower() if str(granularity).lower() in {"small", "medium", "large"} else granularity
        return result

    if env_type == "carla":
        is_walker = executor_type == "walker"
        if is_walker:
            if direction == "forward":
                return {"action_type": "carla_walker", "action_name": "WalkForward", "parameters": _distance_params(granularity)}
            if direction == "backward":
                return {"action_type": "carla_walker", "action_name": "WalkBackward", "parameters": _distance_params(granularity)}
            raise ValueError("CARLA walker supports Move(forward/backward, ...) plus Rotate(left/right)")
        if direction == "forward":
            return {"action_type": "carla_semantic", "action_name": "FollowLane", "parameters": _distance_params(granularity)}
        if direction == "left":
            return {"action_type": "carla_semantic", "action_name": "ChangeLaneLeft", "parameters": {}}
        if direction == "right":
            return {"action_type": "carla_semantic", "action_name": "ChangeLaneRight", "parameters": {}}
        if direction == "backward":
            return {"action_type": "carla_semantic", "action_name": "Stop", "parameters": {}}
        raise ValueError(f"Unsupported CARLA move direction: {direction}")

    mapping = {"forward": "MoveAhead", "backward": "MoveBack", "left": "MoveLeft", "right": "MoveRight"}
    if direction not in mapping:
        raise ValueError(f"{env_type} does not support Move({direction})")
    result: Dict[str, Any] = {"action_type": "navigation", "action_name": mapping[direction]}
    if granularity:
        if str(granularity).lower() in {"small", "medium", "large"}:
            result["granularity"] = granularity
        else:
            result["magnitude"] = float(granularity)
    return result


def _map_rotate(direction: str, angle: Optional[str], env_type: str, executor_type: Optional[str]) -> Dict[str, Any]:
    if direction not in {"left", "right"}:
        raise ValueError("Rotate direction must be left or right")
    if env_type == "virtualhome":
        result: Dict[str, Any] = {"action_type": "navigation", "action_name": "TurnLeft" if direction == "left" else "TurnRight"}
        if angle:
            low = str(angle).lower()
            result["turn_modifier"] = "small" if low in {"small", "30", "30.0"} else "normal"
            result["turn_degrees"] = 30.0 if result["turn_modifier"] == "small" else 90.0
        return result
    if env_type == "carla":
        if executor_type == "walker":
            return {"action_type": "carla_walker", "action_name": "TurnLeft" if direction == "left" else "TurnRight", "parameters": _degree_params(angle)}
        return {"action_type": "carla_semantic", "action_name": "TeleportTurnLeft" if direction == "left" else "TeleportTurnRight", "parameters": {}}
    result: Dict[str, Any] = {"action_type": "navigation", "action_name": "RotateLeft" if direction == "left" else "RotateRight"}
    if angle and str(angle).lower() not in {"small", "medium", "large"}:
        result["degrees"] = float(angle)
    return result


def _map_tilt(direction: str, angle: Optional[str], env_type: str) -> Dict[str, Any]:
    if direction not in {"up", "down"}:
        raise ValueError("Tilt direction must be up or down")
    result: Dict[str, Any] = {"action_type": "navigation", "action_name": "LookUp" if direction == "up" else "LookDown"}
    if angle and str(angle).lower() not in {"small", "medium", "large"}:
        result["degrees"] = float(angle)
    return result


def _map_posture(pose: str, env_type: str) -> Dict[str, Any]:
    key = str(pose).strip().lower()
    if env_type == "virtualhome":
        if key in {"stand", "stand_up", "standup"}:
            return {"action_type": "navigation", "action_name": "StandUp"}
        raise ValueError("VirtualHome supports ChangePosture(stand_up) only")
    if key == "crouch":
        return {"action_type": "navigation", "action_name": "Crouch"}
    if key in {"stand", "stand_up", "standup"}:
        return {"action_type": "navigation", "action_name": "Stand"}
    raise ValueError(f"Unsupported pose: {pose}")


def _map_pick(obj: str, env_type: str) -> Dict[str, Any]:
    if env_type == "virtualhome":
        return {"action_type": "interaction", "action_name": "Grab", "object_type": _vh_obj(obj)}
    return {"action_type": "interaction", "action_name": "PickupObject", "object_type": obj}


def _map_place(obj: str, target: Optional[str], relation: Optional[str], env_type: str) -> Dict[str, Any]:
    if env_type == "virtualhome":
        if not target:
            raise ValueError("VirtualHome Place requires a target")
        action_name = "PutIn" if str(relation or "").strip().lower() in {"in", "inside", "container"} else "PutBack"
        return {"action_type": "interaction", "action_name": action_name, "object_type": _vh_obj(obj), "object2_type": _vh_obj(target)}
    if not target:
        return {"action_type": "interaction", "action_name": "DropHandObject"}
    return {"action_type": "interaction", "action_name": "PutObject", "object_type": target}


def _map_change_state(obj: str, state: str, env_type: str, value: Optional[str] = None) -> Dict[str, Any]:
    key = str(state).strip().lower()
    if env_type == "virtualhome":
        mapping = {"open": "Open", "close": "Close", "closed": "Close", "on": "SwitchOn", "off": "SwitchOff"}
        if key not in mapping:
            raise ValueError(f"VirtualHome does not support ChangeState(..., {state})")
        return {"action_type": "interaction", "action_name": mapping[key], "object_type": _vh_obj(obj)}
    mapping = {
        "open": "OpenObject", "close": "CloseObject", "closed": "CloseObject",
        "on": "ToggleObjectOn", "off": "ToggleObjectOff", "clean": "CleanObject", "dirty": "DirtyObject",
        "sliced": "SliceObject", "slice": "SliceObject", "broken": "BreakObject", "break": "BreakObject",
        "cooked": "CookObject", "cook": "CookObject", "empty": "EmptyLiquidFromObject",
        "used_up": "UseUpObject", "usedup": "UseUpObject",
    }
    if key in {"filled", "fill"}:
        result = {"action_type": "interaction", "action_name": "FillObjectWithLiquid", "object_type": obj}
        if value:
            result["fillLiquid"] = value
        return result
    if key not in mapping:
        raise ValueError(f"Unsupported state: {state}")
    return {"action_type": "interaction", "action_name": mapping[key], "object_type": obj}


def _map_manipulate(obj: str, action: str, env_type: str) -> Dict[str, Any]:
    key = str(action).strip().lower()
    if env_type == "virtualhome":
        mapping = {"touch": "Touch", "look_at": "LookAt", "lookat": "LookAt", "drink": "Drink", "sit": "Sit"}
        if key not in mapping:
            raise ValueError(f"VirtualHome does not support Manipulate(..., {action})")
        return {"action_type": "interaction", "action_name": mapping[key], "object_type": _vh_obj(obj)}
    mapping = {"push": "PushObject", "pull": "PullObject", "throw": "ThrowObject", "directional_push": "DirectionalPush", "directionalpush": "DirectionalPush"}
    if key not in mapping:
        raise ValueError(f"Unsupported manipulation action: {action}")
    result = {"action_type": "interaction", "action_name": mapping[key]}
    if mapping[key] != "ThrowObject":
        result["object_type"] = obj
    return result


def _parse_unified_action(action_string: str, env_type: str = "ai2thor", executor_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
    name, args = _parse_call(action_string)
    low_name = name.strip().lower()
    env_type = (env_type or "ai2thor").lower()
    executor_type = (executor_type or "").lower() or None

    if low_name == "endtask":
        if not args:
            raise ValueError("EndTask requires status DONE or FAIL")
        return _task_completion(args[0])
    if low_name == "communicate":
        if not args:
            raise ValueError("Communicate requires a message")
        return {"action_type": "communication", "action_name": "Communicate", "message": args[0]}
    if low_name == "move":
        if not args:
            raise ValueError("Move requires direction")
        return _map_move(_normalize_direction(args[0]), _normalize_granularity(args[1] if len(args) > 1 else None), env_type, executor_type)
    if low_name == "rotate":
        if not args:
            raise ValueError("Rotate requires direction")
        return _map_rotate(_normalize_direction(args[0]), _normalize_granularity(args[1] if len(args) > 1 else None), env_type, executor_type)
    if low_name == "tilt":
        if not args:
            raise ValueError("Tilt requires direction")
        return _map_tilt(_normalize_direction(args[0]), _normalize_granularity(args[1] if len(args) > 1 else None), env_type)
    if low_name == "changeposture":
        if not args:
            raise ValueError("ChangePosture requires pose")
        return _map_posture(args[0], env_type)
    if low_name == "pick":
        if not args:
            raise ValueError("Pick requires object")
        return _map_pick(args[0], env_type)
    if low_name == "place":
        if not args:
            raise ValueError("Place requires object")
        return _map_place(args[0], args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else None, env_type)
    if low_name == "changestate":
        if len(args) < 2:
            raise ValueError("ChangeState requires object and state")
        return _map_change_state(args[0], args[1], env_type, args[2] if len(args) > 2 else None)
    if low_name == "manipulate":
        if len(args) < 2:
            raise ValueError("Manipulate requires object and action")
        return _map_manipulate(args[0], args[1], env_type)
    return None


def _parse_legacy_action_string(action_string: str, env_type: str = "ai2thor", executor_type: Optional[str] = None) -> Dict[str, Any]:
    if action_string.upper() == "DONE":
        return {"action_type": "task_completion", "action_name": "DONE"}
    if action_string.upper() == "FAIL":
        return {"action_type": "task_completion", "action_name": "FAIL"}

    ai2thor_navigation_actions = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Crouch", "Stand"}
    carla_semantic_actions = {"TeleportForward", "TeleportTurnLeft", "TeleportTurnRight", "ChangeLaneLeft", "ChangeLaneRight", "Stop", "FollowLane", "GoStraight", "Nudge", "SpeedUp", "SlowDown"}
    carla_walker_actions = {"WalkForward", "WalkBackward", "TurnLeft", "TurnRight", "RotateLeft", "RotateRight"}
    interaction_actions = {"PickupObject", "DropHandObject", "PutObject", "ThrowObject", "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff", "SliceObject", "BreakObject", "CookObject", "DirtyObject", "CleanObject", "FillObjectWithLiquid", "EmptyLiquidFromObject", "UseUpObject", "PushObject", "PullObject", "DirectionalPush"}

    if action_string in ai2thor_navigation_actions:
        return {"action_type": "navigation", "action_name": action_string}
    if action_string in carla_walker_actions:
        return {"action_type": "carla_walker", "action_name": action_string, "parameters": {}}
    if action_string in carla_semantic_actions:
        return {"action_type": "carla_semantic", "action_name": action_string, "parameters": {}}
    if action_string == "DropHandObject":
        return {"action_type": "interaction", "action_name": "DropHandObject"}
    if "(" not in action_string or not action_string.endswith(")"):
        raise ValueError(f"Unrecognized action format: {action_string}")

    action_name = action_string[: action_string.index("(")]
    params_str = action_string[action_string.index("(") + 1 : -1].strip()
    if action_name in ai2thor_navigation_actions:
        result: Dict[str, Any] = {"action_type": "navigation", "action_name": action_name}
        if params_str:
            clean = params_str.strip()
            if clean.lower() in {"small", "medium", "large"}:
                result["granularity"] = clean
            else:
                try:
                    value = float(clean) if "." in clean else int(clean)
                    if action_name in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}:
                        result["magnitude"] = value
                    elif action_name in {"RotateLeft", "RotateRight", "LookUp", "LookDown"}:
                        result["degrees"] = value
                except (TypeError, ValueError):
                    result["param"] = clean
        return result

    if action_name in carla_semantic_actions or action_name in carla_walker_actions:
        parameters: Dict[str, Any] = {}
        if params_str:
            for param in params_str.split(","):
                param = param.strip()
                if "=" in param:
                    key, value = param.split("=", 1)
                    key, value = key.strip(), value.strip()
                    try:
                        parameters[key] = float(value) if "." in value else int(value)
                    except ValueError:
                        parameters[key] = value
                else:
                    parameters["value"] = param
        return {"action_type": "carla_semantic" if action_name in carla_semantic_actions else "carla_walker", "action_name": action_name, "parameters": parameters}

    if action_name in interaction_actions:
        result = {"action_type": "interaction", "action_name": action_name}
        if action_name != "DropHandObject":
            if not params_str:
                raise ValueError(f"{action_name} requires object type specification")
            if action_name == "FillObjectWithLiquid" and "," in params_str:
                parts = [p.strip() for p in params_str.split(",", 1)]
                result["object_type"] = parts[0]
                result["fillLiquid"] = parts[1]
            else:
                result["object_type"] = params_str
        return result
    raise ValueError(f"Unknown action: {action_name}")


def parse_action_string(action_string: str, env_type: str = "ai2thor", executor_type: Optional[str] = None) -> Dict[str, Any]:
    action_string = action_string.strip()
    unified = _parse_unified_action(action_string, env_type=env_type, executor_type=executor_type)
    if unified is not None:
        return unified
    return _parse_legacy_action_string(action_string, env_type=env_type, executor_type=executor_type)


def parse_virtualhome_action_string(action_string: str) -> Dict[str, Any]:
    return parse_action_string(action_string, env_type="virtualhome")


_parse_action_string = parse_action_string
