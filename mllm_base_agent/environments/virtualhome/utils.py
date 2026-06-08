"""
VirtualHome Environment Utilities
Provides helper functions for VirtualHome graph state parsing, action conversion,
and text state generation. Adapted to be compatible with this project's evaluation framework.
"""

import math
from typing import Optional, List, Dict, Any, Tuple


# ============================================================================
# VirtualHome Action Categories
# ============================================================================

# Navigation actions that require an object target
# Intentionally empty: object-target navigation actions are disabled.
VH_NAVIGATION_WITH_OBJECT = set()

# Navigation actions that do NOT require an object
# Keep this aligned with core/prompts/virtualhome.py.
VH_NAVIGATION_NO_OBJECT = {
    "TurnLeft",
    "TurnRight",
    "WalkForward",
    "LookUp",
    "LookDown",
    "StandUp",
}

# Interaction actions (single object)
# Keep this aligned with core/prompts/virtualhome.py.
VH_INTERACTION_ONE_OBJECT = {
    "Grab",
    "Open",
    "Close",
    "SwitchOn",
    "SwitchOff",
    "Drink",
    "Sit",
    "LookAt",
    "Touch",
}

# Interaction actions requiring TWO objects (object + target/container)
# Keep this aligned with core/prompts/virtualhome.py.
VH_INTERACTION_TWO_OBJECTS = {
    "PutBack",  # <object> onto <surface>
    "PutIn",  # <object> into <container>
}

# Mapping from project/AI2THOR-style action names to VirtualHome action names
ACTION_NAME_MAP: Dict[str, str] = {
    # Navigation
    "TurnLeft": "TurnLeft",
    "TurnRight": "TurnRight",
    "WalkForward": "WalkForward",
    # AI2THOR-style -> VH equivalents
    "MoveAhead": "WalkForward",
    "MoveBack": "WalkForward",  # legacy compatibility only; prompt does not allow it
    # Interaction
    "Grab": "Grab",
    "Open": "Open",
    "Close": "Close",
    "PutBack": "PutBack",
    "PutIn": "PutIn",
    "SwitchOn": "SwitchOn",
    "SwitchOff": "SwitchOff",
    "Drink": "Drink",
    "Sit": "Sit",
    "StandUp": "StandUp",
    "LookAt": "LookAt",
    "Touch": "Touch",
}

# Object-type aliases to improve robustness against common naming variants
# from different prompts/models (e.g. phone vs cellphone, tv_stand vs tvstand).
OBJECT_TYPE_ALIASES: Dict[str, str] = {
    "phone": "cellphone",
    "cell_phone": "cellphone",
    "mobile_phone": "cellphone",
    "smart_phone": "cellphone",
    "tv_stand": "tvstand",
    "tv-stand": "tvstand",
    "light_switch": "lightswitch",
    "table_lamp": "tablelamp",
    "desk_lamp": "tablelamp",
    "coffee_table": "coffeetable",
    "kitchen_table": "kitchentable",
    "remote_control": "remotecontrol",
    "wine_glass": "wineglass",
    "dish_bowl": "dishbowl",
    "bread_slice": "breadslice",
    "dish_washing_liquid": "dishwashingliquid",
}

# Explicitly removed aliases/actions for prompt-code consistency
VH_REMOVED_ACTIONS = {"Walk", "Find", "Run", "WalkTowards"}
VH_TURN_ALIASES: Dict[str, str] = {"RotateLeft": "TurnLeft", "RotateRight": "TurnRight"}
VH_PROMPT_ALIASES = {"MoveAhead"}
VH_PROMPT_ACTION_NAMES = (
    VH_NAVIGATION_NO_OBJECT
    | VH_INTERACTION_ONE_OBJECT
    | VH_INTERACTION_TWO_OBJECTS
    | {"DONE", "FAIL"}
)
_CANONICAL_ACTION_LOOKUP: Dict[str, str] = {
    name.lower(): name
    for name in (
        set(ACTION_NAME_MAP.keys())
        | set(ACTION_NAME_MAP.values())
        | VH_PROMPT_ACTION_NAMES
        | VH_PROMPT_ALIASES
        | VH_REMOVED_ACTIONS
        | set(VH_TURN_ALIASES.keys())
    )
}


def canonicalize_object_type_name(name: Optional[str]) -> str:
    """Normalize object-type spelling to VirtualHome-friendly canonical form."""
    if name is None:
        return ""
    token = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in token:
        token = token.replace("__", "_")
    token = token.strip("_")
    if not token:
        return ""

    if token in OBJECT_TYPE_ALIASES:
        return OBJECT_TYPE_ALIASES[token]

    compact = token.replace("_", "")
    if compact in OBJECT_TYPE_ALIASES:
        return OBJECT_TYPE_ALIASES[compact]

    # VirtualHome class names are often compact multi-word tokens
    # (e.g. lightswitch, tvstand, coffeetable). Keep this fallback.
    if "_" in token and compact:
        return compact

    return token


def normalize_angle_rad(angle_rad: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def normalize_angle_deg(angle_deg: float) -> float:
    """Normalize angle to [-180, 180]."""
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def quat_to_yaw_rad(quat) -> Optional[float]:
    """Convert Unity quaternion [x, y, z, w] to yaw(rad)."""
    try:
        qx, qy, qz, qw = (
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
        )
    except Exception:
        return None
    return math.atan2(2.0 * (qw * qy + qx * qz), 1.0 - 2.0 * (qy * qy + qz * qz))


def yaw_rad_to_quat(yaw_rad: float) -> List[float]:
    """Convert yaw(rad) to Unity quaternion [x, y, z, w]."""
    half = float(yaw_rad) / 2.0
    return [0.0, float(math.sin(half)), 0.0, float(math.cos(half))]


def yaw_deg_to_quat(yaw_deg: float) -> List[float]:
    """Convert yaw(deg) to Unity quaternion [x, y, z, w]."""
    yaw_rad = math.radians(float(yaw_deg))
    return yaw_rad_to_quat(yaw_rad)


def quat_to_yaw_deg(quat) -> Optional[float]:
    """Convert Unity quaternion [x, y, z, w] to yaw(deg)."""
    yaw_rad = quat_to_yaw_rad(quat)
    if yaw_rad is None:
        return None
    return normalize_angle_deg(math.degrees(yaw_rad))


def quantize_quaternion_yaw(quat, step_degrees: float = 30.0) -> Optional[List[float]]:
    """Snap quaternion yaw to nearest discrete step (default 30°) and rebuild quaternion."""
    if step_degrees <= 0:
        return None
    yaw = quat_to_yaw_rad(quat)
    if yaw is None:
        return None
    step_rad = math.radians(float(step_degrees))
    snapped = normalize_angle_rad(round(yaw / step_rad) * step_rad)
    return yaw_rad_to_quat(snapped)


def quantize_yaw_degrees(yaw_deg: float, step_degrees: float = 30.0) -> Optional[float]:
    """Snap yaw(deg) to nearest discrete step and normalize to [-180, 180]."""
    if step_degrees <= 0:
        return None
    snapped = round(float(yaw_deg) / float(step_degrees)) * float(step_degrees)
    return normalize_angle_deg(snapped)


def coerce_init_rotation_quat(
    char_rotation=None,
    char_yaw_degrees=None,
    step_degrees: float = 30.0,
) -> Optional[List[float]]:
    """Coerce init rotation payload to a discrete quaternion.

    Priority:
    1) `char_yaw_degrees` when provided
    2) `char_rotation` quaternion list [x, y, z, w]
    """
    if char_yaw_degrees is not None:
        try:
            snapped_yaw = quantize_yaw_degrees(float(char_yaw_degrees), step_degrees)
        except Exception:
            snapped_yaw = None
        if snapped_yaw is not None:
            return yaw_deg_to_quat(snapped_yaw)

    snapped_quat = quantize_quaternion_yaw(char_rotation, step_degrees)
    if snapped_quat is not None:
        return snapped_quat
    return None

# VirtualHome object State enum -> AI2THOR-like boolean field mapping
VH_STATE_TO_FIELD: Dict[str, Tuple[str, bool]] = {
    "OPEN": ("isOpen", True),
    "CLOSED": ("isOpen", False),
    "ON": ("isToggled", True),
    "OFF": ("isToggled", False),
    "DIRTY": ("isDirty", True),
    "CLEAN": ("isDirty", False),
    "SITTING": ("isSitting", True),
    "LYING": ("isLying", True),
    "PLUGGED_IN": ("isPluggedIn", True),
    "PLUGGED_OUT": ("isPluggedIn", False),
}


# ============================================================================
# Graph Parsing Utilities
# ============================================================================


def find_character_ids(graph: Dict[str, Any]) -> List[int]:
    """Find all character node IDs in the environment graph.

    VirtualHome character nodes have category 'Characters' or class_name
    starting with 'character'.

    Args:
        graph: VirtualHome environment graph dict with 'nodes' and 'edges' keys.

    Returns:
        List of character node IDs (ordered by id).
    """
    char_ids = []
    for node in graph.get("nodes", []):
        cat = node.get("category", "")
        cn = node.get("class_name", "").lower()
        if cat == "Characters" or cn.startswith("character"):
            char_ids.append(node["id"])
    return sorted(char_ids)


def get_char_relations(
    graph: Dict[str, Any],
    char_id: int,
) -> Dict[str, List[int]]:
    """Return dict mapping relation_type -> [to_node_ids] for a character node.

    Args:
        graph: VirtualHome environment graph.
        char_id: The character's node ID.

    Returns:
        Dict like {"INSIDE": [room_id], "CLOSE": [obj1, obj2], "HOLDS_RH": [apple_id], ...}
    """
    result: Dict[str, List[int]] = {}
    for edge in graph.get("edges", []):
        if edge["from_id"] == char_id:
            rel = edge["relation_type"]
            result.setdefault(rel, []).append(edge["to_id"])
    return result


def find_object_id_in_graph(
    graph: Dict[str, Any],
    object_type: str,
    prefer_close: bool = True,
    char_id: Optional[int] = None,
    require_close: bool = False,
) -> Optional[int]:
    """Search the graph for a node whose class_name matches object_type.

    Matching is case-insensitive and underscore-normalised.
    When prefer_close=True and char_id is given, nodes that are CLOSE to the
    character are returned first.

    Args:
        graph: VirtualHome environment graph.
        object_type: Object type to search for (e.g. "microwave", "apple").
        prefer_close: Whether to prefer objects near the character.
        char_id: The character's node ID (used for CLOSE preference).
        require_close: If True, only return objects in CLOSE relation.

    Returns:
        Node ID, or None if not found.
    """
    target = canonicalize_object_type_name(object_type)

    # Build close set
    close_ids: set = set()
    if prefer_close and char_id is not None:
        for edge in graph.get("edges", []):
            if edge["from_id"] == char_id and edge["relation_type"] == "CLOSE":
                close_ids.add(edge["to_id"])

    close_match: Optional[int] = None
    any_match: Optional[int] = None

    for node in graph.get("nodes", []):
        cn = canonicalize_object_type_name(node.get("class_name", ""))
        # Exact match only to avoid "tv" matching "tvstand"
        if cn == target:
            if node["id"] in close_ids:
                close_match = node["id"]
                break
            elif any_match is None:
                any_match = node["id"]

    if close_match is not None:
        return close_match
    if require_close:
        return None
    return any_match


def build_object_metadata(
    graph: Dict[str, Any],
    char_id: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert VirtualHome graph into an AI2THOR-compatible object list.

    This allows existing evaluators in evaluators/base.py (which expects
    metadata["objects"] in AI2THOR format) to work with VirtualHome.

    State mapping:
        OPEN/CLOSED -> isOpen (bool)
        ON/OFF      -> isToggled (bool)
        DIRTY/CLEAN -> isDirty (bool)
        PLUGGED_IN/OUT -> isPluggedIn (bool)

    Relation mapping:
        INSIDE/ON   -> parentReceptacles (list of parent class_names)
        HOLDS_RH/LH -> isPickedUp = True
        CLOSE       -> visible = True, distance ≈ 1.0

    Args:
        graph: VirtualHome environment graph.
        char_id: Character node ID (used to compute visibility & holding).

    Returns:
        (objects_list, inventory_list) both in AI2THOR-compatible format.
    """
    nodes_by_id: Dict[int, Dict] = {n["id"]: n for n in graph.get("nodes", [])}

    # ── Build relationship sets from edges ──────────────────────────────────
    parent_map: Dict[int, List[str]] = {}  # node_id -> [parent class_names]
    held_ids: set = set()
    close_ids: set = set()
    sitting_target_ids: set = set()

    for edge in graph.get("edges", []):
        rel = edge["relation_type"]
        fid = edge["from_id"]
        tid = edge["to_id"]

        if rel in ("INSIDE", "ON"):
            parent_node = nodes_by_id.get(tid, {})
            parent_map.setdefault(fid, []).append(
                parent_node.get("class_name", str(tid))
            )

        if char_id is not None and fid == char_id:
            if rel in ("HOLDS_RH", "HOLDS_LH"):
                held_ids.add(tid)
            elif rel == "CLOSE":
                close_ids.add(tid)
            elif rel == "SITTING":
                sitting_target_ids.add(tid)

    # ── Build objects list ───────────────────────────────────────────────────
    objects: List[Dict[str, Any]] = []
    for node in graph.get("nodes", []):
        # Skip character nodes
        cat = node.get("category", "")
        cn = node.get("class_name", "").lower()
        if cat == "Characters" or cn.startswith("character"):
            continue

        nid = node["id"]
        states: List[str] = node.get("states", [])
        props: List[str] = node.get("properties", [])

        # Convert states to AI2THOR boolean fields
        state_fields: Dict[str, bool] = {}
        for state_str in states:
            mapping = VH_STATE_TO_FIELD.get(state_str.upper())
            if mapping:
                field_name, field_value = mapping
                state_fields[field_name] = field_value

        obj: Dict[str, Any] = {
            "objectType": node["class_name"],
            "objectId": str(nid),
            "name": node.get("prefab_name", node["class_name"]),
            # State booleans (defaults if not specified)
            "isOpen": state_fields.get("isOpen", False),
            "isToggled": state_fields.get("isToggled", False),
            "isOn": state_fields.get("isToggled", False),
            "isDirty": state_fields.get("isDirty", False),
            "isSitting": state_fields.get("isSitting", False)
            or (nid in sitting_target_ids),
            "isLying": state_fields.get("isLying", False),
            "isPluggedIn": state_fields.get("isPluggedIn", False),
            # Derived from relations
            "isPickedUp": nid in held_ids,
            "visible": nid in close_ids,
            "distance": 1.0 if nid in close_ids else 5.0,
            "parentReceptacles": parent_map.get(nid, []),
            # Properties (map to AI2THOR-style boolean capabilities)
            "pickupable": "GRABBABLE" in props,
            "openable": "CAN_OPEN" in props,
            "toggleable": "HAS_SWITCH" in props,
            "receptacle": any(
                p in props for p in ("SURFACES", "CONTAINERS", "RECIPIENT")
            ),
            "sliceable": "CUTTABLE" in props,
            "drinkable": "DRINKABLE" in props,
            "eatable": "EATABLE" in props,
            "readable": "READABLE" in props,
            "movable": "MOVABLE" in props,
        }
        objects.append(obj)

    inventory = [o for o in objects if o["isPickedUp"]]
    return objects, inventory


def generate_text_state(
    graph: Dict[str, Any],
    task_description: str = "",
    char_id: Optional[int] = None,
) -> str:
    """Generate a human-readable text description of the VirtualHome graph state.

    Describes:
    - The room the character is currently in
    - Objects the character is holding
    - Objects close to the character (with their states)
    - Current task

    Args:
        graph: VirtualHome environment graph.
        task_description: Current task description.
        char_id: Character node ID.

    Returns:
        Multi-line text description.
    """
    nodes_by_id: Dict[int, Dict] = {n["id"]: n for n in graph.get("nodes", [])}

    char_room = "unknown"
    held_objects: List[str] = []
    close_objects: List[str] = []

    if char_id is not None:
        for edge in graph.get("edges", []):
            if edge["from_id"] != char_id:
                continue
            rel = edge["relation_type"]
            target = nodes_by_id.get(edge["to_id"], {})

            if rel == "INSIDE" and target.get("category") == "Rooms":
                char_room = target.get("class_name", "unknown")
            elif rel in ("HOLDS_RH", "HOLDS_LH"):
                held_objects.append(target.get("class_name", "object"))
            elif rel == "CLOSE":
                tcat = target.get("category", "")
                tcn = target.get("class_name", "")
                if tcn and tcat not in ("Rooms", "Characters"):
                    states = target.get("states", [])
                    state_str = f" [{','.join(states)}]" if states else ""
                    close_objects.append(f"{tcn}(id={edge['to_id']}){state_str}")

    lines = [
        f"Current room: {char_room}",
        f"Holding: {', '.join(held_objects) if held_objects else 'nothing'}",
        f"Nearby objects: {', '.join(close_objects[:12]) if close_objects else 'none'}",
    ]
    if task_description:
        lines.append(f"Task: {task_description}")
    return "\n".join(lines)


def format_vh_action_dict(action_dict: Dict[str, Any]) -> str:
    """Format a structured VH action dict into the prompt-compliant action string."""
    if not isinstance(action_dict, dict):
        raise ValueError(f"Expected dict action, got {type(action_dict).__name__}")

    action_name = str(action_dict.get("action_name", "")).strip()
    if not action_name:
        raise ValueError("Missing action_name")
    action_name = _CANONICAL_ACTION_LOOKUP.get(action_name.lower(), action_name)

    upper_name = action_name.upper()
    if upper_name in {"DONE", "FAIL"}:
        return upper_name

    object_type = action_dict.get("object_type")
    object2_type = action_dict.get("object2_type")
    vh_name = ACTION_NAME_MAP.get(action_name, action_name)

    if vh_name == "WalkForward":
        granularity = action_dict.get("granularity")
        if granularity:
            return f"{action_name}({str(granularity).lower()})"
        return action_name

    if vh_name in ("TurnLeft", "TurnRight"):
        mod = str(action_dict.get("turn_modifier", "")).strip().lower()
        if mod == "small":
            return f"{action_name}(small)"
        if mod == "normal":
            return f"{action_name}(normal)"

        deg = action_dict.get("turn_degrees")
        if deg is not None:
            try:
                deg_value = float(deg)
            except (TypeError, ValueError):
                deg_value = None
            if deg_value is not None:
                if abs(deg_value - 30.0) < 1e-6:
                    return f"{action_name}(small)"
                if abs(deg_value - 90.0) < 1e-6:
                    return action_name
        return action_name

    if object_type and object2_type:
        return f"{action_name}({object_type}, {object2_type})"
    if object_type:
        return f"{action_name}({object_type})"
    return action_name


# ============================================================================
# Action String Parser (VirtualHome style)
# ============================================================================


def parse_vh_action_string(
    action_string: str, *, strict_prompt: bool = True
) -> Dict[str, Any]:
    """Parse a VirtualHome-style action string into an action dictionary.

    Supported formats:
        TurnLeft                    -> body turn 90° (default; Unity uses 30°/step)
        TurnLeft(small)             -> 30° (one native VH turn)
        TurnLeft(normal)            -> 90°
        TurnLeft(90) / TurnLeft(30) -> rejected in strict prompt mode
        WalkForward                 -> short forward (default small)
        WalkForward(small|medium|large) -> step length presets (repeat counts)
        WalkForward(0.5)            -> rejected in strict prompt mode
        Grab(apple)                 -> single-object interaction
        PutBack(apple, table)       -> two-object interaction
        PutIn(glass, microwave)     -> two-object interaction
        DONE                        -> task_completion
        FAIL                        -> task_completion

    Args:
        action_string: Raw action string.

    Returns:
        Action dict with action_type, action_name, and optional object_type / object2_type.

    Raises:
        ValueError: If the action format is unrecognised.
    """
    action_string = action_string.strip()

    # Task completion
    if action_string.upper() == "DONE":
        return {"action_type": "task_completion", "action_name": "DONE"}
    if action_string.upper() == "FAIL":
        return {"action_type": "task_completion", "action_name": "FAIL"}

    # With parentheses: ActionName(obj) or ActionName(obj1, obj2)
    if "(" in action_string and action_string.endswith(")"):
        action_name = action_string[: action_string.index("(")]
        params_str = action_string[action_string.index("(") + 1 : -1].strip()
        params = [p.strip() for p in params_str.split(",") if p.strip()]
    else:
        # No parentheses
        action_name = action_string
        params = []

    canonical_name = _CANONICAL_ACTION_LOOKUP.get(action_name.strip().lower())
    if canonical_name:
        action_name = canonical_name

    if action_name in VH_REMOVED_ACTIONS:
        raise ValueError(
            f"'{action_name}' has been removed; use WalkForward + TurnLeft/TurnRight to navigate"
        )
    if action_name in VH_TURN_ALIASES:
        raise ValueError(
            f"'{action_name}' is not allowed; use '{VH_TURN_ALIASES[action_name]}' instead"
        )
    if strict_prompt:
        if action_name not in VH_PROMPT_ACTION_NAMES and action_name not in VH_PROMPT_ALIASES:
            raise ValueError(
                f"'{action_name}' is not allowed by the VirtualHome prompt action space"
            )

    # Determine VH action name
    vh_name = ACTION_NAME_MAP.get(action_name, action_name)

    # Classify action type
    if vh_name in VH_NAVIGATION_WITH_OBJECT:
        action_type = "navigation"
        if not params:
            raise ValueError(f"'{action_name}' requires an object target")
        return {
            "action_type": action_type,
            "action_name": vh_name,
            "object_type": params[0],
        }
    elif vh_name in VH_NAVIGATION_NO_OBJECT:
        nav: Dict[str, Any] = {"action_type": "navigation", "action_name": vh_name}

        if vh_name == "WalkForward":
            if params:
                raw = params[0].strip()
                low = raw.lower()
                if low in ("small", "medium", "large"):
                    nav["granularity"] = low
                else:
                    if strict_prompt:
                        raise ValueError(
                            f"WalkForward only accepts small|medium|large in prompt mode, got {raw!r}"
                        )
                    try:
                        nav["magnitude"] = float(raw) if "." in raw else float(int(raw))
                    except ValueError as exc:
                        raise ValueError(
                            f"WalkForward parameter must be small|medium|large or meters "
                            f"(e.g. 0.25), got {raw!r}"
                        ) from exc
            return nav

        if vh_name in ("TurnLeft", "TurnRight"):
            if len(params) > 1:
                raise ValueError(
                    f"{action_name} accepts at most one argument "
                    f"(small | normal | degrees e.g. 30 or 90)"
                )
            if not params:
                return nav
            raw = params[0].strip()
            low = raw.lower()
            if low == "small":
                nav["turn_degrees"] = 30.0
                nav["turn_modifier"] = "small"
            elif low == "normal":
                nav["turn_degrees"] = 90.0
                nav["turn_modifier"] = "normal"
            else:
                if strict_prompt:
                    raise ValueError(
                        f"{action_name} only accepts small or normal in prompt mode, got {raw!r}"
                    )
                try:
                    deg = float(raw) if "." in raw else float(int(raw))
                except ValueError as exc:
                    raise ValueError(
                        f"{action_name} argument must be small, normal, or numeric degrees; "
                        f"got {raw!r}"
                    ) from exc
                nav["turn_degrees"] = deg
            return nav

        if params:
            raise ValueError(
                f"'{action_name}' does not take parameters (got {', '.join(params)!r})"
            )
        return nav

    elif vh_name in VH_INTERACTION_TWO_OBJECTS:
        if len(params) < 2:
            raise ValueError(
                f"'{action_name}' requires two objects (e.g., PutBack(apple, table))"
            )
        return {
            "action_type": "interaction",
            "action_name": vh_name,
            "object_type": canonicalize_object_type_name(params[0]),
            "object2_type": canonicalize_object_type_name(params[1]),
        }
    elif vh_name in VH_INTERACTION_ONE_OBJECT:
        if not params:
            # Some actions (StandUp etc.) may appear here without objects
            if vh_name in ("StandUp", "Sleep", "WakeUp"):
                return {"action_type": "interaction", "action_name": vh_name}
            raise ValueError(f"'{action_name}' requires an object (e.g., Grab(apple))")
        primary = canonicalize_object_type_name(params[0])
        secondary = (
            canonicalize_object_type_name(params[1]) if len(params) > 1 else None
        )
        return {
            "action_type": "interaction",
            "action_name": vh_name,
            "object_type": primary,
            **({"object2_type": secondary} if secondary else {}),
        }
    else:
        # Fallback: treat as interaction if params given, else navigation
        if params:
            primary = canonicalize_object_type_name(params[0])
            secondary = (
                canonicalize_object_type_name(params[1]) if len(params) > 1 else None
            )
            return {
                "action_type": "interaction",
                "action_name": vh_name,
                "object_type": primary,
                **({"object2_type": secondary} if secondary else {}),
            }
        return {"action_type": "navigation", "action_name": vh_name}
