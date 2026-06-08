"""VirtualHome environment no-summary system prompt."""

from typing import Iterable, List, Optional


_DEFAULT_COMMON_INTERACTABLE_OBJECTS = [
    "microwave",
    "fridge",
    "stove",
    "dishwasher",
    "coffeemaker",
    "toaster",
    "washingmachine",
    "tv",
    "laptop",
    "computer",
    "cellphone",
    "tablelamp",
    "lightswitch",
    "bookshelf",
    "cabinet",
    "drawer",
    "garbagecan",
    "sink",
    "bathtub",
    "toilet",
    "box",
    "bowl",
    "pot",
    "fryingpan",
    "cup",
    "glass",
    "wineglass",
    "mug",
    "breadslice",
    "table",
    "coffeetable",
    "kitchentable",
    "desk",
    "kitchencounter",
    "counter",
    "bed",
    "sofa",
    "chair",
    "armchair",
    "stool",
    "shelves",
    "tvstand",
    "apple",
    "bread",
    "cupcake",
    "tomato",
    "potato",
    "carrot",
    "egg",
    "plate",
    "cutleryfork",
    "cutleryknife",
    "spoon",
    "remotecontrol",
    "book",
    "newspaper",
    "dishbowl",
    "dishwashingliquid",
    "faucet",
    "folder",
    "keyboard",
    "mouse",
    "pie",
    "pillow",
    "radio",
    "salmon",
    "barsoap",
    "soapbox",
    "toothbrush",
    "toothpaste",
    "toiletpaper",
    "poundcake",
]


def _normalize_object_type_token(token: str) -> str:
    norm = str(token or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in norm:
        norm = norm.replace("__", "_")
    norm = norm.strip("_")
    if "_" in norm:
        # VirtualHome class names are usually compact, e.g. soapbox / coffeetable.
        norm = norm.replace("_", "")
    return norm


def _dedupe_and_sort_object_types(object_types: Iterable[str]) -> List[str]:
    uniq = set()
    for token in object_types:
        norm = _normalize_object_type_token(token)
        if norm:
            uniq.add(norm)
    return sorted(uniq)


def _format_object_type_lines(tokens: List[str], chunk_size: int = 16) -> str:
    if not tokens:
        return "- (none)"
    lines = []
    for i in range(0, len(tokens), chunk_size):
        lines.append("- " + ", ".join(tokens[i : i + chunk_size]))
    return "\n".join(lines)


def build_virtualhome_interactable_objects_section(
    interactable_object_types: Optional[List[str]] = None,
) -> str:
    """Build object-name guidance block for VirtualHome prompts."""
    scene_tokens = _dedupe_and_sort_object_types(interactable_object_types or [])
    if scene_tokens:
        source_line = "- Source: current scene graph (runtime extracted)."
        object_lines = _format_object_type_lines(scene_tokens)
    else:
        source_line = "- Source: fallback common object set (scene extraction unavailable)."
        object_lines = _format_object_type_lines(_DEFAULT_COMMON_INTERACTABLE_OBJECTS)

    return (
        "**Interactable Objects in VirtualHome (use EXACT object_type tokens):**\n"
        f"{source_line}\n"
        f"{object_lines}\n"
        "- Naming rule."
        "  - Use the exact token in action args, e.g. `Grab(soapbox)`.\n"
        "  - Do not insert spaces: use `soapbox`, not `soap box`.\n"
        "  - Prefer compact lowercase names, e.g. `coffeetable`, `lightswitch`."
    )

VIRTUALHOME_THINK_SYSTEM_PROMPT_NO_SUMMARY = """You are an embodied agent executing tasks in the VirtualHome (Unity) environment. Your goal is to complete tasks like a human through visual observation and step-by-step actions.

**Embodiment & Sensing:**
- First-person camera field of view (FOV): 60 degrees.
- Maximum interaction distance: 1.5 m. Move closer before interacting.

**Scene Layout:**
- Scenes usually contain connected rooms. If the target is not visible, explore reachable rooms systematically.

{interactable_objects_section}

**Unified Action Space (use these names exactly):**
- Navigation: `Move(direction, granularity)`. VirtualHome supports `Move(forward, Small|Medium|Large)`; rotate first to move in other directions.
- Viewpoint & posture: `Rotate(left|right, small|normal)`, `Tilt(up|down)`, `ChangePosture(stand_up)`.
- Interaction: `Pick(obj)`, `Place(obj, target)`, `ChangeState(obj, state)`, `Manipulate(obj, action)`.
- Supported state values: `open`, `close`, `on`, `off`.
- Supported manipulation actions: `drink`, `sit`, `touch`, `look_at`.
- Task control: `EndTask(DONE)` or `EndTask(FAIL)`.

**Important Notes:**
- Use exact lowercase object tokens from the interactable-object list.
- `Place(obj, target)` places the held object onto or into the target according to backend affordances; if containment is required, use `Place(obj, target, in)`.
- VirtualHome is semantic: no separate tools are required for supported interactions.
- Do not mention or reason about any hidden step budget.

**Human-like Behavior Guidance:**
- **Spatial reasoning & navigation**: Carefully observe the current image and decide whether to explore, approach, or interact.
- **Pre-interaction confirmation**: Confirm that object type and state are correct.
- **Self-verification**: Before ending, confirm the goal state is satisfied.

Your thinking process should include:
- **Observation Description**: What key objects are in the current image and where they are.
- **Reasoning Analysis**: Where the target might be, how far it is, and what subgoals remain.
- **Action Planning**: What action to take next and why.

**Current Task:**
{task_prompt}

**Output Format:**
<THINK>
Observation Description: ...
Reasoning Analysis: ...
Action Planning: ...
</THINK>
<ACTION>
Move(forward, Medium) or Rotate(left, small) or Tilt(down) or Pick(apple) or Place(apple, table) or ChangeState(fridge, open) or Manipulate(glass, drink) or EndTask(DONE) or EndTask(FAIL)
</ACTION>

**Examples:**
- Navigate: `<ACTION>Move(forward, Medium)</ACTION>`
- Turn: `<ACTION>Rotate(right, normal)</ACTION>`
- Open: `<ACTION>ChangeState(fridge, open)</ACTION>`
- Pick: `<ACTION>Pick(apple)</ACTION>`
- Place: `<ACTION>Place(apple, microwave, in)</ACTION>`
- Finish: `<ACTION>EndTask(DONE)</ACTION>`"""

VIRTUALHOME_THINK_SYSTEM_PROMPT = VIRTUALHOME_THINK_SYSTEM_PROMPT_NO_SUMMARY


def get_virtualhome_prompt(
    enable_summary: bool = False,
    interactable_object_types: Optional[List[str]] = None,
) -> str:
    """Return the VirtualHome no-summary prompt with optional scene-specific object list."""
    section = build_virtualhome_interactable_objects_section(
        interactable_object_types=interactable_object_types
    )
    return VIRTUALHOME_THINK_SYSTEM_PROMPT_NO_SUMMARY.replace("{interactable_objects_section}", section)
