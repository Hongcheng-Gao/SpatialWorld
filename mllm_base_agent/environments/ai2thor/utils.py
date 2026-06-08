"""
SpatialWorld Tool Wrapper
Wraps environment interactions as SpatialWorld tools
"""
from typing import Optional
from mllm_base_agent.tools import tool
from core.llm.schemas import EnvAction, EnvObservation

# Global environment instance (set in main)
_env_instance = None


def set_env_instance(env):
    """Set global environment instance
    
    Args:
        env: AI2ThorEnvWrapper instance
    """
    global _env_instance
    _env_instance = env


@tool
def step_3d_env(
    move: Optional[str] = None,
    turn: Optional[float] = None,
    interact: bool = False,
    comment: Optional[str] = None
) -> dict:
    """Execute one high-level action in 3D indoor environment (AI2-THOR)
    
    This is your operation interface in the virtual 3D kitchen environment. You can use this tool to:
    - Move forward/back/left/right (move parameter)
    - Rotate view left/right (turn parameter, positive for right, negative for left)
    - Interact with objects in front, such as opening fridge door (interact parameter)
    
    After executing an action, you will receive a new observation, including:
    - New viewpoint image path
    - Text description of current position and visible objects
    - Feedback on whether the action succeeded
    
    Suggested strategy:
    1. If you don't know where the target object is, first rotate (turn=90 or turn=-90) to look around
    2. After seeing the target object, approach by moving (move="forward")
    3. After reaching in front of the target object, use interaction (interact=True) to open or operate
    
    Args:
        move: Movement direction, optional values: "forward", "back", "left", "right"
        turn: Rotation angle (degrees), positive for right turn, negative for left turn, recommended ±90 degrees
        interact: Whether to attempt interaction with object directly ahead (open, pick up, etc.)
        comment: Brief description of this action step, used to record your thinking process
        
    Returns:
        Dictionary containing the following fields:
        - image_path: Image path of new viewpoint
        - text_state: Text description of environment state
        - reward: Reward obtained in this step
        - done: Whether task is complete
        - success: Whether action executed successfully
    """
    if _env_instance is None:
        raise RuntimeError("Environment instance not initialized, please call set_env_instance() first")
    
    # Create action object
    action = EnvAction(
        move=move,
        turn=turn,
        interact=interact,
        comment=comment
    )
    
    # Execute action
    observation = _env_instance.step(action)
    
    # Return result (convert to dict for LLM parsing)
    return {
        "image_path": observation.image_path,
        "text_state": observation.text_state,
        "reward": observation.reward,
        "done": observation.done,
        "success": observation.metadata.get("lastActionSuccess", False)
    }


# Tool list for graph use
env_tools = [step_3d_env]
