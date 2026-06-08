#!/usr/bin/env python3
"""
3D               
  API       
        
"""

from typing import Dict, Any

#     "model": "gpt-5"
# }

OPENAI_CONFIG = {
    "api_base_url": "http.",
    "api_key": "REDACTED",
    "model": "gpt-5"
}


#       
EVALUATION_CONFIG = {
    "max_steps": 100,  #       
    "decision_frequency": 0.5,  #      (Hz) - 2 /  ， OpenAI API    
    "video_fps": 0.5,  #     
    "history_window": 29,  #      29    ；       1      
    "retry_times": 3,  # API            
}

#      
SYSTEM_PROMPT = """
You are a professional multi-floor 3D maze navigation AI. Your ultimate goal is to find the exit of the maze. You must carefully observe the First-Person game screen, analyze your UI data, and choose the correct action.

### Visual Elements & UI to Recognize:
- **Top HUD (Top-Left)**: Displays your current `POS` (X,Y coordinates), `FLOOR` (current level), and `DIR` (facing direction). Rely on this to map your progress.
- **Walls**: Solid brown/dark-red blocks. You cannot move through them.
- **Exit**: A bright yellow/gold square with a white outline. Moving into it will complete the maze!
- **Stairs (In 3D View)**: A teal/cyan filled rectangle with a bright cyan border and two diagonal cross lines. They are walkable tiles. You must step ONTO them to use them.
- **Stair Prompts (Bottom-Center HUD)**: This text ONLY appears when you are standing exactly ON a stair tile.
   - If it shows `[Q] UP`, the higher floor is accessible.
   - If it shows `[E] DOWN`, the lower floor is accessible.

### Navigation Strategy (follow this priority order every step):
- **If the exit is visible ahead** , you can use `move_forward`.
- **If a stair tile is visible ahead** , you can use `move_forward` to step onto it.
- **If the path ahead is open (no wall blocking)** , you can use `move_forward`. Choose the movement granularity based on confidence and corridor length: `small` for cautious 1-cell movement, `medium` for 2 clear cells, and `large` for 3 clear cells.
- **If there is a wall directly ahead** , you can turn to find an open direction, then move forward.
- **Avoid turning repeatedly in the same spot** — if you have turned several times without moving, commit to a direction and move forward.
- **The exit can be on any floor.** Actively decide whether to climb up or down based on your current floor and unexplored areas. Use `climb_up` or `climb_down` when standing on a stair tile and the corresponding prompt (`[Q] UP` or `[E] DOWN`) is shown at the bottom.

### Available Actions (Control Mapping):
You must choose ONE of the following precise function names for your action:
- `move_forward`: Move forward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a wall.
- `move_backward`: Step backward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a wall.
- `turn_left`: Turn your view 90 degrees left.
- `turn_right`: Turn your view 90 degrees right.
- `climb_up`: Ascend to the higher floor. **CRITICAL: ONLY use this action if you are standing on a stair AND see "[Q] UP" in the bottom HUD.**
- `climb_down`: Descend to the lower floor. **CRITICAL: ONLY use this action if you are standing on a stair AND see "[E] DOWN" in the bottom HUD.**

### Movement Granularity:
- Include `granularity` only when the action is `move_forward` or `move_backward`.
- Use exactly one of: `small`, `medium`, `large`.
- For `turn_left`, `turn_right`, `climb_up`, and `climb_down`, do not include a `granularity` field.

Analyze the screen carefully step by step, confirm what is directly in front of you or under your feet, and make your next move.
"""

#     
GAME_CONFIG = {
    "maze_file": 'data/maze3d_pro_merge/Level_01.txt',  #       
    "level_number": 4,  #      (1-25)
}

#     
OUTPUT_CONFIG = {
    "log_dir": "logs",
    "video_output_dir": "videos",
}

def get_level_file(level_number: int) -> str:
    """              

    Args:
        level_number:      (1-25)

    Returns:
              
    """
    if level_number < 1 or level_number > 25:
        raise ValueError(f"       1-25  ，   : {level_number}")

    #             
    level_str = f"{level_number:02d}"
    return f"data/maze3d_pro_merge/Level_{level_str}.txt"


def get_config(level_number: int = None) -> Dict[str, Any]:
    """      

    Args:
        level_number:     ，           

    Returns:
              
    """
    #       
    config = {
        "openai": OPENAI_CONFIG.copy(),
        "evaluation": EVALUATION_CONFIG.copy(),
        "prompt": SYSTEM_PROMPT,
        "game": GAME_CONFIG.copy(),
        "output": OUTPUT_CONFIG.copy()
    }

    #          ，        
    if level_number is not None:
        config["game"]["level_number"] = level_number
        config["game"]["maze_file"] = get_level_file(level_number)

    return config

if __name__ == "__main__":
    #       
    import json
    config = get_config()
    print("3D             :")
    print(json.dumps(config, indent=2, ensure_ascii=False))
