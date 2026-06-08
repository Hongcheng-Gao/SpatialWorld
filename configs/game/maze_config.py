#!/usr/bin/env python3
"""
        
  API       
"""

from typing import Dict, Any
import json

#     "model": "gpt-4.1"
# }

OPENAI_CONFIG = {
    "api_base_url": "http.",
    "api_key": "REDACTED",
    "model": "Gemini-3-Flash-Preview"
}


#       
EVALUATION_CONFIG = {
    "max_steps": 100,  #       
    "decision_frequency": 1.0,  #      (Hz)
    "video_fps": 1.0,  #     
    "history_window": 29,  #      29    ；       1      
    "retry_times": 3,  # API            
}

#      
SYSTEM_PROMPT = """
You are a professional 3D maze navigation AI. Your ultimate goal is to find the exit of the maze. You must carefully observe the first-person game screen, read the HUD data, and choose the correct action each step.

### Visual Elements & UI to Recognize:
- **Top-Left HUD**: Displays your current `POS` (X, Y coordinates) in cyan. Use this to track your position and detect when you are stuck (position not changing across steps).
- **Top-Right HUD**: Displays your `DIR` (facing direction): NORTH, SOUTH, EAST, or WEST. This tells you which way you are currently looking.
- **Top-Center Compass**: A circular compass with a red triangular arrow pointing in your current facing direction, and an "N" label at the north side. Use it to quickly read your orientation.
- **Walls**: Brick-red / orange-brown blocks. A wall directly in front means `move_forward` is blocked. Side walls indicate corridors to the left or right.
- **Exit**: A glowing gold / orange panel that pulses. Moving into it completes the maze.
- **Floor & Ceiling**: The floor is gray (lighter near you, darker at the horizon). The ceiling is near-black dark blue. These are not interactive.

### Navigation Strategy:
- **If the path ahead is open (no wall blocking you)** , you can use `move_forward` to make progress. Choose the movement granularity based on confidence and corridor length: `small` for cautious 1-cell movement, `medium` for 2 clear cells, and `large` for 3 clear cells.
- **If there is a wall directly ahead** , you can turn to find an open direction, then move forward.
- **Avoid spinning in place** — if you have turned multiple times without moving, commit to a direction and move forward.

### Available Actions (Control Mapping):
- `move_forward`: Move forward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a wall.
- `turn_left`: Turn 90 degrees to the left (changes your DIR).
- `turn_right`: Turn 90 degrees to the right (changes your DIR).

### Movement Granularity:
- Include `granularity` only when the action is `move_forward`.
- Use exactly one of: `small`, `medium`, `large`.
- For `turn_left` and `turn_right`, do not include a `granularity` field.

"""

#     
GAME_CONFIG = {
    "maze_file": 'data/maze3d/Level_01.txt',  #       
    "level_number": 1,  #      (1-20)
}

#     
OUTPUT_CONFIG = {
    "log_dir": "logs",
    "video_output_dir": "videos",
}

def get_level_file(level_number: int) -> str:
    """              

    Args:
        level_number:      (1-20)

    Returns:
              
    """
    if level_number < 1 or level_number > 20:
        raise ValueError(f"       1-20  ，   : {level_number}")

    #             
    level_str = f"{level_number:02d}"
    return f"data/maze3d/Level_{level_str}.txt"


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
    config = get_config()
    print("      :")
    print(json.dumps(config, indent=2, ensure_ascii=False))
