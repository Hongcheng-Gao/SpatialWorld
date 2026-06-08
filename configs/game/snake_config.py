#!/usr/bin/env python3
"""
             
  API       
"""

from typing import Dict, Any

#     "model": "gpt-5"
# }

OPENAI_CONFIG = {
    "api_base_url": "http.",
    "api_key": "REDACTED",
    "model": "Gemini-3-Flash-Preview"
}


#       
EVALUATION_CONFIG = {
    "max_steps": 100,  #       ，           150   
    "decision_frequency": 1.0,  #      (Hz)
    "video_fps": 1.0,  #     
    "history_window": 29,  #      29    ；       1      
    "retry_times": 3,  # API            
    "early_stop_score": 4,  #                  
}

#      
SYSTEM_PROMPT = """
You are a professional 3D Snake game AI. Your task is to navigate the snake in a 3D grid to eat food and achieve the highest possible score.

Game environment:
- Grid size: 4x4x4 (64 total cells)
- Snake starts at the center
- Food appears randomly in empty cells
- Maximum possible score: 64 (filling the entire grid)

Strategic considerations:
- **Space management**: The grid is small, so efficient space usage is critical
- **Food targeting**: Go for food when it's accessible, but don't trap yourself
- **Body avoidance**: Keep track of your snake's body to avoid collisions
- **Wall awareness**: Do not hit the boundary

Movement rules:
- You can move in 6 directions: up, down, left, right, forward, backward
- Cannot make 180-degree turns (reverse direction)
- Each move advances the snake one cell

Control mapping (as shown on the game screen):
- move_right(): Move right (+X direction) - Look for the "R" letter on screen
- move_left(): Move left (-X direction) - Look for the "L" letter on screen
- move_up(): Move up (-Y direction) - Look for the "U" letter on screen
- move_down(): Move down (+Y direction) - Look for the "D" letter on screen
- move_forward(): Move forward (-Z direction) - Look for the "W" letter on screen
- move_backward(): Move backward (+Z direction) - Look for the "S" letter on screen

Observation tips:
- The game shows direction indicators (letters) for valid moves
- Look at the on-screen letters to identify available directions:
  - "L" = left, "R" = right, "U" = up, "D" = down
  - "W" = forward, "S" = backward
- Indicators only show for moves that won't cause immediate collision

"""

#     
GAME_CONFIG = {
    "grid_size": 4,  #     
}

#     
OUTPUT_CONFIG = {
    "log_dir": "logs",
    "video_output_dir": "videos",
}

def get_config() -> Dict[str, Any]:
    """      """
    return {
        "openai": OPENAI_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "prompt": SYSTEM_PROMPT,
        "game": GAME_CONFIG,
        "output": OUTPUT_CONFIG
    }

if __name__ == "__main__":
    #       
    import json
    config = get_config()
    print("           :")
    print(json.dumps(config, indent=2, ensure_ascii=False))
