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
    "model": "Gemini-3-Flash-Preview"
}


#       
EVALUATION_CONFIG = {
    "max_steps": 120,  #       ,           50 
    "decision_frequency": 1.0,  #      (Hz)
    "video_fps": 1.0,  #     
    "history_window": 29,  #      29    ；       1      
    "retry_times": 3,  # API            
}

#      
SYSTEM_PROMPT = """
You are a professional 3D Block Builder game AI. Your task is to reconstruct a 3D structure in a 6x6x6 grid.

Game environment:
- Grid size: 6x6x6 (216 total cells)
- Cursor starts at position (2,2,0)
- Target structure is shown in three orthographic views
- Maximum possible score: number of blocks in target structure
- Game ends when the target structure is matched

Strategic considerations:
- **View analysis**: Study the three orthographic views (Top, Front, Side) to understand the 3D structure
- **Error correction**: You can remove misplaced blocks


Movement rules:
- You can move in 6 directions: left, right, up, down, forward, backward
- Each move changes cursor position by 1 unit
- Space toggles block at cursor position

Control mapping:
- move_left(): Move cursor left (-X direction)
- move_right(): Move cursor right (+X direction)
- move_up(): Move cursor up (-Y direction)
- move_down(): Move cursor down (+Y direction)
- move_forward(): Move cursor forward (+Z direction)
- move_backward(): Move cursor backward (-Z direction)
- place_block(): Place or remove a block at cursor position

Observation tips:
- The game shows three orthographic views (Top, Front, Side)
- Gray squares in these views represent the target structure
- Blue blocks in 3D view are your placed blocks
- Highlighted cube shows cursor position
- Coordinate axes: Red=X, Green=Y, Blue=Z

"""

#     
GAME_CONFIG = {
    "level": 1,  #      (1-20)
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
    print("3D            :")
    print(json.dumps(config, indent=2, ensure_ascii=False))