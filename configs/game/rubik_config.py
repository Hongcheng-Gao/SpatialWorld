#!/usr/bin/env python3
"""
          
  API       
"""

from typing import Dict, Any

OPENAI_CONFIG = {
    "api_base_url": "http.",
    "api_key": "REDACTED",
    "model": "Gemini-3-Flash-Preview"
}

#       
EVALUATION_CONFIG = {
    "max_steps": 50,  #       ,           20 
    "decision_frequency": 0.5,  #      (Hz) -  OpenAI API    
    "video_fps": 0.5,  #     
    "history_window": 29,  #      29    ；       1      
    "retry_times": 3,  # API            
}

#      
SYSTEM_PROMPT = """
You are a Rubik's Cube solving expert. Your task is to solve the 3D Rubik's Cube by strategically rotating its faces.

FACE IDENTIFICATION:
Each face has a letter in the center and a distinct color:
- F (FRONT) - Red face with F letter in center
- B (BACK) - Orange face with B letter in center
- L (LEFT) - Blue face with L letter in center
- R (RIGHT) - Green face with R letter in center
- U (UP) - Yellow face with U letter in center
- D (DOWN) - White face with D letter in center

MOVE TYPES:
- FACE ROTATIONS (90 degrees, with animation): Rotate a single face clockwise (cw) or anticlockwise (acw).
- VIEW ROTATIONS (45 degrees, immediate): Rotate the entire cube around an axis (x/y/z) to change your viewing angle without changing the cube state.

Important tips:
- First, analyze the current state of the cube carefully
- Identify which pieces are in the correct positions and which need to be moved
- Plan your moves carefully to avoid unnecessary rotations

"""

#       
GAME_CONFIG = {
    "initial_config": "1",  #      (1-20)

}

#     
OUTPUT_CONFIG = {
    "log_dir": "logs",
    "video_output_dir": "videos",
}

#       
RUBIK_ACTIONS = {
    #      
    "rotate_front_cw": ("FRONT", "CLOCKWISE"),
    "rotate_front_acw": ("FRONT", "ANTICLOCKWISE"),
    "rotate_back_cw": ("BACK", "CLOCKWISE"),
    "rotate_back_acw": ("BACK", "ANTICLOCKWISE"),
    "rotate_left_cw": ("LEFT", "CLOCKWISE"),
    "rotate_left_acw": ("LEFT", "ANTICLOCKWISE"),
    "rotate_right_cw": ("RIGHT", "CLOCKWISE"),
    "rotate_right_acw": ("RIGHT", "ANTICLOCKWISE"),
    "rotate_up_cw": ("UP", "CLOCKWISE"),
    "rotate_up_acw": ("UP", "ANTICLOCKWISE"),
    "rotate_down_cw": ("DOWN", "CLOCKWISE"),
    "rotate_down_acw": ("DOWN", "ANTICLOCKWISE"),

    #        (45 )
    "rotate_cube_x_cw": ("RIGHT", "CLOCKWISE"),  # X       (RIGHT face)
    "rotate_cube_x_acw": ("RIGHT", "ANTICLOCKWISE"),  # X       (RIGHT face)
    "rotate_cube_y_cw": ("UP", "CLOCKWISE"),  # Y       (UP face)
    "rotate_cube_y_acw": ("UP", "ANTICLOCKWISE"),  # Y       (UP face)
    "rotate_cube_z_cw": ("FRONT", "CLOCKWISE"),  # Z       (FRONT face)
    "rotate_cube_z_acw": ("FRONT", "ANTICLOCKWISE"),  # Z       (FRONT face)
}


def get_config() -> Dict[str, Any]:
    """      """
    return {
        "openai": OPENAI_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "prompt": SYSTEM_PROMPT,
        "game": GAME_CONFIG,
        "output": OUTPUT_CONFIG,
        "actions": RUBIK_ACTIONS
    }


if __name__ == "__main__":
    #       
    import json
    config = get_config()
    print("        :")
    print(json.dumps(config, indent=2, ensure_ascii=False))