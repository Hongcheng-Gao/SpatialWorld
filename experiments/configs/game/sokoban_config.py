"""
       
"""

OPENAI_CONFIG = {
    "api_base_url": "htt.",
    "api_key": "REDACTED",
    "model": "gpt-5"
}

# System prompt
SYSTEM_PROMPT = """
You are a Sokoban game expert. Game rules:
- Red square: Player
- Orange square: Box
- Green square: Target position
- Gray square: Wall

Game objective: Push boxes to green target positions.

Important rules:
1. You can only push boxes, not pull them
2. Boxes cannot pass through walls or other boxes
3. If a box is pushed into a corner, it may get stuck and become unmovable
4. Plan your movement path carefully to avoid deadlocks

The current game state is displayed on the map. Analyze the current situation, develop a strategy, and choose the best move direction.
"""

#       
EVALUATION_CONFIG = {
    "max_steps": 200,  #       
    "decision_frequency": 1.0,  #      (Hz)
    "video_fps": 1.0,  #     
    "history_window": 29,  #      29    ；       1      
    "retry_times": 3,  # API            
}

#     
OUTPUT_CONFIG = {
    "log_dir": "logs",
    "video_dir": "videos",
    "max_steps": 200
}

#       
LEVEL_CONFIG = {
    "available_levels": [0, 1, 2, 3, 4],
    "level_descriptions": {
        0: "15x15     ",
        1: "15x15     ",
        2: "22x11    ",
        3: "20x20     ",
        4: "25x20     "
    }
}