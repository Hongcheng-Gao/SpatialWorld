"""
MLLM      

   AI   Pygame     ，               
"""

__version__ = "1.0.0"
__author__ = "MLLM Game Evaluation Framework"

#       
from core.data_classes import FrameData, GameState, Action
from core.input_source import GameInputSource
from core.evaluation_engine import EvaluationEngine
from input_sources.pygame_input_source import PygameInputSource
from utils.logger import GameEvaluationLogger, setup_global_logger, get_logger

__all__ = [
    #      
    "FrameData",
    "GameState",
    "Action",

    #     
    "GameInputSource",

    #     
    "PygameInputSource",

    #     
    "EvaluationEngine",

    #     
    "GameEvaluationLogger",
    "setup_global_logger",
    "get_logger",
]