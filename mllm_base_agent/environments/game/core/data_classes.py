"""
       
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import time


@dataclass
class FrameData:
    """    """
    image: np.ndarray  #     ，    (H, W, C)
    timestamp: float   #    
    frame_number: int  #    
    metadata: Dict[str, Any]  #      

    def __post_init__(self):
        """    """
        if not isinstance(self.image, np.ndarray):
            raise ValueError("image must be a numpy array")
        if len(self.image.shape) != 3:
            raise ValueError("image must have shape (H, W, C)")
        if self.image.shape[2] not in [3, 4]:
            raise ValueError("image must have 3 or 4 channels")


@dataclass
class GameState:
    """     """
    raw_state: Dict[str, Any]  #       
    normalized_state: Dict[str, Any]  #      

    def __post_init__(self):
        """    """
        if not isinstance(self.raw_state, dict):
            raise ValueError("raw_state must be a dictionary")
        if not isinstance(self.normalized_state, dict):
            raise ValueError("normalized_state must be a dictionary")


@dataclass
class Action:
    """   """
    type: str  #     ，  "key_press", "key_release", "mouse_click", "mouse_move"
    key: Optional[str] = None  #     ，  "w", "a", "s", "d", "space"
    mouse_button: Optional[int] = None  #     ，0=  , 1=  , 2=  
    mouse_pos: Optional[tuple] = None  #      (x, y)
    duration: Optional[float] = None  #       （ ）
    metadata: Optional[Dict[str, Any]] = None  #      

    def __post_init__(self):
        """    """
        if self.type not in ["key_press", "key_release", "mouse_click", "mouse_move"]:
            raise ValueError(f"Invalid action type: {self.type}")

        if self.type.startswith("key") and self.key is None:
            raise ValueError("Key actions require 'key' parameter")

        if self.type.startswith("mouse") and self.mouse_pos is None:
            raise ValueError("Mouse actions require 'mouse_pos' parameter")

        if self.type == "mouse_click" and self.mouse_button is None:
            raise ValueError("Mouse click actions require 'mouse_button' parameter")