"""
         
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from .data_classes import FrameData, GameState, Action


class GameInputSource(ABC):
    """         """

    def __init__(self):
        """      """
        self.capabilities = self._get_capabilities()

    @abstractmethod
    def _get_capabilities(self) -> Dict[str, bool]:
        """
                  

        Returns:
            Dict[str, bool]:     ，     ：
                - "real_time":       
                - "supports_actions":         
                - "supports_state_reading":         
                - "supports_reset":       
                - "headless":         
        """
        pass

    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """
             

        Args:
            **kwargs:      

        Returns:
            bool:        
        """
        pass

    @abstractmethod
    def capture_frame(self) -> Optional[FrameData]:
        """
               

        Returns:
            Optional[FrameData]:      ，        None
        """
        pass

    @abstractmethod
    def execute_action(self, action: Action) -> bool:
        """
            

        Args:
            action:     

        Returns:
            bool:         
        """
        pass

    def get_game_state(self) -> Optional[GameState]:
        """
              （    ）

        Returns:
            Optional[GameState]:       ，       None
        """
        return None

    def reset_game(self) -> bool:
        """
            （    ）

        Returns:
            bool:       
        """
        return False

    @abstractmethod
    def close(self):
        """
                 
        """
        pass

    def is_ready(self) -> bool:
        """
                   

        Returns:
            bool:       
        """
        return True

    def get_info(self) -> Dict[str, Any]:
        """
               

        Returns:
            Dict[str, Any]:     
        """
        return {
            "capabilities": self.capabilities,
            "type": self.__class__.__name__,
        }