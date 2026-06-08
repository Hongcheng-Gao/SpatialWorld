"""
Environment Adapter Abstract Base Class
Defines unified interface that all environments must implement
   spatial-planning/envs/base.py
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from core.llm.schemas import EnvObservation


class BaseEnv(ABC):
    """Environment Adapter Abstract Base Class
    
    All concrete environments (AI2-THOR, ProcTHOR, CARLA, etc.) must implement this interface
    to ensure the Agent can seamlessly switch between different environments.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize environment
        
        Args:
            config: Environment configuration dictionary
        """
        self.config = config or {}
        self.step_counter = 0
        self.action_sequence = []
    
    @abstractmethod
    def reset(self, task_description: str, scene: Optional[str] = None) -> EnvObservation:
        """Reset environment to initial state
        
        Args:
            task_description: Task description
            scene: Optional scene name or scene object
            
        Returns:
            Initial observation
        """
        pass
    
    @abstractmethod
    def step_with_action_dict(self, action_dict: dict) -> Tuple[EnvObservation, Optional[str]]:
        """Execute one step using action dictionary
        
        Args:
            action_dict: Action dictionary containing action_type, action_name, object_type
            
        Returns:
            (observation, error_message) tuple
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close environment and release resources"""
        pass
    
    def get_action_sequence(self) -> str:
        """Get string representation of action sequence
        
        Returns:
            Action sequence string
        """
        if not self.action_sequence:
            return "(no action records)"
        return "->".join(self.action_sequence)
    
    @abstractmethod
    def _generate_text_state(self, metadata: Any) -> str:
        """Generate text state description
        
        Args:
            metadata: Environment metadata
            
        Returns:
            Text state description
        """
        pass
