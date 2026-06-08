"""Base environment interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from mllm_base_agent.llm.schemas import EnvObservation


class BaseEnv(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.step_counter = 0
        self.action_sequence = []

    @abstractmethod
    def reset(self, task_description: str, scene: Optional[str] = None) -> EnvObservation:
        raise NotImplementedError

    @abstractmethod
    def step_with_action_dict(self, action_dict: dict) -> Tuple[EnvObservation, Optional[str]]:
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    def get_action_sequence(self) -> str:
        if not self.action_sequence:
            return '(no action records)'
        return '->'.join(self.action_sequence)

    @abstractmethod
    def _generate_text_state(self, metadata: Any) -> str:
        raise NotImplementedError
